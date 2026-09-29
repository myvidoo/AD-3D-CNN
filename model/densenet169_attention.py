# -*- coding: utf-8 -*-
"""
Model definition module: 3D DenseNet-169 + switchable feature recalibration modules
(CBAM / ECA / axis spatial gating)

Source: extracted verbatim from cnn_model_v3_7_d169_group_comparison.py (lines 251-489),
     with two harmless changes only:
       1. LABEL_SMOOTHING moved from a global constant to a parameter of get_loss_function()
          (default 0.05);
       2. removed the dependency on the module-level device global (get_model already accepts a
          device parameter).

This module has zero file-path dependencies and depends only on torch and monai, so it can be
imported and used standalone.

Best model (Run 128) configuration: attention_type='axial' (axis spatial gating,
AxisSpatialGating3D), dropout_rate=0.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import monai
from monai.networks.nets import densenet169
from monai.losses import FocalLoss


# ==================== Attention modules ====================#

class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation Block"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        reduction_channels = max(1, in_channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduction_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention3D(nn.Module):
    """3D CBAM spatial attention module: Max/AvgPool along the channel dimension first, then a 7x7x7 convolution"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class CBAM3D(nn.Module):
    """3D CBAM attention: channel attention first, then spatial attention"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM3D, self).__init__()
        self.channel_att = SEBlock3D(in_channels=in_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention3D(kernel_size=7)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ECANet3D(nn.Module):
    """3D ECA attention: replaces the fully connected layer with a 1D convolution, keeping lightweight cross-channel interaction"""
    def __init__(self, in_channels, kernel_size=3):
        super(ECANet3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = y.unsqueeze(1)
        y = self.conv(y).squeeze(1)
        y = self.sigmoid(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class AxisSpatialGating3D(nn.Module):
    """3D axis spatial gating: channel compression followed by per-axis (D/H/W)
    mean pooling and sigmoid gating, applied multiplicatively to the input.

    Note: this module implements "axis spatial gating" rather than axial self-attention.
    Its mechanism is: first compress the channels with a 1×1×1 convolution, then perform
    spatial mean pooling along each of the depth/height/width axes to obtain per-axis
    descriptors, produce per-axis gating weights via a 1×1×1 convolution and a sigmoid, and
    finally recalibrate the feature map along the axes by element-wise multiplication.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(AxisSpatialGating3D, self).__init__()
        self.in_channels = in_channels
        mid_channels = max(8, in_channels // reduction_ratio)
        self.conv = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.attn_d = nn.Conv3d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.attn_h = nn.Conv3d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.attn_w = nn.Conv3d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compress the channels first, then generate per-axis gating weights along D/H/W
        x_compressed = self.act(self.bn(self.conv(x)))

        d_attn = x_compressed.mean(dim=[3, 4], keepdim=True)
        h_attn = x_compressed.mean(dim=[2, 4], keepdim=True)
        w_attn = x_compressed.mean(dim=[2, 3], keepdim=True)

        a_d = self.sigmoid(self.attn_d(d_attn))
        a_h = self.sigmoid(self.attn_h(h_attn))
        a_w = self.sigmoid(self.attn_w(w_attn))

        return x * a_d * a_h * a_w


# ==================== Backbone network ====================#

class densenet169WithAttention(nn.Module):
    """densenet169 with a switchable attention mechanism"""
    def __init__(self, spatial_dims, in_channels, num_classes, attention_type='none', pretrained=False):
        super(densenet169WithAttention, self).__init__()
        self.backbone = densenet169(
            spatial_dims=spatial_dims, in_channels=in_channels,
            out_channels=num_classes, pretrained=pretrained
        )
        self.features = self.backbone.features
        self.attention_type = str(attention_type).lower()

        # Compute the feature channel count dynamically to avoid hard-coding
        self.features.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, in_channels, 64, 64, 64)
            out = self.features(dummy)
            final_conv_channels = out.shape[1]

        self.attention = self._make_attention(final_conv_channels, self.attention_type)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(final_conv_channels, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def _make_attention(self, channels, attention_type):
        if attention_type == 'none':
            return nn.Identity()
        if attention_type == 'se':
            return SEBlock3D(in_channels=channels, reduction_ratio=16)
        if attention_type == 'cbam':
            return CBAM3D(in_channels=channels, reduction_ratio=16)
        if attention_type == 'eca':
            return ECANet3D(in_channels=channels, kernel_size=3)
        if attention_type == 'axial':
            # The parameter string 'axial' is a historical name (kept consistent with the
            # config field in all_training_results.csv and in the weight checkpoints); the
            # actual module is axis spatial gating.
            return AxisSpatialGating3D(in_channels=channels, reduction_ratio=32)
        raise ValueError(f"Unsupported attention type: {attention_type}")

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ModelWithDropout(nn.Module):
    """Model wrapper with Dropout"""
    def __init__(self, base_model, dropout_rate=0.3):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        features = self.base_model(x)
        if self.dropout is not None:
            features = self.dropout(features)
        return features


# ==================== Factory functions ====================#

def get_model(model_name, num_classes, device, dropout_rate=0.0, attention_type='none'):
    """Get the specified model"""
    model_name = model_name.lower()
    attention_type = str(attention_type).lower()
    base_model = None

    if model_name == "resnet10":
        base_model = monai.networks.nets.resnet10(
            spatial_dims=3, n_input_channels=1, num_classes=num_classes, pretrained=False,
        )
    elif model_name == "resnet18":
        base_model = monai.networks.nets.resnet18(
            spatial_dims=3, n_input_channels=1, num_classes=num_classes, pretrained=False,
        )
    elif model_name == "densenet169":
        if attention_type != 'none':
            base_model = densenet169WithAttention(
                spatial_dims=3,
                in_channels=1,
                num_classes=num_classes,
                attention_type=attention_type,
                pretrained=False
            )
        else:
            base_model = densenet169(
                spatial_dims=3, in_channels=1, out_channels=num_classes, pretrained=False
            )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = ModelWithDropout(base_model, dropout_rate=dropout_rate)
    return model.to(device)


def get_loss_function(loss_name, weight=None, label_smoothing=0.05):
    """Get the loss function"""
    loss_name = loss_name.lower()
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    elif loss_name == "ce_ls":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weight)
    elif loss_name == "focal":
        return FocalLoss(
            to_onehot_y=True, use_softmax=True,
            weight=weight, gamma=2.0, alpha=0.25,
        )
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def get_scheduler(scheduler_name, optimizer, num_epochs, train_loader_len):
    """Get the learning-rate scheduler"""
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosine":
        # Standard cosine annealing, decaying from the initial lr to 1e-7 over 200 epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,  # period length = total epochs
            eta_min=1e-7  # minimum learning rate
        )
        scheduler.step_frequency = "epoch"

    elif scheduler_name == "cosine_half":
        # Cosine annealing, but completing only half a period (faster decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
        scheduler.step_frequency = "epoch"

    elif scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=4,            # lower the learning rate if the monitored metric does not improve for 4 consecutive epochs
            threshold=1e-5,        # the AUC improvement must exceed 0.00001
            threshold_mode='rel',  # relative threshold (default); 'abs' may also be used
            cooldown=0,            # wait 0 epochs after lowering the learning rate before resuming monitoring
            min_lr=1e-7,
            eps=1e-8,              # small value to prevent division by zero
        )
        scheduler.step_frequency = "epoch"

    else:
        scheduler = None

    return scheduler
