# -*- coding: utf-8 -*-
"""
Grid-search optimised edition: 3D CNN training with MONAI on a three-class 3D MRI dataset
Based on the 4.1.4VS fixed edition, with grid-search functionality added
[FIX] Resolved the autocast argument conflict in newer PyTorch versions
[NEW] Added the ScaleIntensityRanged a_max parameter to the grid search
[NEW] Training results are saved locally to a log, for later analysis
[NEW] Fixed test-set split; added multi-sample frequency visualisation
Version 3.5 changelog: revised the model-selection logic of choose mode to consider the stability of validation AUC, F1 and ACC jointly, avoiding overfitting-driven selection caused by fluctuation in a single metric.
Version 3.6 revised the group-level comparison method, using a more explicit "Saliency Maps Aggregation".
"""
# Step one: global warning filtering (must be at the very top of the file, before any other business logic)
# ==============================================================================
import warnings
import sys

# 1. Ignore all requests-related warnings (fixes the urllib3/chardet version mismatch)
warnings.filterwarnings("ignore", category=Warning, module="requests")

# 2. Ignore PyTorch deprecation warnings (fixes the autocast deprecated issue)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.cuda.amp.autocast.*")

# 3. Ignore other common noise
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")
# Fix the OpenMP conflict issue

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 


import random
import gc
import ast
import re
import csv
import json
import math
import textwrap
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Mixed-precision training is no longer used
import monai
from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandRotated, RandFlipd, RandZoomd, RandGaussianNoised, RandAdjustContrastd, EnsureTyped,
)
from monai.losses import FocalLoss
from monai.networks.nets import densenet169
from monai.optimizers import Novograd
from monai.optimizers.lr_scheduler import WarmupCosineSchedule, LinearLR, ExponentialLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, roc_auc_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import seaborn as sns
import itertools
from datetime import datetime
import optuna  # NEW: Bayesian optimisation library

# ==================== Hyperparameter grid definition ====================#

# LR scheduler options: "none", "cosine", "cosine_half", "reduce_on_plateau"
SCHEDULER_OPTIONS = ["reduce_on_plateau"]

# Loss-function options: "ce", "ce_ls", "focal"
LOSS_OPTIONS = ["ce"]

# Model configuration   "densenet169"
MODEL_OPTIONS = ["densenet169"]

# Attention mechanism (only effective for densenet169), options: ["none", "se", "cbam", "eca", "axial"]
ATTENTION_OPTIONS = [ "cbam"]

# Dropout rate  [0.0, 0.2]
DROPOUT_OPTIONS = [0.2]

# Gradient-accumulation steps; with a smaller batch size, gradient accumulation can emulate a larger batch size. Options: [1,2], etc.
GRAD_ACCUM_OPTIONS = [1]

# Data-augmentation intensity. Options: 0 (no augmentation), 1 (mild augmentation), 2 (strong augmentation)
AUGMENTATION_OPTIONS = [0]

# Training parameters[2,3,4,5]
BATCH_SIZE_OPTIONS = [5]

# Learning-rate settings[2e-5, 5e-5, 1e-4]
LEARNING_RATE_OPTIONS = [5e-5]



EPOCH_OPTIONS = [ 200 ]

# [NEW] Value range of the ScaleIntensityRanged a_max parameter
A_MAX_OPTIONS = [1100]
DEFAULT_A_MAX = A_MAX_OPTIONS[0]

# Dataset split ratios
# ✅ Recommendation: keep the definitions, but use only the default split configuration in the grid search
DATA_SPLIT_OPTIONS = [
    {"train": 0.7, "val": 0.15, "test": 0.15}
   # ,{"train": 0.8, "val": 0.1, "test": 0.1}
]
DATA_SPLIT = DATA_SPLIT_OPTIONS[0]

# Global result file paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results3.7")
GLOBAL_TRAINING_RESULTS_CSV = os.path.join(RESULTS_DIR, "all_training_results.csv")
FIXED_DATA_SPLIT_JSON = os.path.join(os.path.dirname(__file__), "fixed_data_split.json")
TEST_EVALUATION_OUTPUT_DIR = os.path.join(RESULTS_DIR, "test_evaluation_results")
MODEL_SELECTION_OUTPUT_DIR = os.path.join(RESULTS_DIR, "model_selection_results")
GROUP_COMPARISON_OUTPUT_DIR = os.path.join(RESULTS_DIR, "group_level_comparison")

# ==================== Fixed parameters ====================#

# Data paths
EXCEL_PATH = r"<AUTHOR_DATA_ROOT>/MONAILabel/AD\NIfTI_organized_monailabel_mask_process_ants_20260525_fast_ws.xlsx"

# Classification settings
CLASS_NAMES = ["Class_0", "Class_1", "Class_2"]
LABEL_MAP_FLOAT_TO_INT = {0.0: 0, 0.5: 1, 1.0: 2}
NUM_CLASSES = 3
INPUT_SHAPE = (182, 218, 182)

# Training control
NUM_WORKERS = 4  # temporarily set to 0 to avoid DataLoader worker crashes; in production this can be set to min(4, os.cpu_count())
# How many epochs between regular checkpoint saves
CHECKPOINT_INTERVAL = 20
# Early-stopping patience (stop if auc, f1 and acc all fail to improve within this window)
EARLY_STOP_PATIENCE = 15
LABEL_SMOOTHING = 0.05

# Random seed (to further improve reproducibility)
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
# Fix the cuDNN random seed to reduce nondeterminism, at the cost of possibly sacrificing some performance
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(f"Using device: {device}")
# if torch.cuda.is_available():

#     print(f"GPU: {torch.cuda.get_device_name(0)}")    
#    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
# ==================== Data-augmentation parameter dictionary ====================#

# ==================== Augmentation parameters optimised for registered MRI ====================#
AUGMENTATION_PARAMS = {
    0: {
        "rot_range": 0.0, "rot_prob": 0.0, "flip_lr_prob": 0.0,
        "zoom_range": (1.0, 1.0), "zoom_prob": 0.0,
        "noise_std": 0.0, "noise_prob": 0.0,
        "contrast_gamma": (1.0, 1.0), "contrast_prob": 0.0,
    },
    1: {  # very mild augmentation (preferred for registered data)
        "rot_range": 0.035, # about 2 degrees, a tiny perturbation
        "rot_prob": 0.2, 
        "flip_lr_prob": 0.5, # left-right flip only
        "zoom_range": (0.99, 1.01), "zoom_prob": 0.1,
        "noise_std": 0.005, "noise_prob": 0.1,
        "contrast_gamma": (0.98, 1.02), "contrast_prob": 0.1, # slight contrast adjustment
    },
    2: {  # mild augmentation
        "rot_range": 0.07,  # about 4 degrees
        "rot_prob": 0.3, 
        "flip_lr_prob": 0.5,
        "zoom_range": (0.97, 1.03), "zoom_prob": 0.2,
        "noise_std": 0.01, "noise_prob": 0.15,
        "contrast_gamma": (0.95, 1.05), "contrast_prob": 0.15,
    },
    # Augmentation intensity 3 is not recommended, as it degrades the ANTs registration result
}

# Worker init for DataLoader, for reproducibility across worker processes
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# ==================== Utility functions ====================#

def get_data_transforms(augmentation_intensity, a_max=1100):
    """Get the data transforms"""
    # If the requested intensity is out of range, force it down to protect the registration result
    if augmentation_intensity > 2:
        augmentation_intensity = 2
        
    current_aug_params = AUGMENTATION_PARAMS[augmentation_intensity]

    train_transforms = Compose([
        LoadImaged(keys=["image"], ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
        
        # The white-matter peak is at 1000 here, so this clipping is a perfect fit
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=a_max, b_min=0.0, b_max=1.0, clip=True
        ),

        RandRotated(
            keys=["image"], range_x=current_aug_params['rot_range'],
            range_y=current_aug_params['rot_range'], range_z=current_aug_params['rot_range'],
            prob=current_aug_params['rot_prob'], keep_size=True, padding_mode='zeros'
        ),
        
        # [Change 2] Superior-inferior (1) and anterior-posterior (2) flips are strictly forbidden; only left-right (0) flips are allowed
        # Note: confirm that axis 0 of your data is the left-right (L-R) axis. In the MNI standard, axis 0 is usually L-R.
        # RandFlipd(keys=["image"], prob=current_aug_params['flip_lr_prob'], spatial_axis=0),
        
        RandZoomd(
            keys=["image"], prob=current_aug_params['zoom_prob'],
            min_zoom=current_aug_params['zoom_range'][0],
            max_zoom=current_aug_params['zoom_range'][1],
            keep_size=True, padding_mode='constant', constant_values=0
        ),
        RandGaussianNoised(
            keys=["image"], prob=current_aug_params['noise_prob'],
            mean=0.0, std=current_aug_params['noise_std']
        ),
        RandAdjustContrastd(
            keys=["image"], prob=current_aug_params['contrast_prob'],
            gamma=current_aug_params['contrast_gamma']
        ),
        EnsureTyped(keys=["image"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image"], ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=a_max, b_min=0.0, b_max=1.0, clip=True
        ),
        EnsureTyped(keys=["image"]),
    ])

    return train_transforms, val_transforms
# ==================== Model definition ====================#

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
    """3D CBAM spatial attention module: first Max/AvgPool along the channel dimension, then a 7x7x7 convolution"""
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
    """3D ECA attention: replaces the fully connected layer with a 1D convolution, keeping cross-channel interaction lightweight"""
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

class AxialAttention3D(nn.Module):
    """3D Axial Attention: cross-axis attention along depth, height and width separately."""
    def __init__(self, in_channels, reduction_ratio=16):
        super(AxialAttention3D, self).__init__()
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
        # Compress the channels first, then generate attention maps along the D/H/W axes separately
        x_compressed = self.act(self.bn(self.conv(x)))

        d_attn = x_compressed.mean(dim=[3, 4], keepdim=True)
        h_attn = x_compressed.mean(dim=[2, 4], keepdim=True)
        w_attn = x_compressed.mean(dim=[2, 3], keepdim=True)

        a_d = self.sigmoid(self.attn_d(d_attn))
        a_h = self.sigmoid(self.attn_h(h_attn))
        a_w = self.sigmoid(self.attn_w(w_attn))

        return x * a_d * a_h * a_w

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

        # Compute the number of feature channels dynamically to avoid hard-coding
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
            return AxialAttention3D(in_channels=channels, reduction_ratio=32)
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

def get_loss_function(loss_name, weight=None):
    """Get the loss function"""
    loss_name = loss_name.lower()
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    elif loss_name == "ce_ls":
        return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, weight=weight)
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
            T_max=num_epochs,  # cycle length = total epochs
            eta_min=1e-7  # minimum learning rate
        )
        scheduler.step_frequency = "epoch"
    
    elif scheduler_name == "cosine_half":
        # Cosine annealing but completing only half a cycle (a faster decay)
        # Complete the decay from max_lr to eta_min within 200 epochs
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
            patience=4,            # lower the LR if the monitored metric does not improve for 4 consecutive epochs
            threshold=1e-5,        # the AUC improvement must exceed 0.00001
            threshold_mode='rel',  # relative threshold (default); 'abs' is also available
            cooldown=0,            # wait 0 epochs after lowering the LR before resuming monitoring
            min_lr=1e-7,
            eps=1e-8,              # small value to prevent division by zero
        )
        scheduler.step_frequency = "epoch"

    else:
        scheduler = None
    
    return scheduler
# ==================== Training and evaluation functions ====================#

def train_epoch(model, loader, optimizer, loss_func, device, scheduler=None,
                grad_accum_steps=1, writer=None, epoch=0):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    total_samples = 0

    optimizer.zero_grad()
    progress_bar = tqdm(loader, desc="Train", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        inputs, labels = batch["image"].to(device), batch["label"].to(device)

        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()

        # Update the parameters only once the accumulation step count is reached
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            if scheduler and hasattr(scheduler, 'step_frequency') and scheduler.step_frequency == "step":
                scheduler.step()

        epoch_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        current_lr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})

    # Handle the final incomplete accumulation step
    if (len(loader) % grad_accum_steps) != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler and hasattr(scheduler, 'step_frequency') and scheduler.step_frequency == "step":
            scheduler.step()

    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and \
            (not hasattr(scheduler, 'step_frequency') or scheduler.step_frequency == "epoch"):
        scheduler.step()

    avg_loss = epoch_loss / max(total_samples, 1)
    if writer is not None:
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('LR/train', optimizer.param_groups[0]["lr"], epoch)

    return avg_loss

def validate_epoch(model, loader, loss_func, device, num_classes, writer=None, epoch=0, tag="Val"):
    """Validate for one epoch"""
    model.eval()
    epoch_loss = 0
    y_true, y_pred, y_proba = [], [], []
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=tag, leave=False):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            epoch_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_proba.extend(probs.cpu().numpy())

    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)

    metrics = {
        'loss': epoch_loss / max(total_samples, 1),
        'accuracy': np.mean(y_true == y_pred)
    }

    try:
        if num_classes > 2:
            auc_scores = []
            for i in range(num_classes):
                mask_true = (y_true == i)
                if mask_true.sum() > 0 and (~mask_true).sum() > 0:
                    auc_score = roc_auc_score(mask_true.astype(int), y_proba[:, i])
                    auc_scores.append(auc_score)
                    metrics[f'auc_class_{i}'] = auc_score
                else:
                    metrics[f'auc_class_{i}'] = 0.5
            metrics['auc'] = np.mean(auc_scores) if auc_scores else 0.5
        else:
            metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1]) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        metrics['auc'] = 0.5
        for i in range(num_classes):
            metrics[f'auc_class_{i}'] = 0.5

    try:
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
        for i in range(num_classes):
            metrics[f'f1_class_{i}'] = float(f1_per_class[i]) if i < len(f1_per_class) else 0.0
    except Exception:
        metrics['f1_macro'] = 0.0
        metrics['f1_weighted'] = 0.0
        for i in range(num_classes):
            metrics[f'f1_class_{i}'] = 0.0

    if writer is not None:
        writer.add_scalar(f'Loss/{tag}', metrics['loss'], epoch)
        writer.add_scalar(f'AUC/{tag}', metrics['auc'], epoch)
        writer.add_scalar(f'F1/{tag}', metrics['f1_macro'], epoch)
        writer.add_scalar(f'Acc/{tag}', metrics['accuracy'], epoch)

    return metrics

def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, metrics, config=None, tag=None):
    """Save a checkpoint; a tag may be added to mark e.g. the best metric"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'current_lr': optimizer.param_groups[0]['lr'],
        'metrics': metrics,
        'model_name': config.get('model_name', 'grid_search_model') if config else 'grid_search_model',
        'num_classes': NUM_CLASSES,
        'config': config if config else {}
    }
    name = f"checkpoint_epoch_{epoch:03d}"
    if tag:
        name += f"_{tag}"
    path = os.path.join(checkpoint_dir, f"{name}.pth")
    torch.save(checkpoint, path)
    return path


def append_global_training_result(result, csv_path=GLOBAL_TRAINING_RESULTS_CSV):
    """Append each training result to the global CSV file."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    result_df = pd.DataFrame([result])
    if not os.path.exists(csv_path):
        result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    else:
        result_df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')


def save_fixed_data_split(test_split, train_ratio, val_ratio, test_ratio, random_state=42, file_path=FIXED_DATA_SPLIT_JSON):
    """Save the fixed test-set file list so that all subsequent evaluations use the same test split."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    payload = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'random_state': random_state,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'test_split': [
            {'file_path': item['image'], 'label': int(item['label'])} for item in test_split
        ]
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_fixed_data_split(file_path=FIXED_DATA_SPLIT_JSON):
    """Load the fixed test-set split information."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load fixed data split file: {e}")
        return None


def parse_config_from_string(config_str):
    if isinstance(config_str, dict):
        return config_str
    if not isinstance(config_str, str):
        return {}
    try:
        return ast.literal_eval(config_str)
    except Exception:
        try:
            return json.loads(config_str)
        except Exception:
            return {}


def parse_config_from_run_dir(run_dir):
    """Parse part of the training configuration from the run_dir directory name."""
    config = {}
    basename = os.path.basename(run_dir)
    patterns = {
        'model_name': r'GRID_SEARCH_\d+_([^_]+)',
        'dropout_rate': r'dropout_([0-9.]+)',
        'augmentation_intensity': r'aug_([0-9]+)',
        'batch_size': r'bs_([0-9]+)',
        'learning_rate': r'lr_([0-9.e-]+)',
        'amax': r'amax_([0-9]+)',
        'scheduler_name': r'sched_([^_]+)',
        'loss_name': r'loss_([^_]+)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, basename)
        if match:
            value = match.group(1)
            if key in ['batch_size', 'augmentation_intensity', 'amax']:
                config[key] = int(value)
            elif key == 'dropout_rate':
                config[key] = float(value)
            elif key == 'learning_rate':
                config[key] = float(value)
            else:
                config[key] = value
    return config


def resolve_run_config(run_dir, global_csv_path=GLOBAL_TRAINING_RESULTS_CSV):
    """Try to resolve the training configuration from several sources."""
    # 1) training_result.csv
    training_result_path = os.path.join(run_dir, 'training_result.csv')
    if os.path.exists(training_result_path):
        try:
            df = pd.read_csv(training_result_path)
            if 'config' in df.columns and len(df) > 0:
                config = parse_config_from_string(df.iloc[0]['config'])
                if config:
                    return config
        except Exception:
            pass

    # 2) global CSV
    if os.path.exists(global_csv_path):
        try:
            df = pd.read_csv(global_csv_path)
            row = df[df['run_dir'] == run_dir]
            if len(row) > 0:
                config = parse_config_from_string(row.iloc[0]['config'])
                if config:
                    return config
        except Exception:
            pass

    # 3) fallback: parse the directory name
    return parse_config_from_run_dir(run_dir)


def find_checkpoint_by_tag(run_dir, tag):
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for name in os.listdir(checkpoint_dir):
        if name.endswith(f'_{tag}.pth') or name.endswith(f'_{tag}.pt'):
            candidates.append(os.path.join(checkpoint_dir, name))
    if not candidates:
        return None
    # choose latest by epoch number if multiple
    candidates.sort()
    return candidates[-1]

# ==================== Single-training-run execution function ====================#

def run_single_training(config, run_id):
    """Run a single training configuration"""
    print(f"\n{'='*80}")
    print(f"RUN {run_id}: {config}")
    print(f"{'='*80}")

    # Unpack the configuration
    model_name = config['model_name']
    scheduler_name = config['scheduler_name']
    loss_name = config['loss_name']
    attention_type = config.get('attention_type', 'none')
    use_attention = attention_type != 'none'
    dropout_rate = config['dropout_rate']
    grad_accum_steps = config['grad_accum_steps']
    augmentation_intensity = config['augmentation_intensity']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    train_ratio = config['train_ratio']
    val_ratio = config['val_ratio']
    test_ratio = config['test_ratio']
    a_max = config['a_max']  # NEW: unpack a_max

    # Create the run directory
    attn_tag = f"_Attn_{attention_type}" if use_attention and model_name == "densenet169" else ""
    # [Change] add a_max to the directory name
    run_dir = f"<AUTHOR_DATA_ROOT>/MONAILabel/cnn_Grid_3.7.2_cbam/Grid_SEARCH_{run_id}_{model_name}{attn_tag}_dropout_{dropout_rate}_aug_{augmentation_intensity}_bs_{batch_size}_lr_{learning_rate:.0e}_amax_{a_max}_sched_{scheduler_name}_loss_{loss_name}"
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = None
    try:
        writer = SummaryWriter(log_dir=log_dir)
        # 1. Load the data
        df = pd.read_excel(EXCEL_PATH)
        df = df.dropna(subset=['file_path', 'label'])
        df['label_int'] = df['label'].map(LABEL_MAP_FLOAT_TO_INT)
        data_list = []
        for _, row in df.iterrows():
            if os.path.exists(row["file_path"]):
                data_list.append({
                    "image": row["file_path"],
                    "label": int(row["label_int"])
                })
        all_labels = [item["label"] for item in data_list]

        # 2. Dataset split
        fixed_split = load_fixed_data_split()
        if fixed_split and 'test_split' in fixed_split:
            fixed_test_paths = {item['file_path'] for item in fixed_split['test_split']}
            test_data = [item for item in data_list if item['image'] in fixed_test_paths]
            remaining_data = [item for item in data_list if item['image'] not in fixed_test_paths]
            remaining_labels = [item['label'] for item in remaining_data]
            if len(test_data) >= max(1, int(len(data_list) * test_ratio)):
                print(f"✓ Using existing fixed test split from {FIXED_DATA_SPLIT_JSON}")
                train_val_data, train_val_labels = remaining_data, remaining_labels
            else:
                print("⚠️ Existing fixed split incomplete or mismatched; regenerating split from current dataset")
                fixed_split = None

        if fixed_split is None:
            train_val_data, test_data, train_val_labels, test_labels = train_test_split(
                data_list, all_labels, test_size=test_ratio, stratify=all_labels, random_state=42
            )
            save_fixed_data_split(
                test_split=test_data,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_state=42,
            )
            print(f"✓ Fixed test split saved to: {FIXED_DATA_SPLIT_JSON}")
            val_ratio_adj = val_ratio / (1 - test_ratio)
            train_data, val_data, train_labels, val_labels = train_test_split(
                train_val_data, train_val_labels, test_size=val_ratio_adj, stratify=train_val_labels, random_state=42
            )
        else:
            val_ratio_adj = val_ratio / (1 - test_ratio)
            train_data, val_data, train_labels, val_labels = train_test_split(
                train_val_data, train_val_labels, test_size=val_ratio_adj, stratify=train_val_labels, random_state=42
            )

        # 3. Data transforms
        # [Change] pass the a_max parameter
        train_transforms, val_transforms = get_data_transforms(augmentation_intensity, a_max=a_max)
        # Use the same preprocessing at test time as at validation time
        test_transforms = val_transforms

        # 4. Create the data loaders
        train_ds = Dataset(data=train_data, transform=train_transforms)
        val_ds = Dataset(data=val_data, transform=val_transforms)
        test_ds = Dataset(data=test_data, transform=test_transforms)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
            pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(GLOBAL_SEED)
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False,
            worker_init_fn=worker_init_fn
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False,
            worker_init_fn=worker_init_fn
        )

        # 5. Compute the class weights
        class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * NUM_CLASSES
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)

        # 6. Initialise the model
        model = get_model(model_name, NUM_CLASSES, device, dropout_rate, attention_type=attention_type)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {model_name}, Params: {total_params:,}, Attention: {attention_type}")

        # 7. Loss function and optimiser
        loss_func = get_loss_function(loss_name, weight=class_weights_tensor).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4,
            betas=(0.9, 0.999), eps=1e-8
        )

        # 8. Scheduler
        scheduler = get_scheduler(scheduler_name, optimizer, num_epochs, len(train_loader))

        # 10. Training loop
        epoch_csv_path = os.path.join(run_dir, "epoch_metrics.csv")
        epoch_headers = [
            'run_id', 'epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_auc', 'val_f1_macro',
            'learning_rate'
        ]
        for i in range(NUM_CLASSES):
            epoch_headers.append(f'f1_class_{i}')
        for i in range(NUM_CLASSES):
            epoch_headers.append(f'auc_class_{i}')
        epoch_headers.extend(['best_acc_so_far', 'best_auc_so_far', 'best_f1_so_far'])

        with open(epoch_csv_path, 'w', newline='', encoding='utf-8-sig') as epoch_file:
            epoch_writer = csv.writer(epoch_file)
            epoch_writer.writerow(epoch_headers)

            best_auc, best_f1, best_acc = 0.0, 0.0, 0.0
            best_auc_path = best_f1_path = best_acc_path = None
            patience_counter = 0

            for epoch in range(num_epochs):
                train_loss = train_epoch(
                    model, train_loader, optimizer, loss_func, device, scheduler,
                    grad_accum_steps, writer=writer, epoch=epoch
                )
                val_metrics = validate_epoch(
                    model, val_loader, loss_func, device, NUM_CLASSES, writer=writer, epoch=epoch, tag="Val"
                )

                if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['auc'])

                current_auc = val_metrics['auc']
                current_f1 = val_metrics['f1_macro']
                current_acc = val_metrics['accuracy']

                improved = False
                # Update and save the best model (delete the old file first if one exists)
                if current_auc > best_auc:
                    best_auc = current_auc
                    improved = True
                    # Delete the old best-AUC file
                    if best_auc_path and os.path.exists(best_auc_path):
                        os.remove(best_auc_path)
                    best_auc_path = save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_dir, val_metrics, config=config, tag='best_auc')
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    improved = True
                    if best_f1_path and os.path.exists(best_f1_path):
                        os.remove(best_f1_path)
                    best_f1_path = save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_dir, val_metrics, config=config, tag='best_f1')
                if current_acc > best_acc:
                    best_acc = current_acc
                    # Only save the best model; this does not take part in the early-stopping decision
                    if best_acc_path and os.path.exists(best_acc_path):
                        os.remove(best_acc_path)
                    best_acc_path = save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_dir, val_metrics, config=config, tag='best_acc')

                # Early-stopping check: triggered when neither AUC nor F1 improves within EARLY_STOP_PATIENCE
                if improved:
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1} (no improvement in last {EARLY_STOP_PATIENCE} epochs)")
                    break

                # Periodically save a checkpoint
                if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_dir, val_metrics, config=config)

                if epoch % 20 == 0 or epoch == num_epochs - 1:
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | Loss={train_loss:.4f} | Val AUC={current_auc:.4f} | "
                        f"F1={current_f1:.4f} | Acc={current_acc:.4f} | "
                        f"Best(AUC={best_auc:.4f}, F1={best_f1:.4f}, Acc={best_acc:.4f})"
                    )

                epoch_row = [
                    run_id,
                    epoch + 1,
                    train_loss,
                    val_metrics['loss'],
                    val_metrics['accuracy'],
                    val_metrics['auc'],
                    val_metrics['f1_macro'],
                    optimizer.param_groups[0]['lr'],
                ]
                for i in range(NUM_CLASSES):
                    epoch_row.append(val_metrics.get(f'f1_class_{i}', 0.0))
                for i in range(NUM_CLASSES):
                    epoch_row.append(val_metrics.get(f'auc_class_{i}', 0.0))
                epoch_row.extend([best_acc, best_auc, best_f1])
                epoch_writer.writerow(epoch_row)

        # Save the final results
        final_metrics = validate_epoch(model, val_loader, loss_func, device, NUM_CLASSES)

        result = {
            'run_id': run_id,
            'config': config,
            'best_auc': best_auc,
            'best_f1': best_f1,
            'best_acc': best_acc,
            'final_auc': final_metrics['auc'],
            'final_f1': final_metrics['f1_macro'],
            'final_acc': final_metrics['accuracy'],
            'total_params': total_params,
            'run_dir': run_dir,
            'status': 'completed'
        }

        # Save the results to CSV
        result_df = pd.DataFrame([result])
        result_path = os.path.join(run_dir, "training_result.csv")
        result_df.to_csv(result_path, index=False)
        append_global_training_result(result)

        print(f"✓ Run {run_id} completed. Best AUC: {best_auc:.4f}, F1: {best_f1:.4f}, Acc: {best_acc:.4f}")
        return result

    except Exception as e:
        print(f"❌ Run {run_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'run_id': run_id,
            'config': config,
            'status': 'failed',
            'error': str(e)
        }

    finally:
        # Free GPU memory and close the logging
        if writer is not None:
            writer.close()

        # Clean up objects to prevent memory leaks
        try:
            del model, optimizer, scheduler
            del train_loader, val_loader, test_loader
            del train_ds, val_ds, test_ds
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
def safe_torch_load(model_path, map_location=device):
    """Checkpoint loading compatible with different PyTorch versions"""
    try:
        return torch.load(model_path, map_location=map_location)
    except Exception:
        return torch.load(model_path, map_location=map_location, weights_only=False)


def _coerce_config_value(value):
    """Restore configuration values from CSV/strings to their original types as far as possible"""
    if isinstance(value, np.generic):
        value = value.item()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, str):
        text = value.strip()
        lower = text.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower in {"none", ""}:
            return None
        try:
            if any(ch in lower for ch in [".", "e"]):
                return float(text)
            return int(text)
        except Exception:
            return text

    return value


def normalize_inference_config(config=None):
    """Fill in the configuration required for inference and normalise the types"""
    defaults = {
        'model_name': 'densenet169',
        'scheduler_name': 'cosine',
        'loss_name': 'ce',
        'attention_type': 'none',
        'use_amp': False,
        'dropout_rate': 0.0,
        'grad_accum_steps': 1,
        'augmentation_intensity': 0,
        'batch_size': 1,
        'learning_rate': 1e-4,
        'num_epochs': 0,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'a_max': 1122,
    }

    normalized = defaults.copy()
    for key, value in (config or {}).items():
        parsed = _coerce_config_value(value)
        if parsed is not None:
            normalized[key] = parsed

    normalized['model_name'] = str(normalized['model_name']).lower()
    normalized['scheduler_name'] = str(normalized['scheduler_name']).lower()
    normalized['loss_name'] = str(normalized['loss_name']).lower()
    normalized['attention_type'] = str(normalized.get('attention_type', 'none')).lower()
    if normalized.get('attention_type', 'none') == 'none' and 'use_attention' in (config or {}):
        if bool(_coerce_config_value(config['use_attention'])):
            normalized['attention_type'] = 'se'
    normalized['use_amp'] = bool(normalized['use_amp'])
    normalized['dropout_rate'] = float(normalized['dropout_rate'])
    normalized['grad_accum_steps'] = int(float(normalized['grad_accum_steps']))
    normalized['augmentation_intensity'] = int(float(normalized['augmentation_intensity']))
    normalized['batch_size'] = int(float(normalized['batch_size']))
    normalized['learning_rate'] = float(normalized['learning_rate'])
    normalized['num_epochs'] = int(float(normalized['num_epochs']))
    normalized['train_ratio'] = float(normalized['train_ratio'])
    normalized['val_ratio'] = float(normalized['val_ratio'])
    normalized['test_ratio'] = float(normalized['test_ratio'])
    normalized['a_max'] = float(normalized['a_max'])
    return normalized


def extract_config_from_training_result(run_dir):
    """Recover the training configuration from training_result.csv in the same directory"""
    result_csv = os.path.join(run_dir, "training_result.csv")
    if not os.path.exists(result_csv):
        return None

    try:
        df = pd.read_csv(result_csv)
        if df.empty:
            return None

        row = df.iloc[0].to_dict()
        if 'config' in row and isinstance(row['config'], str) and row['config'].strip():
            parsed = ast.literal_eval(row['config'])
            if isinstance(parsed, dict):
                return parsed

        flattened = {}
        for key, value in row.items():
            if str(key).startswith('config_'):
                flattened[key.replace('config_', '')] = value
        return flattened if flattened else None
    except Exception as e:
        print(f"Warning: failed to parse training_result.csv: {e}")
        return None


def parse_config_from_run_dir(run_dir):
    """Parse the configuration from the run directory name; handles old checkpoints that contain no config"""
    run_name = Path(run_dir).name
    pattern = (
        r"GRID_SEARCH_\d+_(?P<model>.+?)(?:_Attn_(?P<attention_type>[^_]+))?_dropout_(?P<dropout>-?[\d.]+)"
        r"_aug_(?P<aug>\d+)_bs_(?P<bs>\d+)_lr_(?P<lr>[\deE.+-]+)_amax_(?P<amax>[\d.]+)"
        r"_sched_(?P<sched>.+?)_loss_(?P<loss>.+)$"
    )
    match = re.search(pattern, run_name)
    if not match:
        return None

    attention_type = match.group('attention_type') or 'none'
    return {
        'model_name': match.group('model').lower(),
        'attention_type': attention_type.lower(),
        'dropout_rate': float(match.group('dropout')),
        'augmentation_intensity': int(match.group('aug')),
        'batch_size': int(match.group('bs')),
        'learning_rate': float(match.group('lr')),
        'a_max': float(match.group('amax')),
        'scheduler_name': match.group('sched').lower(),
        'loss_name': match.group('loss').lower(),
    }


def load_checkpoint_for_inference(checkpoint_path):
    """Load a checkpoint and automatically resolve the training hyperparameters"""
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_obj = safe_torch_load(checkpoint_path, map_location=device)
    run_dir = str(Path(checkpoint_path).resolve().parent.parent)

    config = None
    config_source = None
    if isinstance(checkpoint_obj, dict) and isinstance(checkpoint_obj.get('config'), dict) and checkpoint_obj.get('config'):
        config = checkpoint_obj['config']
        config_source = 'checkpoint.config'

    if not config:
        config = extract_config_from_training_result(run_dir)
        if config:
            config_source = 'training_result.csv'

    if not config:
        config = parse_config_from_run_dir(run_dir)
        if config:
            config_source = 'run directory name'

    config = normalize_inference_config(config)
    print(f"✓ Inference config resolved from: {config_source or 'default values'}")
    print(
        f"  model={config['model_name']}, attention_type={config['attention_type']}, "
        f"dropout={config['dropout_rate']}, a_max={config['a_max']}, aug={config['augmentation_intensity']}"
    )

    return checkpoint_obj, config, run_dir


def build_model_from_checkpoint(checkpoint_obj, config):
    """Rebuild the model from the resolved configuration and load the weights"""
    if isinstance(checkpoint_obj, nn.Module):
        model = checkpoint_obj.to(device)
        model.eval()
        return model

    model = get_model(
        model_name=config['model_name'],
        num_classes=NUM_CLASSES,
        device=device,
        dropout_rate=config['dropout_rate'],
        attention_type=config.get('attention_type', 'none')
    )

    if isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj.get('model_state_dict', checkpoint_obj)
        try:
            model.base_model.load_state_dict(state_dict, strict=True)
        except Exception:
            model.load_state_dict(state_dict, strict=False)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint_obj)}")

    model.eval()
    return model


def safe_build_model_from_checkpoint(checkpoint_path):
    checkpoint_obj, config, run_dir = load_checkpoint_for_inference(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint_obj, config)
    return model, config, run_dir


def load_single_nifti_sample(sample_path, a_max):
    """Load a single 3D NIfTI sample using the same preprocessing as validation/test"""
    sample_path = os.path.abspath(sample_path)
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    _, inference_transforms = get_data_transforms(augmentation_intensity=0, a_max=a_max)
    sample = inference_transforms({"image": sample_path})
    image_tensor = sample["image"]

    if hasattr(image_tensor, "as_tensor"):
        image_tensor = image_tensor.as_tensor()

    image_tensor = image_tensor.float()
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    volume = image_tensor[0].cpu().numpy()
    input_tensor = image_tensor.unsqueeze(0).to(device)
    return input_tensor, volume


def _replace_inplace_activations(module, inplace=False):
    """Replace inplace activation layers in the module with non-inplace ones to avoid autograd conflicts under Grad-CAM."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=inplace))
        elif isinstance(child, nn.LeakyReLU):
            setattr(module, name, nn.LeakyReLU(inplace=inplace))
        else:
            _replace_inplace_activations(child)

def prepare_model_for_gradcam(model):
    """Prepare a model instance that can be safely backpropagated for Grad-CAM."""
    if model is None:
        return None

    model.eval()
    base_model = model.base_model if hasattr(model, "base_model") else model
    if base_model is not None:
        _replace_inplace_activations(base_model, inplace=False)
    return model


class GradCAM3D:
    """Grad-CAM implementation for 3D CNNs (compatible with MONAI DenseNet / attention modules)"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            self.target_layer.register_forward_hook(self._forward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        # Keep the gradient graph so that activations are not detached early in Grad-CAM
        self.activations = output

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        with torch.enable_grad():
            outputs = self.model(input_tensor)

            if class_idx is None:
                class_idx = int(torch.argmax(outputs, dim=1).item())

            score = outputs[:, class_idx].sum()

            if self.activations is None:
                raise RuntimeError("Failed to capture activations for Grad-CAM.")

            # Use autograd.grad to compute the gradient of the target score with respect to the target layer
            grads = torch.autograd.grad(
                outputs=score,
                inputs=self.activations,
                retain_graph=False,
                allow_unused=False,
            )[0]

        self.gradients = grads.detach()

        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach(), outputs.detach(), class_idx


def get_gradcam_candidate_layers(model):
    """Return the dictionary of candidate layers in the model usable for Grad-CAM."""
    base_model = model.base_model if hasattr(model, 'base_model') else model
    candidate_layers = {}

    if hasattr(base_model, 'attention'):
        candidate_layers['attention'] = base_model.attention
        if hasattr(base_model.attention, 'spatial_att'):
            candidate_layers['attention.spatial_att'] = base_model.attention.spatial_att

    if hasattr(base_model, 'features'):
        feature_layers = list(base_model.features.children())
        candidate_layers['features[-1]'] = feature_layers[-1] if feature_layers else base_model.features

    return candidate_layers


def get_gradcam_target_layer(model, target_layer_choice=None):
    """Select the most suitable target layer for Grad-CAM."""
    base_model = model.base_model if hasattr(model, 'base_model') else model
    attention_type = getattr(base_model, 'attention_type', 'none') if hasattr(base_model, 'attention_type') else 'none'
    attention_type = str(attention_type).lower()

    candidate_layers = get_gradcam_candidate_layers(model)

    if target_layer_choice:
        if target_layer_choice in candidate_layers:
            return candidate_layers[target_layer_choice]
        raise ValueError(
            f"Requested Grad-CAM target layer '{target_layer_choice}' not found. "
            f"Available: {list(candidate_layers.keys())}"
        )

    if attention_type == 'cbam' and 'attention.spatial_att' in candidate_layers:
        return candidate_layers['attention.spatial_att']

    if attention_type in ('se', 'eca', 'axial') and 'features[-1]' in candidate_layers:
        return candidate_layers['features[-1]']

    if 'features[-1]' in candidate_layers:
        return candidate_layers['features[-1]']

    if 'attention' in candidate_layers:
        return candidate_layers['attention']

    raise ValueError("Could not locate a suitable Grad-CAM target layer.")


def apply_mri_mask_to_heatmap(heatmap, volume, epsilon=1e-8):
    """Mask the Grad-CAM heatmap with the original MRI volume and renormalise."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    volume = np.asarray(volume, dtype=np.float32)

    if heatmap.shape != volume.shape:
        raise ValueError(
            f"Heatmap shape {heatmap.shape} must match volume shape {volume.shape} for MRI masking."
        )

    mask = (volume != 0).astype(np.float32)
    masked_heatmap = heatmap * mask
    brain_max = float(masked_heatmap[mask > 0].max()) if np.any(mask > 0) else 0.0
    return masked_heatmap / (brain_max + epsilon)


def normalize_to_01(array_2d):
    """Normalise a slice to [0,1] for display"""
    array_2d = np.asarray(array_2d, dtype=np.float32)
    min_val = float(array_2d.min())
    max_val = float(array_2d.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(array_2d, dtype=np.float32)
    return (array_2d - min_val) / (max_val - min_val)


def _select_distributed_slices(scores, num_slices):
    """Select several high-attention slices according to the weight distribution, ensuring limited slice coverage."""
    if num_slices <= 0:
        return []

    sorted_indices = np.argsort(scores)[::-1]
    selected = []
    min_distance = max(1, len(scores) // max(1, num_slices * 2))

    for idx in sorted_indices:
        if all(abs(idx - sel) >= min_distance for sel in selected):
            selected.append(int(idx))
            if len(selected) >= num_slices:
                break

    if len(selected) < num_slices:
        for idx in sorted_indices:
            if int(idx) not in selected:
                selected.append(int(idx))
                if len(selected) >= num_slices:
                    break

    return sorted(selected)


def extract_best_slices_by_view(volume, heatmap, axis, num_slices=5):
    """Along the specified view, select the slices with the highest heatmap weight and return the image and heatmap slices."""
    reduce_axes = tuple(idx for idx in range(3) if idx != axis)
    slice_scores = heatmap.sum(axis=reduce_axes)
    selected_indices = _select_distributed_slices(slice_scores, num_slices)
    normalized_scores = normalize_to_01(slice_scores.astype(np.float32))

    slices = []
    for slice_idx in selected_indices:
        if axis == 0:  # Sagittal
            image_slice = volume[slice_idx, :, :]
            heatmap_slice = heatmap[slice_idx, :, :]
        elif axis == 1:  # Coronal
            image_slice = volume[:, slice_idx, :]
            heatmap_slice = heatmap[:, slice_idx, :]
        else:  # Axial
            image_slice = volume[:, :, slice_idx]
            heatmap_slice = heatmap[:, :, slice_idx]

        image_slice = np.rot90(image_slice)
        heatmap_slice = np.rot90(heatmap_slice)
        slices.append((slice_idx, normalize_to_01(image_slice), normalize_to_01(heatmap_slice)))

    return selected_indices, normalized_scores, slices


def save_gradcam_figure(volume, heatmap, sample_path, predicted_label, probabilities, output_dir, target_layer_name=None, num_slices=8):
    """Save the multi-slice overlay visualisation for the three views Axial/Coronal/Sagittal."""
    os.makedirs(output_dir, exist_ok=True)

    views = [
        ("Axial", 2),
        ("Coronal", 1),
        ("Sagittal", 0),
    ]

    cols = num_slices + 1
    fig, axes = plt.subplots(3, cols, figsize=(4 * cols, 14))

    for row, (view_name, axis) in enumerate(views):
        selected_indices, score_curve, slices = extract_best_slices_by_view(volume, heatmap, axis, num_slices=num_slices)

        for col in range(num_slices):
            ax = axes[row, col]
            slice_idx, image_slice, heatmap_slice = slices[col]
            overlay_alpha = np.clip(heatmap_slice * 0.75, 0.0, 0.7)

            ax.imshow(image_slice, cmap='gray', vmin=0.0, vmax=1.0)
            ax.imshow(heatmap_slice, cmap='plasma', vmin=0.0, vmax=1.0, alpha=overlay_alpha)
            ax.set_title(f"{view_name} Slice {slice_idx}", fontsize=9)
            ax.axis('off')

        curve_ax = axes[row, num_slices]
        curve_ax.plot(score_curve, color='#333333', linewidth=1.4, label='Score')
        curve_ax.fill_between(range(len(score_curve)), score_curve, color='#999999', alpha=0.2)
        for idx in selected_indices:
            curve_ax.axvline(idx, color='#d62728', linestyle='--', linewidth=1.0, alpha=0.8)
        curve_ax.set_title(f"{view_name} Weight Distribution", fontsize=10)
        curve_ax.set_xlabel("Slice index")
        curve_ax.set_ylabel("Normalized weight")
        curve_ax.set_xlim(0, len(score_curve) - 1)
        curve_ax.set_ylim(0, 1.0)
        curve_ax.grid(alpha=0.25)
        curve_ax.tick_params(axis='both', which='major', labelsize=8)

    prob_text = ", ".join([f"{name}={prob:.3f}" for name, prob in zip(CLASS_NAMES, probabilities)])
    title = f"3D Grad-CAM Overlay Visualization\nPredicted: {predicted_label} | {Path(sample_path).name}\n{prob_text}"
    if target_layer_name:
        title = f"{target_layer_name} Grad-CAM Comparison\n" + title

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    sample_name = Path(sample_path).name.replace('.nii.gz', '').replace('.nii', '')
    suffix = f"_{target_layer_name.replace('.', '_')}" if target_layer_name else ""
    figure_path = os.path.join(output_dir, f"gradcam_{sample_name}{suffix}.png")
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return figure_path


def predict_single_sample_with_gradcam(checkpoint_path, sample_path, output_dir=None, target_class=None, target_layer_choice=None):
    """Automatically resolve the checkpoint configuration and produce the single-sample classification prediction and 3D Grad-CAM visualisation"""
    checkpoint_obj, config, run_dir = load_checkpoint_for_inference(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint_obj, config)

    # Modify inplace activation layers only on the Grad-CAM path, to avoid affecting other functionality
    model = prepare_model_for_gradcam(model)

    input_tensor, volume = load_single_nifti_sample(sample_path, a_max=config['a_max'])

    with torch.no_grad():
        outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy()
    predicted_idx = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_idx]

    print("\n🔎 Single-sample prediction result")
    print(f"Sample path: {sample_path}")
    print(f"Predicted class: {predicted_label} (index={predicted_idx})")
    for idx, prob in enumerate(probabilities):
        print(f"  {CLASS_NAMES[idx]}: {prob:.4%}")

    if output_dir is None:
        output_dir = os.path.join(run_dir, "single_case_gradcam_comparison")

    candidate_layers = get_gradcam_candidate_layers(model)
    if target_layer_choice is None:
        preferred_order = ["attention.spatial_att", "attention", "features[-1]"]
        layer_choices = [name for name in preferred_order if name in candidate_layers]
    else:
        layer_choices = [target_layer_choice]
    results = []

    for layer_choice in layer_choices:
        try:
            gradcam = GradCAM3D(model, get_gradcam_target_layer(model, target_layer_choice=layer_choice))
        except ValueError:
            print(f"⚠️ Target layer '{layer_choice}' not found in the model; skipped.")
            continue

        input_for_cam = input_tensor.clone().detach().requires_grad_(True)
        try:
            cam_tensor, outputs, cam_class_idx = gradcam(input_for_cam, class_idx=target_class)
        finally:
            gradcam.remove_hooks()

        heatmap = cam_tensor[0, 0].detach().cpu().numpy()
        heatmap = apply_mri_mask_to_heatmap(heatmap, volume)
        layer_output_dir = os.path.join(output_dir, layer_choice.replace('.', '_'))
        figure_path = save_gradcam_figure(
            volume,
            heatmap,
            sample_path,
            predicted_label,
            probabilities,
            layer_output_dir,
            target_layer_name=layer_choice
        )

        print(f"✓ Grad-CAM target layer: {layer_choice}, class index: {cam_class_idx}")
        print(f"✓ Visualization saved to: {figure_path}")

        results.append({
            'target_layer': layer_choice,
            'class_index': int(cam_class_idx),
            'figure_path': figure_path,
        })

    return {
        'checkpoint_path': checkpoint_path,
        'sample_path': sample_path,
        'predicted_class_index': predicted_idx,
        'predicted_class_name': predicted_label,
        'probabilities': probabilities.tolist(),
        'gradcam_results': results,
        'figure_path': results[0]['figure_path'] if results else None,
        'all_figure_paths': [item['figure_path'] for item in results],
        'resolved_config': config,
    }

# ==================== Grid-search main functions ====================#

def generate_hyperparameter_combinations():
    """Generate all hyperparameter combinations"""
    combinations = []

    # Base parameter combinations
    # [Change] a_max, the data split and the epoch count have been moved out of the grid search; epochs are fixed to EPOCH_OPTIONS[0]
    base_params = itertools.product(
        MODEL_OPTIONS,
        SCHEDULER_OPTIONS,
        LOSS_OPTIONS,
        DROPOUT_OPTIONS,
        GRAD_ACCUM_OPTIONS,
        AUGMENTATION_OPTIONS,
        BATCH_SIZE_OPTIONS,
        LEARNING_RATE_OPTIONS,
    )

    for model, sched, loss, dropout, grad_acc, aug, bs, lr in base_params:
        # The attention mechanism is only effective for densenet169
        attention_options = ATTENTION_OPTIONS if model == "densenet169" else ["none"]
        for attention_type in attention_options:
            split = DATA_SPLIT
            config = {
                'model_name': model,
                'scheduler_name': sched,
                'loss_name': loss,
                'attention_type': attention_type,
                'use_amp': False,
                'dropout_rate': dropout,
                'grad_accum_steps': grad_acc,
                'augmentation_intensity': aug,
                'batch_size': bs,
                'learning_rate': lr,
                'num_epochs': EPOCH_OPTIONS[0],
                'train_ratio': split['train'],
                'val_ratio': split['val'],
                'test_ratio': split['test'],
                'a_max': DEFAULT_A_MAX
            }
            combinations.append(config)

    return combinations

def run_grid_search(max_runs=None, resume_from=0):
    """Run the grid search"""
    print("🔍 Starting Grid Search for 3D CNN Hyperparameters")
    print(f"Total possible combinations: {len(generate_hyperparameter_combinations())}")

    combinations = generate_hyperparameter_combinations()
    if max_runs:
        combinations = combinations[:max_runs]

    print(f"Running {len(combinations)} combinations (starting from {resume_from})")

    results = []
    start_time = datetime.now()

    for i, config in enumerate(combinations[resume_from:], resume_from + 1):
        try:
            result = run_single_training(config, i)
            results.append(result)

            # Save intermediate results every 10 runs
            if i % 10 == 0:
                save_grid_search_results(results, f"grid_search_results_checkpoint_{i}.csv")
                elapsed = datetime.now() - start_time
                eta = elapsed / i * (len(combinations) - i)
                print(f"\n📊 Progress: {i}/{len(combinations)} completed")
                print(f"Elapsed: {elapsed}, ETA: {eta}")

        except KeyboardInterrupt:
            print(f"\n⏹️ Grid search interrupted at run {i}")
            break
        except Exception as e:
            print(f"❌ Unexpected error in run {i}: {e}")
            results.append({
                'run_id': i,
                'config': config,
                'status': 'error',
                'error': str(e)
            })

    # Save the final results
    save_grid_search_results(results, "grid_search_final_results.csv")

    # Analyse the results
    analyze_grid_search_results(results)

    return results

def save_grid_search_results(results, filename):
    """Save the grid-search results"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, filename)

    # Flatten the configuration dictionaries
    flattened_results = []
    for result in results:
        flat_result = {
            'run_id': result.get('run_id', ''),
            'status': result.get('status', ''),
            'error': result.get('error', ''),
            'best_auc': result.get('best_auc', 0),
            'best_f1': result.get('best_f1', 0),
            'best_acc': result.get('best_acc', 0),
            'final_auc': result.get('final_auc', 0),
            'final_f1': result.get('final_f1', 0),
            'final_acc': result.get('final_acc', 0),
            'total_params': result.get('total_params', 0),
            'run_dir': result.get('run_dir', '')
        }

        # Add the configuration parameters
        config = result.get('config', {})
        for key, value in config.items():
            flat_result[f'config_{key}'] = value

        flattened_results.append(flat_result)

    df = pd.DataFrame(flattened_results)
    df.to_csv(output_path, index=False)
    print(f"✓ Results saved to {output_path}")

def analyze_grid_search_results(results):
    """Analyse the grid-search results"""
    print("\n" + "="*80)
    print("GRID SEARCH ANALYSIS")
    print("="*80)

    # Filter the successful results
    successful_results = [r for r in results if r.get('status') == 'completed']

    if not successful_results:
        print("No successful runs to analyze")
        return

    # Best results
    best_auc = max(successful_results, key=lambda x: x['best_auc'])
    best_f1 = max(successful_results, key=lambda x: x['best_f1'])
    best_acc = max(successful_results, key=lambda x: x['best_acc'])

    print(f"🏆 Best AUC: {best_auc['best_auc']:.4f} (Run {best_auc['run_id']})")
    print(f"🏆 Best F1: {best_f1['best_f1']:.4f} (Run {best_f1['run_id']})")
    print(f"🏆 Best Accuracy: {best_acc['best_acc']:.4f} (Run {best_acc['run_id']})")

    # Parameter-importance analysis (simplified)
    print("\n📈 Parameter Importance Analysis:")

    # Analysis by model
    model_performance = {}
    for result in successful_results:
        model = result['config']['model_name']
        if model not in model_performance:
            model_performance[model] = []
        model_performance[model].append(result['best_auc'])

    print("\nModel Performance (AUC):")
    for model, scores in model_performance.items():
        avg_auc = np.mean(scores)
        std_auc = np.std(scores)
        print(f"  {model}: {avg_auc:.4f} ± {std_auc:.4f} (n={len(scores)})")

    # Save the detailed analysis
    analysis = {
        'best_auc_run': best_auc['run_id'],
        'best_auc_score': best_auc['best_auc'],
        'best_auc_config': best_auc['config'],
        'best_f1_run': best_f1['run_id'],
        'best_f1_score': best_f1['best_f1'],
        'best_f1_config': best_f1['config'],
        'best_acc_run': best_acc['run_id'],
        'best_acc_score': best_acc['best_acc'],
        'best_acc_config': best_acc['config'],
        'total_runs': len(results),
        'successful_runs': len(successful_results),
        'model_performance': model_performance
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    analysis_df = pd.DataFrame([analysis])
    analysis_df.to_csv(os.path.join(RESULTS_DIR, "grid_search_analysis.csv"), index=False)
    print(f"✓ Analysis saved to {os.path.join(RESULTS_DIR, 'grid_search_analysis.csv')}")


def load_all_successful_runs(global_csv_path=GLOBAL_TRAINING_RESULTS_CSV):
    if not os.path.exists(global_csv_path):
        return []
    df = pd.read_csv(global_csv_path)
    df = df[df['status'] == 'completed']
    return df.to_dict(orient='records')


def evaluate_checkpoint_on_test_set(checkpoint_path, test_files):
    checkpoint_obj, config, run_dir = load_checkpoint_for_inference(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint_obj, config)
    model.eval()

    data_list = []
    for item in test_files:
        if os.path.exists(item['file_path']):
            data_list.append({'image': item['file_path'], 'label': int(item['label'])})
        else:
            print(f"⚠️ Test file missing: {item['file_path']}")

    if not data_list:
        raise ValueError("No valid test samples found for evaluation.")

    _, test_transforms = get_data_transforms(0, a_max=config.get('a_max', DEFAULT_A_MAX))
    dataset = Dataset(data=data_list, transform=test_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test Evaluation", leave=False):
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_proba.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    metrics = {
        'accuracy': np.mean(y_true == y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    try:
        if y_proba.shape[1] == 2:
            metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
    except Exception:
        metrics['auc'] = 0.5

    metrics['num_samples'] = len(y_true)
    metrics['checkpoint_path'] = checkpoint_path
    metrics['config'] = config
    return metrics


def run_test_mode():
    os.makedirs(TEST_EVALUATION_OUTPUT_DIR, exist_ok=True)
    runs = load_all_successful_runs()
    if not runs:
        raise ValueError("No completed runs found in global training results.")

    fixed_split = load_fixed_data_split()
    if not fixed_split or 'test_split' not in fixed_split:
        raise ValueError("fixed_data_split.json is missing or invalid. Please run training once to generate it.")

    test_files = fixed_split['test_split']
    evaluations = []

    for run in runs:
        run_dir = run['run_dir']
        if not os.path.isdir(run_dir):
            print(f"⚠️ Skipping missing run directory: {run_dir}")
            continue

        config = resolve_run_config(run_dir)
        if not config:
            print(f"⚠️ Could not resolve config for run: {run_dir}")
            continue

        best_checkpoint = find_checkpoint_by_tag(run_dir, 'best_auc')
        if not best_checkpoint:
            print(f"⚠️ No best_auc checkpoint found for run: {run_dir}; skipping for test evaluation")
            continue

        try:
            metrics = evaluate_checkpoint_on_test_set(best_checkpoint, test_files)
            metrics['run_dir'] = run_dir
            metrics['run_id'] = run.get('run_id')
            metrics['best_checkpoint'] = best_checkpoint
            metrics['source'] = run.get('source_file', '')
            evaluations.append(metrics)
            print(f"✓ Evaluated run {run.get('run_id')} | AUC={metrics['auc']:.4f} | F1={metrics['f1_macro']:.4f} | Acc={metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"❌ Evaluation failed for run {run.get('run_id')}: {e}")

    if not evaluations:
        raise ValueError("No runs were successfully evaluated on the test set.")

    eval_df = pd.DataFrame(evaluations)
    eval_df = eval_df.sort_values(by='auc', ascending=False).reset_index(drop=True)
    eval_df.to_csv(os.path.join(TEST_EVALUATION_OUTPUT_DIR, 'test_evaluation_summary.csv'), index=False)
    eval_df.to_csv(os.path.join(TEST_EVALUATION_OUTPUT_DIR, 'test_evaluation_details.csv'), index=False)
    print(f"✓ Test evaluation summary saved to {TEST_EVALUATION_OUTPUT_DIR}")
    return eval_df


def run_choose_mode(top_k=None, score_threshold=0.01):
    """Select the best model: composite scoring and robustness rules based on all_training_results.csv."""
    # Read the global training results and keep only the completed records
    if not os.path.exists(GLOBAL_TRAINING_RESULTS_CSV):
        raise ValueError(f"Global training results file not found: {GLOBAL_TRAINING_RESULTS_CSV}")

    df = pd.read_csv(GLOBAL_TRAINING_RESULTS_CSV)
    df = df[df['status'] == 'completed'].copy()
    if df.empty:
        raise ValueError("No completed runs found in global training results.")

    # Make sure the directory structure exists so the analysis results can be saved later
    os.makedirs(MODEL_SELECTION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Parse the config field and fill in the key metrics
    df['config'] = df['config'].apply(parse_config_from_string)
    df['best_auc'] = pd.to_numeric(df.get('best_auc', 0), errors='coerce').fillna(0.0)
    df['best_f1'] = pd.to_numeric(df.get('best_f1', 0), errors='coerce').fillna(0.0)

    candidates = []
    for _, row in df.iterrows():
        run_dir = row.get('run_dir', '')
        if not isinstance(run_dir, str) or not os.path.isdir(run_dir):
            print(f"⚠️ Run directory missing or invalid, skipping: {run_dir}")
            continue

        epoch_csv_path = os.path.join(run_dir, 'epoch_metrics.csv')
        min_val_loss = float('inf')
        total_epochs = 0
        if os.path.exists(epoch_csv_path):
            try:
                epoch_df = pd.read_csv(epoch_csv_path)
                if 'val_loss' in epoch_df.columns:
                    min_val_loss = float(pd.to_numeric(epoch_df['val_loss'], errors='coerce').min(skipna=True))
                total_epochs = len(epoch_df)
            except Exception as e:
                print(f"⚠️ Failed to parse epoch_metrics for {run_dir}: {e}")
        else:
            print(f"⚠️ Missing epoch_metrics.csv for run {run_dir}; using fallback values.")

        total_score = 0.6 * float(row['best_auc']) + 0.4 * float(row['best_f1'])
        candidates.append({
            'run_id': row.get('run_id', ''),
            'run_dir': run_dir,
            'best_auc': float(row['best_auc']),
            'best_f1': float(row['best_f1']),
            'best_acc': float(row.get('best_acc', 0)),
            'total_score': total_score,
            'min_val_loss': min_val_loss,
            'total_epochs': int(total_epochs),
            'config': row['config'],
        })

    if not candidates:
        raise ValueError("No valid completed runs available for model selection.")

    candidates_df = pd.DataFrame(candidates)
    candidates_df = candidates_df.sort_values(by='total_score', ascending=False).reset_index(drop=True)
    candidates_df['rank'] = candidates_df.index + 1

    # If top_k is specified, the full candidate ranking is still saved to the output, but only the top top_k are printed
    if top_k is not None:
        top_display = min(top_k, len(candidates_df))
    else:
        top_display = len(candidates_df)

    # Select Top1 by total score and run a robustness comparison within the band
    top_score = float(candidates_df.loc[0, 'total_score'])
    close_candidates = candidates_df[candidates_df['total_score'] >= top_score - score_threshold].copy()

    if len(close_candidates) > 1:
        # First compare the minimum validation loss
        close_candidates['min_val_loss'] = pd.to_numeric(close_candidates['min_val_loss'], errors='coerce').fillna(float('inf'))
        min_loss = close_candidates['min_val_loss'].min()
        loss_candidates = close_candidates[close_candidates['min_val_loss'] <= min_loss + 1e-8].copy()
        if len(loss_candidates) > 1:
            # Then compare the number of training epochs, preferring the model that converged faster
            loss_candidates['total_epochs'] = pd.to_numeric(loss_candidates['total_epochs'], errors='coerce').fillna(float('inf'))
            best_row = loss_candidates.loc[loss_candidates['total_epochs'].idxmin()]
        else:
            best_row = loss_candidates.iloc[0]
    else:
        best_row = close_candidates.iloc[0]

    # Save the analysis results for later review
    candidates_df['config'] = candidates_df['config'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    candidates_df.to_csv(os.path.join(MODEL_SELECTION_OUTPUT_DIR, 'model_selection_analysis.csv'), index=False)

    best_checkpoint = find_checkpoint_by_tag(best_row['run_dir'], 'best_auc')
    if not best_checkpoint:
        raise ValueError(f"Selected run {best_row['run_dir']} does not contain a best_auc checkpoint.")

    best_output = {
        'selected_run_id': str(best_row['run_id']),
        'selected_run_dir': str(best_row['run_dir']),
        'selected_checkpoint': str(best_checkpoint),
        'best_auc': float(best_row['best_auc']),
        'best_f1': float(best_row['best_f1']),
        'best_acc': float(best_row['best_acc']),
        'total_score': float(best_row['total_score']),
        'min_val_loss': float(best_row['min_val_loss']),
        'total_epochs': int(best_row['total_epochs']),
        'config': best_row['config'],
    }

    with open(os.path.join(MODEL_SELECTION_OUTPUT_DIR, 'best_model_selected.txt'), 'w', encoding='utf-8') as f:
        json.dump(best_output, f, indent=2, ensure_ascii=False)

    print(f"✓ Best model selected: run_id={best_output['selected_run_id']}, run_dir={best_output['selected_run_dir']}")
    print(f"  best_auc={best_output['best_auc']:.4f}, best_f1={best_output['best_f1']:.4f}, total_score={best_output['total_score']:.4f}")
    print(f"  min_val_loss={best_output['min_val_loss']:.4f}, total_epochs={best_output['total_epochs']}")
    print(f"  Selection results saved to {os.path.join(MODEL_SELECTION_OUTPUT_DIR, 'model_selection_analysis.csv')} and best_model_selected.txt")

    return best_output


def select_group_samples(test_files, label_int, n_samples=5):
    subset = [item for item in test_files if int(item['label']) == label_int]
    if not subset:
        return []
    if len(subset) <= n_samples:
        return subset
    random.seed(GLOBAL_SEED)
    return random.sample(subset, n_samples)


def build_group_heatmap(heatmaps):
    if not heatmaps:
        raise ValueError("Heatmap list is empty.")
    stacked = np.stack(heatmaps, axis=0)
    mean_map = np.mean(stacked, axis=0)
    cv_map = np.std(stacked, axis=0) / (np.mean(stacked, axis=0) + 1e-8)
    return mean_map, cv_map


def generate_group_comparison_visualizations(model, config, selected_samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Modify inplace activation layers only on the Grad-CAM path, to avoid affecting training/test
    model = prepare_model_for_gradcam(model)
    model.eval()

    # Collect the mean heatmap and the original volume for each group
    all_group_heatmaps = {}
    all_group_volumes = {}
    all_group_freq_maps = {}
    for label_int, samples in selected_samples.items():
        heatmaps = []
        volumes = []
        binary_masks = []
        for sample in samples:
            sample_path = sample['file_path']
            input_tensor, volume = load_single_nifti_sample(sample_path, a_max=config.get('a_max', DEFAULT_A_MAX))
            with torch.no_grad():
                outputs = model(input_tensor)
            predicted_idx = int(torch.argmax(torch.softmax(outputs, dim=1), dim=1).item())

            layer = get_gradcam_target_layer(model, target_layer_choice='features[-1]')
            gradcam = GradCAM3D(model, layer)
            input_for_cam = input_tensor.clone().detach().requires_grad_(True)
            try:
                cam_tensor, _, _ = gradcam(input_for_cam, class_idx=predicted_idx)
            finally:
                gradcam.remove_hooks()

            heatmap = cam_tensor[0, 0].detach().cpu().numpy()
            heatmap = apply_mri_mask_to_heatmap(heatmap, volume)
            heatmaps.append(heatmap)
            volumes.append(volume)

            mu = np.mean(heatmap)
            sigma = np.std(heatmap)
            threshold = mu + 2.0 * sigma
            mask = (heatmap > threshold).astype(np.float32)
            binary_masks.append(mask)

        mean_map, _ = build_group_heatmap(heatmaps)
        all_group_heatmaps[label_int] = mean_map
        all_group_volumes[label_int] = volumes

        if binary_masks:
            consensus_map = np.mean(np.stack(binary_masks, axis=0), axis=0)
        else:
            consensus_map = np.zeros_like(mean_map, dtype=np.float32)
        all_group_freq_maps[label_int] = consensus_map

    display_group_names = {0: 'CN', 1: 'MCI', 2: 'AD'}
    group_colors = {0: '#0072B2', 1: '#D55E00', 2: '#CC0000'}
    axes = ['Sagittal', 'Coronal', 'Axial']
    axis_idx = [0, 1, 2]

    num_slices_per_axis = int(config.get('group_vis_slices_per_axis', 8))
    num_slices_per_axis = max(8, num_slices_per_axis)

    rows = len(display_group_names) * len(axes)
    cols = num_slices_per_axis + 1
    width_ratios = [1.0] * num_slices_per_axis + [0.8]

    def _render_figure(theme='dark'):
        if theme == 'dark':
            bg_color = '#F0F0F0'
            heatmap_cmap = 'plasma'
            title_color = 'black'
            axis_label_color = 'black'
            spine_color = '#333333'
        else:
            bg_color = 'white'
            heatmap_cmap = 'inferno'
            title_color = 'black'
            axis_label_color = 'black'
            spine_color = '#333333'

        fig = plt.figure(figsize=(7.0, 9.0))
        fig.patch.set_facecolor(bg_color)
        gs = fig.add_gridspec(rows, cols, width_ratios=width_ratios, hspace=0.26, wspace=0.04)

        fig.suptitle('Group-level Saliency Map Aggregation', fontsize=10, fontweight='bold', fontfamily='Arial', color=title_color, y=0.995)
        fig.text(0.98, 0.99, 'a', fontsize=10, fontweight='bold', fontfamily='Arial', color=title_color, ha='right', va='top')
        # fig.text(0.94, 0.945, 'Weight', fontsize=8, fontweight='bold', fontfamily='Arial', color=title_color, ha='center', va='bottom')

        row_axes = {}
        for row_idx, (label_int, axis) in enumerate([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]):
            consensus_map = all_group_freq_maps[label_int]
            sample_vol = all_group_volumes[label_int][0]
            weight_vector = np.mean(consensus_map, axis=tuple([d for d in range(3) if d != axis]))
            best_slices = _select_distributed_slices(weight_vector, num_slices_per_axis)

            row_label = f"{display_group_names[label_int]} - {axes[axis]}\n(n={len(selected_samples[label_int])})"
            row_color = group_colors[label_int]

            for s_i, slice_idx in enumerate(best_slices):
                if axis == 0:
                    img = np.rot90(sample_vol[slice_idx, :, :])
                    heat = np.rot90(consensus_map[slice_idx, :, :])
                    position_label = f'x={slice_idx}'
                elif axis == 1:
                    img = np.rot90(sample_vol[:, slice_idx, :])
                    heat = np.rot90(consensus_map[:, slice_idx, :])
                    position_label = f'y={slice_idx}'
                else:
                    img = np.rot90(sample_vol[:, :, slice_idx])
                    heat = np.rot90(consensus_map[:, :, slice_idx])
                    position_label = f'z={slice_idx}'

                ax = fig.add_subplot(gs[row_idx, s_i])
                ax.set_facecolor(bg_color)
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.imshow(heat, cmap=heatmap_cmap, alpha=0.55)
                ax.axis('off')

                if s_i == 0:
                    ax.text(-0.12, 0.5, row_label, transform=ax.transAxes, va='center', ha='right', fontsize=8, fontweight='bold', fontfamily='Arial', color=row_color)

                ax.text(0.02, 0.96, position_label, transform=ax.transAxes, fontsize=6, color='white', va='top', ha='left', path_effects=[patheffects.withStroke(linewidth=1.0, foreground='black')])

                if s_i == 0:
                    row_axes[row_idx] = ax

            axc = fig.add_subplot(gs[row_idx, cols - 1])
            x_vals = np.arange(len(weight_vector))
            axc.plot(x_vals, weight_vector, color='#333333', linewidth=1.0)
            axc.fill_between(x_vals, weight_vector, color='#999999', alpha=0.15)
            for s_idx in best_slices:
                axc.axvline(s_idx, color='#999999', linestyle='--', linewidth=0.8, alpha=0.7)
            axc.set_xlim(0, len(weight_vector) - 1)
            axc.set_ylim(0, 1)
            axc.set_xticks([0, len(weight_vector) - 1])
            axc.set_yticks([0, 1])
            axc.tick_params(axis='both', labelsize=6, colors=axis_label_color)
            axc.set_xlabel('Slice', fontsize=7, fontfamily='Arial', labelpad=2, color=axis_label_color)
            # axc.set_ylabel('Weight', fontsize=7, fontfamily='Arial', labelpad=2, color=axis_label_color)
            axc.spines['top'].set_visible(False)
            axc.spines['right'].set_visible(False)
            axc.spines['bottom'].set_color(spine_color)
            axc.spines['left'].set_color(spine_color)
            axc.yaxis.grid(True, linestyle='--', linewidth=0.3, color='#E0E0E0', alpha=0.7)
            axc.xaxis.grid(False)
            axc.xaxis.set_tick_params(pad=2)
            axc.text(0.98, 0.85, 'Selected\nSlice', transform=axc.transAxes, fontsize=7, fontfamily='Arial', ha='right', va='top', color=axis_label_color, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
            # Manual adjustment of the position of the "Weight" label so that it sits closer to the curve
            if row_idx == 0:
                axc_pos = axc.get_position()
                fig.text(
                    axc_pos.x0 + axc_pos.width / 2,
                    axc_pos.y1 + 0.02,
                    'Weight',
                    fontsize=8,
                    fontweight='bold',
                    fontfamily='Arial',
                    color=axis_label_color,
                    ha='center',
                    va='bottom'
            )
        for boundary_row in [3, 6]:
            top_axis = row_axes[boundary_row - 1]
            bottom_axis = row_axes[boundary_row]
            y_top = top_axis.get_position().y0
            y_bottom = bottom_axis.get_position().y1
            y_mid = (y_top + y_bottom) / 2.0
            fig.add_artist(plt.Line2D([0.03, 0.96], [y_mid, y_mid], color='#CCCCCC', linewidth=0.5, linestyle='-'))

        cax = fig.add_axes([0.96, 0.12, 0.015, 0.76])
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=heatmap_cmap), cax=cax)
        cbar.set_label('Consensus Probability (0-1)', fontsize=8, fontfamily='Arial')
        cbar.set_ticks([0.0, 1.0])
        cbar.ax.tick_params(labelsize=7)

        caption_prefix = 'Fig. X |'
        caption_body = (
            ' Saliency frequency maps showing the proportion of samples (mean + 2σ threshold) '
            'where each voxel was identified as significant. Rows represent anatomical views; columns '
            'represent representative slices selected by frequency weight distribution. Heatmaps were '
            'generated using Grad-CAM on the deepest feature layer (features[-1]) and aggregated '
            'across n=10 samples per group using a dynamic threshold (mean + 2σ).'
        )
        caption_text = textwrap.fill(caption_prefix + caption_body, width=120)

        fig.text(0.02, 0.08,
                 'Coordinates are in MNI152 standard space.',
                 fontsize=7, fontfamily='Arial', color=title_color, ha='left', va='bottom')
        fig.text(0.03, 0.01,
                 caption_text,
                 fontsize=9, fontfamily='Arial', color='#333333', ha='left', va='bottom', linespacing=1.3)

        return fig

    dark_fig = _render_figure(theme='dark')
    consensus_tiff_path = os.path.join(output_dir, 'group_saliency_consensus_aggregation.tiff')
    consensus_png_path = os.path.join(output_dir, 'group_saliency_consensus_aggregation.png')
    dark_fig.savefig(consensus_tiff_path, dpi=600, format='tiff', bbox_inches='tight')
    dark_fig.savefig(consensus_png_path, dpi=600, bbox_inches='tight')
    plt.close(dark_fig)

    white_fig = _render_figure(theme='white')
    white_tiff_path = os.path.join(output_dir, 'group_saliency_consensus_aggregation_white.tiff')
    white_png_path = os.path.join(output_dir, 'group_saliency_consensus_aggregation_white.png')
    white_fig.savefig(white_tiff_path, dpi=600, format='tiff', bbox_inches='tight')
    white_fig.savefig(white_png_path, dpi=600, bbox_inches='tight')
    plt.close(white_fig)

    return {
        'group_consensus_figure_tiff': consensus_tiff_path,
        'group_consensus_figure_png': consensus_png_path,
        'group_consensus_figure_white_tiff': white_tiff_path,
        'group_consensus_figure_white_png': white_png_path,
    }


def run_group_comparison_mode(model_checkpoint, n_samples_per_group=5):
    fixed_split = load_fixed_data_split()
    if not fixed_split or 'test_split' not in fixed_split:
        raise ValueError("fixed_data_split.json is missing or invalid. Please run training once to generate it.")

    test_files = fixed_split['test_split']
    groups = {
        0: select_group_samples(test_files, 0, n_samples_per_group),
        1: select_group_samples(test_files, 1, n_samples_per_group),
        2: select_group_samples(test_files, 2, n_samples_per_group)
    }

    for label_int, samples in groups.items():
        if not samples:
            raise ValueError(f"Not enough samples in group {label_int} for group comparison.")

    if not model_checkpoint:
        raise ValueError("Please specify a model_checkpoint path for group_comparison mode.")

    model, config, run_dir = safe_build_model_from_checkpoint(model_checkpoint)
    output_dir = os.path.join(GROUP_COMPARISON_OUTPUT_DIR, os.path.basename(Path(model_checkpoint).parent))
    os.makedirs(output_dir, exist_ok=True)

    results = generate_group_comparison_visualizations(model, config, groups, output_dir)
    print(f"✓ Group comparison figures saved to {output_dir}")
    return results

# ==================== Bayesian optimisation functions ====================#

def objective(trial):
    """Optuna objective function"""
    # Define the search space
    model_name = trial.suggest_categorical('model_name', MODEL_OPTIONS)
    scheduler_name = trial.suggest_categorical('scheduler_name', SCHEDULER_OPTIONS)
    loss_name = trial.suggest_categorical('loss_name', LOSS_OPTIONS)
    dropout_rate = trial.suggest_categorical('dropout_rate', DROPOUT_OPTIONS)
    grad_accum_steps = trial.suggest_categorical('grad_accum_steps', GRAD_ACCUM_OPTIONS)
    augmentation_intensity = trial.suggest_categorical('augmentation_intensity', AUGMENTATION_OPTIONS)
    batch_size = trial.suggest_categorical('batch_size', BATCH_SIZE_OPTIONS)
    learning_rate = trial.suggest_categorical('learning_rate', LEARNING_RATE_OPTIONS)
    # epochs are fixed as a hyperparameter and do not take part in the search
    num_epochs = EPOCH_OPTIONS[0]
    # a_max is fixed to DEFAULT_A_MAX in the current configuration and has been removed from the search space
    a_max = DEFAULT_A_MAX

    # The attention mechanism is only effective for densenet169
    attention_type = trial.suggest_categorical('attention_type', ATTENTION_OPTIONS) if model_name == "densenet169" else "none"

    # Dataset split - hard-coded to avoid Optuna warnings and database storage issues
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    config = {
        'model_name': model_name,
        'scheduler_name': scheduler_name,
        'loss_name': loss_name,
        'attention_type': attention_type,
        'use_amp': False,
        'dropout_rate': dropout_rate,
        'grad_accum_steps': grad_accum_steps,
        'augmentation_intensity': augmentation_intensity,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'a_max': a_max  # NEW
    }

    # Run the training
    run_id = trial.number + 1  # Optuna trial numbering starts from 0
    result = run_single_training(config, run_id)

    # Return the metric to optimise (ACC; Optuna maximises by default)
    if result.get('status') == 'completed':
        return result['best_auc']
    else:
        # Return the default value on failure
        return 0.0

def run_bayesian_optimization(n_trials=50, timeout=None, study_name="3d_cnn_hyperopt", resume=True):
    """Run Bayesian optimisation"""
    print("🔬 Starting Bayesian Optimization for 3D CNN Hyperparameters")
    print(f"Number of trials: {n_trials}")
    print(f"Study name: {study_name}, Resume: {resume}")

    # Create or load the Optuna study, using SQLite storage
    storage_url = f"sqlite:///{study_name}.db"
    
    sampler = optuna.samplers.TPESampler(multivariate=True)

    if resume:
        # Try to load the existing study
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            completed_trials = len([t for t in study.trials if str(t.state).endswith('COMPLETE')])
            print(f"✓ Resumed existing study with {completed_trials} completed trials")
        except:
            # If it does not exist, create a new study
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=storage_url,
                sampler=sampler,
            )
            print("✓ Created new study")
    else:
        # Force creation of a new study (overwrites the old one)
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=storage_url,
            sampler=sampler,
        )
        print("✓ Created new study (overwrite existing)")

    # Work out how many more trials still need to run
    completed_trials = len([t for t in study.trials if str(t.state).endswith('COMPLETE')])
    remaining_trials = max(0, n_trials - completed_trials)
    
    if remaining_trials > 0:
        print(f"Running {remaining_trials} additional trials...")
        # Run the optimisation
        study.optimize(objective, n_trials=remaining_trials, timeout=timeout)
    else:
        print("All requested trials already completed!")

    # Output the best result
    print("\n" + "="*80)
    print("BAYESIAN OPTIMIZATION RESULTS")
    print("="*80)

    best_trial = study.best_trial
    print(f"🏆 Best AUC: {best_trial.value:.4f}")
    print(f"🏆 Best Trial: {best_trial.number}")
    print(f"🏆 Best Params: {best_trial.params}")
    print(f"🏆 Total trials completed: {len(study.trials)}")

    # Debug: print the state of all trials
    print("\nDEBUG: All trials status:")
    for i, trial in enumerate(study.trials):
        print(f"  Trial {i}: state={trial.state}, value={trial.value}")

    # Save the results of all trials
    try:
        results = []
        for trial in study.trials:
            result = {
                'trial_number': trial.number,
                'auc': trial.value if trial.value is not None else 0.0,
                'params': trial.params,
                'state': trial.state.name if hasattr(trial.state, 'name') else str(trial.state)
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        df.to_csv("bayesian_optimization_results.csv", index=False)
        print("✓ All trial results saved to bayesian_optimization_results.csv")
    except Exception as e:
        print(f"Warning: Failed to save results: {e}")

    # Visualisation (if plotly is installed)
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html("optimization_history.html")
        print("✓ Optimization history plot saved to optimization_history.html")

        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html("param_importances.html")
        print("✓ Parameter importances plot saved to param_importances.html")
    except Exception as e:
        print(f"Visualization failed: {e}")

    return study
# ==================== Run grid search or Bayesian optimisation ====================#

# ... (the original code remains unchanged down to the end of the file) ...

def run_batch_inference(checkpoint_path, dataset_list_path, output_dir, excel_sheet_name='Sheet1'):
    """
    Run batch inference and evaluation on a new dataset list.
    :param checkpoint_path: path to the trained model checkpoint
    :param dataset_list_path: path to the new dataset list (Excel/CSV); must contain the 'file_path' and 'label' columns
    :param output_dir: root directory for saving the results
    :param excel_sheet_name: Excel sheet name
    """
    print(f"\n🚀 Starting Batch Inference on new dataset: {dataset_list_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the checkpoint and resolve the configuration
    checkpoint_obj, config, run_dir = load_checkpoint_for_inference(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint_obj, config)
    model.eval()
    
    # 2. Load the new dataset list
    # Excel or CSV is supported
    if dataset_list_path.lower().endswith('.xlsx') or dataset_list_path.lower().endswith('.xls'):
        df = pd.read_excel(dataset_list_path, sheet_name=excel_sheet_name)
    else:
        df = pd.read_csv(dataset_list_path)
    
    # Check the required columns
    if 'file_path' not in df.columns:
        raise ValueError("Dataset list must contain a 'file_path' column")
    
    # If a 'label' column is present, run supervised evaluation; otherwise only predict
    has_labels = 'label' in df.columns
    if has_labels:
        df['label_int'] = df['label'].map(LABEL_MAP_FLOAT_TO_INT)
        all_labels = df['label_int'].tolist()
    else:
        print("⚠️ Label column not found. Running in prediction-only mode.")
    
    # 3. Data preprocessing (using the test/validation transforms)
    # Note: data augmentation must not be used at test time; always use augmentation_intensity=0
    _, test_transforms = get_data_transforms(0, a_max=config['a_max'])
    
    # 4. Build the Dataset (MONAI-compatible)
    # Filter to the files that exist and preserve the index correspondence
    data_list = []
    valid_file_paths = [] # used to record results later, solving the metadata-loss problem
    
    for idx, row in df.iterrows():
        img_path = row['file_path']
        if os.path.exists(img_path):
            sample_dict = {"image": img_path}
            if has_labels:
                sample_dict["label"] = int(row['label_int'])
            data_list.append(sample_dict)
            valid_file_paths.append(img_path) # keep the path order consistent with the DataLoader output order
        else:
            print(f" File not found: {img_path}, skipping...")
    
    print(f"✅ Found {len(data_list)} valid samples out of {len(df)} listed.")
    dataset = Dataset(data=data_list, transform=test_transforms)
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    # 5. Inference loop
    y_true = []
    y_pred = []
    y_proba = []
    # We no longer read paths from the batch; we use the valid_file_paths list directly
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batch Inference"):
            inputs = batch["image"].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Collect the results
            y_pred.extend(preds.cpu().numpy())
            y_proba.extend(probs.cpu().numpy())
            
            # If ground-truth labels are available
            if has_labels:
                labels = batch["label"].cpu().numpy()
                y_true.extend(labels)
    
    # 6. Generate the evaluation report and detailed results
    # At this point the lengths and orders of y_pred, y_proba and valid_file_paths correspond one-to-one
    prediction_details = []
    
    for i in range(len(valid_file_paths)):
        detail = {
            "file_path": valid_file_paths[i],
        }
        if has_labels:
            # Note: here we must ensure the length of y_true matches valid_file_paths
            # If any samples were skipped during data loading, the logic above already keeps them in sync
            detail["true_label"] = y_true[i] if i < len(y_true) else "N/A"
            # Get the class name
            true_label_key = str(y_true[i]) if i < len(y_true) else None
            # Simple handling here: if the mapping table is {0.0:0, 0.5:1, 1.0:2}, we look up in reverse or use the index directly
            # In practice CLASS_NAMES is ["Class_0", "Class_1", "Class_2"]
            # If your labels are 0, 1, 2
            if i < len(y_true) and int(y_true[i]) < len(CLASS_NAMES):
                detail["true_label_name"] = CLASS_NAMES[int(y_true[i])]
            else:
                detail["true_label_name"] = "Unknown"
                
        pred_label = y_pred[i]
        detail["predicted_label"] = pred_label
        detail["predicted_label_name"] = CLASS_NAMES[pred_label] if pred_label < len(CLASS_NAMES) else "Unknown"
        
        # Add the per-class probabilities
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            if i < len(y_proba) and cls_idx < len(y_proba[i]):
                detail[f"prob_{cls_name}"] = y_proba[i][cls_idx]
            else:
                detail[f"prob_{cls_name}"] = 0.0
                
        prediction_details.append(detail)

    # 7. Generate the evaluation report (if labels are available)
    report = {}
    if has_labels and len(y_true) > 0 and len(y_pred) > 0:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_proba = np.array(y_proba)
        
        # Compute the metrics
        report['accuracy'] = np.mean(y_true == y_pred)
        try:
            # Handle binary or multi-class classification
            if len(y_proba[0]) == 2:
                report['auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                report['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except Exception as e:
            print(f"Warning: AUC calculation failed: {e}")
            report['auc'] = 0.5
        report['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        report['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title('Confusion Matrix - New Dataset')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = os.path.join(output_dir, 'confusion_matrix_new_dataset.png')
        plt.savefig(cm_path)
        plt.close()
        
        print(f"\n📋 Evaluation Report on New Dataset:")
        print(f"   Accuracy: {report['accuracy']:.4f}")
        print(f"   AUC: {report['auc']:.4f}")
        print(f"   F1 Macro: {report['f1_macro']:.4f}")
    
    # 8. Save the results
    # Save the detailed predictions
    details_df = pd.DataFrame(prediction_details)
    details_df.to_csv(os.path.join(output_dir, 'detailed_predictions.csv'), index=False)
    
    # Save the summary report
    if has_labels and report:
        report_df = pd.DataFrame([report])
        report_df.to_csv(os.path.join(output_dir, 'summary_report.csv'), index=False)
        print(f"✅ Batch inference completed. Summary report saved.")
    else:
        print(f"✅ Batch prediction completed. Results saved.")
        
    print(f"✅ Detailed predictions saved to: {os.path.join(output_dir, 'detailed_predictions.csv')}")
    return report if has_labels else None


if __name__ == "__main__":
    try:
        # Run modes:
        # - "train": keeps the original grid-search / Bayesian-optimisation training workflow
        # - "inference": reads the specified checkpoint, predicts a single NIfTI sample and generates a 3D Grad-CAM
        # - "batch_inference": batch inference and evaluation on a new dataset list
        # - "test": unified evaluation of all successfully completed runs on the fixed test set
        # - "choose": intelligently selects the best model from past runs based on validation AUC stability
        # - "group_comparison": Grad-CAM group comparison across the three class groups of the fixed test set
        RUN_MODE = "batch_inference"  # "train", "inference", "batch_inference", "test", "choose", "group_comparison"

        # ==================== Inference-mode configuration ====================#
        CHECKPOINT_PATH = r"<AUTHOR_DATA_ROOT>\MONAILabel\cnn_Grid_3.7\Grid_SEARCH_128_densenet169_Attn_axial_dropout_0.0_aug_1_bs_5_lr_1e-04_amax_1100_sched_reduce_on_plateau_loss_ce\checkpoints\checkpoint_epoch_044_best_auc.pth"
        SAMPLE_NIFTI_PATH = r"<AUTHOR_DATA_ROOT>/datasets/miriad\miriad_classified_processed_SyN_fast_ws\1\miriad_188_AD_M_01_MR_1_ws.nii.gz"
        INFERENCE_OUTPUT_DIR = r"<AUTHOR_DATA_ROOT>/MONAILabel/AD_VS/results3.7_128AUC/visualisation_output/1_miriad"
        TARGET_CLASS = None  # None = run Grad-CAM on the predicted class; 0/1/2 may also be specified manually

        # ==================== Training-mode configuration ====================# 
        SEARCH_METHOD = "grid"  # "grid" or "bayesian"
        MAX_RUNS = 0  # maximum number of grid-search runs
        BAYESIAN_TRIALS = 250  # number of trials for Bayesian optimisation
        RESUME_FROM = 0  # run index to resume from (grid search only)
        RESUME_BAYESIAN = True  # set to False to restart the optimisation after changing network parameters
        STUDY_NAME = "bayesian_v3.7"  # Bayesian-optimisation study name

        print("🚀 Starting 3D CNN workflow")
        print(f"Run mode: {RUN_MODE}")

        if RUN_MODE == "train":
            print(f"Search method: {SEARCH_METHOD}")
            if SEARCH_METHOD == "grid":
                print(f"Max runs: {MAX_RUNS if MAX_RUNS else 'All'}")
                print(f"Resume from: {RESUME_FROM}")
                results = run_grid_search(max_runs=MAX_RUNS, resume_from=RESUME_FROM)
            elif SEARCH_METHOD == "bayesian":
                print(f"Number of trials: {BAYESIAN_TRIALS}")
                print(f"Resume: {RESUME_BAYESIAN}")
                print(f"Study name: {STUDY_NAME}")
                study = run_bayesian_optimization(
                    n_trials=BAYESIAN_TRIALS,
                    study_name=STUDY_NAME,
                    resume=RESUME_BAYESIAN
                )
            else:
                raise ValueError("Invalid SEARCH_METHOD. Choose 'grid' or 'bayesian'")

            print("\n✅ Optimization completed successfully!")

        elif RUN_MODE == "inference":
            if "best_model_checkpoint.pth" in CHECKPOINT_PATH or "your_sample.nii.gz" in SAMPLE_NIFTI_PATH:
                raise ValueError("Please set CHECKPOINT_PATH and SAMPLE_NIFTI_PATH before running inference mode.")

            inference_result = predict_single_sample_with_gradcam(
                checkpoint_path=CHECKPOINT_PATH,
                sample_path=SAMPLE_NIFTI_PATH,
                output_dir=INFERENCE_OUTPUT_DIR,
                target_class=TARGET_CLASS,
            )

            print("\n✅ Inference completed successfully!")
            print(f"Predicted class: {inference_result['predicted_class_name']}")
            if inference_result.get('figure_path'):
                print(f"Visualization file: {inference_result['figure_path']}")
            if inference_result.get('all_figure_paths'):
                print("All generated visualization files:")
                for path in inference_result['all_figure_paths']:
                    print(f"  {path}")


        elif RUN_MODE == "batch_inference":
            # Specify the trained model checkpoint
            #CHECKPOINT_PATH = r"<AUTHOR_DATA_ROOT>/MONAILabel/AD_VS_OUT_PY3.2.3_d264\GRID_SEARCH_1_densenet169_AttnSE_dropout_0.0_aug_2_bs_4_lr_1e-04_amax_1122_sched_cosine_loss_ce\checkpoints\checkpoint_epoch_132_best_acc.pth"
            
            # Specify the new dataset list file (Excel or CSV)
            NEW_DATASET_LIST_PATH = r"<AUTHOR_DATA_ROOT>/MONAILabel/AD\miriad_lastly_classified_processed_SyN_fast_ws_69.xlsx" # must contain a 'file_path' column; if a 'label' column is present, metrics will be computed
            
            # Result output directory
            BATCH_OUTPUT_DIR = r"<AUTHOR_DATA_ROOT>/MONAILabel/AD_VS\results3.7_128AUC\batch_inference_results_auc_miriad_69"
            
            # Run the batch inference
            final_report = run_batch_inference(
                checkpoint_path=CHECKPOINT_PATH,
                dataset_list_path=NEW_DATASET_LIST_PATH,
                output_dir=BATCH_OUTPUT_DIR
            )
            print("\n✅ Batch Inference completed successfully!")

        elif RUN_MODE == "test":
            print("🔍 Running unified test evaluation for all completed runs")
            eval_df = run_test_mode()
            print(f"\n✅ Test evaluation completed. Summary saved to {TEST_EVALUATION_OUTPUT_DIR}")
            print(eval_df[['run_id', 'run_dir', 'best_checkpoint', 'auc', 'f1_macro', 'accuracy']].head(20))

        elif RUN_MODE == "choose":
            print("🤖 Running intelligent model selection (choose mode)")
            best_output = run_choose_mode()
            print(f"\n✅ Model selection completed. Best run: {best_output['selected_run_dir']}")
            print(f"Saved selection to {os.path.join(MODEL_SELECTION_OUTPUT_DIR, 'best_model_selected.txt')}")

        elif RUN_MODE == "group_comparison":
            print("📊 Running group-level comparison mode")
            GROUP_CHECKPOINT_PATH = CHECKPOINT_PATH
            group_results = run_group_comparison_mode(
                model_checkpoint=GROUP_CHECKPOINT_PATH,
                n_samples_per_group=10
            )
            print(f"\n✅ Group comparison completed. Outputs saved under {GROUP_COMPARISON_OUTPUT_DIR}")
            print(group_results)

        else:
            raise ValueError("Invalid RUN_MODE. Choose 'train', 'inference', 'batch_inference', 'test', 'choose', or 'group_comparison'.")

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()