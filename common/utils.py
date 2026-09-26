# -*- coding: utf-8 -*-
"""
Shared function library: data pipeline, training/validation loops, checkpoint I/O,
configuration parsing and data splitting.

Source: extracted verbatim from cnn_model_v3_7_d169_group_comparison.py, with
     path/device parameterisation: all file paths (EXCEL_PATH / FIXED_DATA_SPLIT_JSON /
     GLOBAL_TRAINING_RESULTS_CSV) and device were changed from module-level globals to
     function parameters, injected uniformly by config.

This module reads no hard-coded absolute paths.
"""

import os
import ast
import re
import json
import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandRotated, RandFlipd, RandZoomd, RandGaussianNoised, RandAdjustContrastd, EnsureTyped,
)

from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm


# ==================== Methodological constants (not paths; safe as module-level constants) ====================#

CLASS_NAMES = ["Class_0", "Class_1", "Class_2"]
LABEL_MAP_FLOAT_TO_INT = {0.0: 0, 0.5: 1, 1.0: 2}
NUM_CLASSES = 3
INPUT_SHAPE = (182, 218, 182)

NUM_WORKERS = 4
CHECKPOINT_INTERVAL = 20
EARLY_STOP_PATIENCE = 15
LABEL_SMOOTHING = 0.05
GLOBAL_SEED = 42


# ==================== Data augmentation parameters ====================#

AUGMENTATION_PARAMS = {
    0: {
        "rot_range": 0.0, "rot_prob": 0.0, "flip_lr_prob": 0.0,
        "zoom_range": (1.0, 1.0), "zoom_prob": 0.0,
        "noise_std": 0.0, "noise_prob": 0.0,
        "contrast_gamma": (1.0, 1.0), "contrast_prob": 0.0,
    },
    1: {  # Very weak augmentation (the first choice for registered data)
        "rot_range": 0.035,  # about 2 degrees, a tiny perturbation
        "rot_prob": 0.2,
        "flip_lr_prob": 0.5,  # left-right flip only
        "zoom_range": (0.99, 1.01), "zoom_prob": 0.1,
        "noise_std": 0.005, "noise_prob": 0.1,
        "contrast_gamma": (0.98, 1.02), "contrast_prob": 0.1,
    },
    2: {  # Mild augmentation
        "rot_range": 0.07,  # about 4 degrees
        "rot_prob": 0.3,
        "flip_lr_prob": 0.5,
        "zoom_range": (0.97, 1.03), "zoom_prob": 0.2,
        "noise_std": 0.01, "noise_prob": 0.15,
        "contrast_gamma": (0.95, 1.05), "contrast_prob": 0.15,
    },
}


# ==================== Data pipeline ====================#

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_data_transforms(augmentation_intensity, a_max=1100):
    """Get the data transforms"""
    if augmentation_intensity > 2:
        augmentation_intensity = 2

    current_aug_params = AUGMENTATION_PARAMS[augmentation_intensity]

    train_transforms = Compose([
        LoadImaged(keys=["image"], ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=a_max, b_min=0.0, b_max=1.0, clip=True
        ),
        RandRotated(
            keys=["image"], range_x=current_aug_params['rot_range'],
            range_y=current_aug_params['rot_range'], range_z=current_aug_params['rot_range'],
            prob=current_aug_params['rot_prob'], keep_size=True, padding_mode='zeros'
        ),
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


# ==================== Training / validation ====================#

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


# ==================== checkpoint I/O ====================#

def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, metrics,
                    config=None, tag=None, num_classes=NUM_CLASSES, extra_state=None):
    """Save a checkpoint; a tag may be added to mark the best metric, etc.

    extra_state: optional dict, used to additionally save the local state of the training loop
                for resumable training (such as best_auc / best_f1 / best_acc /
                patience_counter); it is written to the checkpoint's 'extra_state' field and
                read back when resuming.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'current_lr': optimizer.param_groups[0]['lr'],
        'metrics': metrics,
        'model_name': config.get('model_name', 'grid_search_model') if config else 'grid_search_model',
        'num_classes': num_classes,
        'config': config if config else {}
    }
    if extra_state is not None:
        checkpoint['extra_state'] = extra_state
    if tag == 'latest':
        # Overwriting latest checkpoint with a fixed file name, for resumable training (loses at most 1 epoch)
        name = "checkpoint_latest"
    else:
        name = f"checkpoint_epoch_{epoch:03d}"
        if tag:
            name += f"_{tag}"
    path = os.path.join(checkpoint_dir, f"{name}.pth")
    torch.save(checkpoint, path)
    return path


def find_latest_checkpoint(checkpoint_dir, pattern="checkpoint_epoch_*.pth"):
    """Find the path of the checkpoint file with the largest epoch in checkpoint_dir.

    Only files whose names start with 'checkpoint_epoch_' and end with '.pth' are matched
    (including tagged best_* checkpoints and untagged periodic checkpoints); the largest is
    taken by the epoch number in the file name.
    Returns None when there is no match.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir)
             if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if not files:
        return None

    def _epoch_num(fname):
        m = re.search(r"epoch_(\d+)", fname)
        return int(m.group(1)) if m else 0

    latest = max(files, key=_epoch_num)
    return os.path.join(checkpoint_dir, latest)


def safe_torch_load(model_path, map_location='cpu'):
    """checkpoint loading compatible with different PyTorch versions"""
    try:
        return torch.load(model_path, map_location=map_location)
    except Exception:
        return torch.load(model_path, map_location=map_location, weights_only=False)


# ==================== Configuration parsing ====================#

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


def _coerce_config_value(value):
    """Try to restore a configuration value from a CSV/string to its original type"""
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
    """Fill in the configuration required for inference and normalise types"""
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


def parse_config_from_run_dir(run_dir):
    """Parse the configuration from the run directory name, compatible with old checkpoints that contain no config"""
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


# ==================== Model loading (for inference) ====================#

def build_model_from_checkpoint(checkpoint_obj, config, device, num_classes=NUM_CLASSES):
    """Rebuild the model from the parsed configuration and load the weights"""
    from model.densenet169_attention import get_model

    if isinstance(checkpoint_obj, nn.Module):
        model = checkpoint_obj.to(device)
        model.eval()
        return model

    model = get_model(
        model_name=config['model_name'],
        num_classes=num_classes,
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


def load_ensemble_models(checkpoint_paths, config, device):
    """Load the 5-fold ensemble models (one model per checkpoint, sharing the same config)"""
    models = []
    for path in checkpoint_paths:
        checkpoint_obj = safe_torch_load(path, map_location=device)
        model = build_model_from_checkpoint(checkpoint_obj, config, device)
        models.append(model)
    return models


# ==================== Fixed test-set split ====================#

def save_fixed_data_split(test_split, train_ratio, val_ratio, test_ratio,
                          random_state=42, file_path="fixed_data_split.json"):
    """Save the fixed test-set file list so that all subsequent evaluations use the same test split."""
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
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


def load_fixed_data_split(file_path="fixed_data_split.json"):
    """Load the saved fixed test-set split information."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load fixed data split file: {e}")
        return None


# ==================== Training result logging ====================#

def append_global_training_result(result, csv_path="all_training_results.csv"):
    """Append each training result to the global CSV file."""
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    result_df = pd.DataFrame([result])
    if not os.path.exists(csv_path):
        result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    else:
        result_df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')


def load_dataset_from_excel(excel_path, label_map=LABEL_MAP_FLOAT_TO_INT, data_root=None):
    """Read the data list Excel and return [{'image': full path, 'label': int}, ...], keeping only files that exist.

    If the file_path column holds a relative file name and data_root is provided, data_root is
    joined on; if file_path is already an absolute path it is used as-is (compatible with lists
    whose paths have not been stripped).
    """
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['file_path', 'label'])
    df['label_int'] = df['label'].map(label_map)
    data_list = []
    for _, row in df.iterrows():
        fp = row["file_path"]
        if data_root and not os.path.isabs(fp):
            fp = os.path.join(data_root, fp)
        fp = os.path.normpath(fp)  # normalise path separators to ease matching with fixed_split
        if os.path.exists(fp):
            data_list.append({"image": fp, "label": int(row["label_int"])})
    return data_list


# ==================== Grad-CAM helpers ====================#

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
    """Prepare a model instance that can safely be backpropagated through, for Grad-CAM."""
    if model is None:
        return None
    model.eval()
    base_model = model.base_model if hasattr(model, "base_model") else model
    if base_model is not None:
        _replace_inplace_activations(base_model, inplace=False)
    return model
