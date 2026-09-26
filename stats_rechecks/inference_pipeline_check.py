# -*- coding: utf-8 -*-
"""
Correctness verification script: runs the ADNI internal test set (fixed 144-sample split) through exactly the same inference code path as the OASIS-2 validation,
to confirm that the inference pipeline is correct. If the internal test-set accuracy matches the paper's 5-fold result, the low OASIS-2 performance is a genuine domain shift.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

# [AD_CNN_code adaptation] WORKSPACE → the root of this repository; the model definition uses the legacy training script
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "03_training", "legacy"))
from cnn_model_v3_7_d169_group_comparison import (
    get_data_transforms, get_model, NUM_CLASSES, device, NUM_WORKERS
)

SPLIT_JSON = os.path.join(WORKSPACE, "data", "fixed_data_split.json")
CHECKPOINT_DIR = os.path.join(WORKSPACE, "weights")

# The packaged fixed_data_split.json contains only relative file names, which must be joined with data.data_root from config.yaml to form absolute paths
sys.path.insert(0, WORKSPACE)
from config import load_config, resolve_path                      # noqa: E402

_cfg = load_config()
DATA_ROOT = _cfg["data"].get("data_root") or ""


def _abs(p):
    """Join a relative path with data_root; if data_root is not configured, return it unchanged."""
    if os.path.isabs(p) or not DATA_ROOT:
        return p
    return os.path.join(DATA_ROOT, p)


def load_models():
    models = []
    for i in range(1, 6):
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, f"fold_{i}_best_geo.pth"),
                                map_location=device, weights_only=False)
        model = get_model(model_name='densenet169', num_classes=NUM_CLASSES, device=device,
                          dropout_rate=0.0, attention_type='axial')
        sd = checkpoint['model_state_dict']
        if hasattr(model, 'base_model') and not any(k.startswith('base_model.') for k in sd):
            model.base_model.load_state_dict(sd)
        else:
            model.load_state_dict(sd)
        model.eval()
        models.append(model)
    return models

def main():
    split = json.load(open(SPLIT_JSON, encoding='utf-8'))
    test_list = split['test_split']
    test_data = [{"image": _abs(r['file_path']), "label": int(r['label'])}
                 for r in test_list if os.path.exists(_abs(r['file_path']))]
    print(f"ADNI internal test set: {len(test_data)}/{len(test_list)} samples present")
    if not test_data:
        print("\nNo test images found (configure ADNI data via data.data_root in config.yaml); skipping this check and exiting 0.")
        return

    models = load_models()
    from monai.data import Dataset
    _, test_transforms = get_data_transforms(0, a_max=1100)
    loader = torch.utils.data.DataLoader(Dataset(data=test_data, transform=test_transforms),
                                         batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    y_true, probs = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch['image'].to(device)
            y_true.extend(batch['label'].cpu().numpy())
            logits = torch.mean(torch.stack([m(inputs) for m in models]), dim=0)
            probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.array(probs); y_true = np.array(y_true); y_pred = probs.argmax(1)

    acc = accuracy_score(y_true, y_pred)
    aucs = []
    for i in range(3):
        yb = (y_true == i).astype(int)
        if yb.min() != yb.max():
            aucs.append(roc_auc_score(yb, probs[:, i]))
    print(f"\nADNI internal test set (5-fold ensemble): Accuracy={acc:.4f}, per-class AUC={[f'{a:.4f}' for a in aucs]}")

if __name__ == "__main__":
    main()
