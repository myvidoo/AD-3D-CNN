# -*- coding: utf-8 -*-
"""
5-fold ensemble inference: load the 5 best weights → equally weighted average of the logits → softmax → output per-sample prediction probabilities.

Corresponds to the paper §3.2 (generates ensemble_test_predictions.csv, the data source for Table 2 / Figures 2–4 / Figure S4).

Usage:
    python inference/predict_ensemble.py
"""

import os
import sys
import json
import gc
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config, resolve_path

from common.utils import NUM_CLASSES, get_data_transforms, parse_config_from_string
from model.densenet169_attention import get_model
from monai.data import Dataset, DataLoader


def main():
    parser = argparse.ArgumentParser(description="5-fold ensemble inference")
    parser.add_argument("--run_id", type=int, default=None, help="Run ID (defaults to config best_run_id)")
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = cfg["data"].get("data_root") or None

    run_id = args.run_id or cfg["best_run_id"]
    grid_results = resolve_path(cfg, "data.grid_results")
    fixed_split_path = resolve_path(cfg, "data.fixed_split")
    weights_dir = resolve_path(cfg, "weights_dir")
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read the Run 128 configuration
    df = pd.read_csv(grid_results)
    row = df[df['run_id'] == run_id]
    assert not row.empty, f"run_id={run_id} not found"
    config = parse_config_from_string(row.iloc[0]['config'])
    print("[INFO] config:", config)

    # 2. Load the fixed test set (file_path holds a file name, which is joined with data_root)
    with open(fixed_split_path, 'r') as f:
        fixed_split = json.load(f)
    test_files = fixed_split.get('test_split', [])
    print(f"[INFO] test samples: {len(test_files)}")

    _, test_transforms = get_data_transforms(0, a_max=config.get('a_max', 1100))
    test_data = []
    for item in test_files:
        fp = item['file_path']
        if data_root and not os.path.isabs(fp):
            fp = os.path.join(data_root, fp)
        test_data.append({'image': fp, 'label': int(item['label'])})
    test_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 3. Load the 5 models
    models = []
    for i in range(1, 6):
        path = os.path.join(weights_dir, f"fold_{i}_best_geo.pth")
        assert os.path.exists(path), f"missing: {path}"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = get_model(config['model_name'], NUM_CLASSES, device,
                          config.get('dropout_rate', 0.0), config.get('attention_type', 'none'))
        sd = ckpt['model_state_dict']
        if hasattr(model, 'base_model') and not any(k.startswith('base_model.') for k in sd):
            model.base_model.load_state_dict(sd)
        else:
            model.load_state_dict(sd)
        model.eval()
        models.append(model)
    print(f"[INFO] loaded {len(models)} models")

    # 4. Ensemble inference
    records = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Ensemble inference")):
            inputs = batch['image'].to(device)
            label = int(batch['label'].cpu().numpy()[0])
            logits_list = []
            for m in models:
                logits = m(inputs)
                logits_list.append(logits.cpu().numpy().flatten())
            mean_logits = np.mean(np.array(logits_list), axis=0)
            probs = F.softmax(torch.tensor(mean_logits), dim=0).numpy()
            pred = int(np.argmax(probs))
            # the file_path column stores the file name (path stripped, consistent with the released data/ensemble_test_predictions.csv)
            records.append({
                'file_path': test_files[i]['file_path'],
                'true_label': label,
                'prob_cn': probs[0], 'prob_mci': probs[1], 'prob_ad': probs[2],
                'pred_label': pred
            })

    out = pd.DataFrame(records)
    out_csv = os.path.join(output_dir, "ensemble_test_predictions.csv")
    out.to_csv(out_csv, index=False)
    print(f"[INFO] saved -> {out_csv}  (shape {out.shape})")

    # 5. Quick verification (should match the paper: ACC 0.8681 / Macro-AUC 0.9630 / Macro-F1 0.8675)
    from sklearn.metrics import roc_auc_score, f1_score
    acc = (out['pred_label'] == out['true_label']).mean()
    y_true = out['true_label'].values
    y_prob = out[['prob_cn', 'prob_mci', 'prob_ad']].values
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    f1 = f1_score(y_true, out['pred_label'].values, average='macro')
    print(f"[VERIFY] ACC={acc:.4f} (expect 0.8681), Macro-AUC={auc:.4f} (expect 0.9630), "
          f"Macro-F1={f1:.4f} (expect 0.8675)")

    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
