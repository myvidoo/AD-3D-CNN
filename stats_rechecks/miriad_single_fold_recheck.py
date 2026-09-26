# -*- coding: utf-8 -*-
"""
Verification script: MIRIAD external validation — single-fold model AUC recomputation (paper §3.4 / S7)
=========================================================================
Background: §3.4 of the paper / supplementary S7 report the MIRIAD single-fold model AD vs HC AUC as
        0.8459, 0.8790, 0.6928, 0.8658, 0.8554 (mean 0.8278), reported alongside the ensemble model
        AUC of 0.8790. The paper states explicitly that the two are not the same evaluation target,
        that no significance test was performed on the difference, and that no ensemble gain is claimed on this basis (see the main text §2.4 / §3.4).
        This script re-evaluates each single-fold model's
        AD vs HC AUC on MIRIAD using the 5-fold best checkpoints, and compares the 5-fold mean against the paper's numbers.
Data sources:
  - Model weights: <AUTHOR_DATA_ROOT>/MONAILabel/AD_VS\\results3.7\\results_5fold_geo_resume\\checkpoints\\fold_*_best_geo.pth
  - MIRIAD list: <AUTHOR_DATA_ROOT>/MONAILabel/AD\\miriad_lastly_classified_processed_SyN_fast_ws_69.xlsx
  - Dependency: cnn_model_v3_7_d169_group_comparison.py (model definition; must be in the same directory or on sys.path)
Run    : python verify_script_MIRIAD_single_fold_AUC_recompute.py > verification_result_MIRIAD_single_fold_AUC.txt
Note   : this script is a verification script written during the paper-preparation stage; it requires the original training environment of
        torch + GPU + MONAI. Runtime is about 5 folds × 69 inference samples.
"""
import os
import gc
import sys

# A Chinese-locale Windows console defaults to GBK; this script prints characters such as ✓/✗/η²/χ², so the encoding error strategy must be relaxed
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass


import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# ---- model definition and data pipeline (shared with the training/evaluation scripts) ----
# Note: cnn_model_v3_7_d169_group_comparison.py lives in the paper-preparation directory 03;
#       at runtime it must be on sys.path (the original script ran under <AUTHOR_DATA_ROOT>/MONAILabel/AD_VS).
SYS_PATHS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "03_training", "legacy")),
]
for p in SYS_PATHS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
from cnn_model_v3_7_d169_group_comparison import (
    get_data_transforms, get_model, NUM_CLASSES, CLASS_NAMES, device, NUM_WORKERS
)

MIRIAD_EXCEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "data_lists",
                                                 "MIRIAD_69_subject_master_list_miriad_lastly_classified_processed_SyN_fast_ws_69.xlsx"))
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))


import re
# ---- MIRIAD image root (data.miriad_root in config.yaml) ----
# file_path values in the archived list may contain placeholders or old machine paths; when the
# stored path does not exist it is resolved against miriad_root by shortening the suffix step by
# step (a reproducer only has to set miriad_root in config.yaml; the xlsx needs no editing).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import load_config, resolve_path  # noqa: E402
_CFG = load_config()
MIRIAD_ROOT = resolve_path(_CFG, "data.miriad_root")


def locate_image(stored):
    """Return an existing image path: check as-is first, then shorten the suffix under MIRIAD_ROOT."""
    if stored and os.path.exists(str(stored)):
        return str(stored)
    comps = [c for c in re.split(r"[\\/]+", str(stored)) if c]
    for k in range(len(comps)):
        cand = os.path.join(MIRIAD_ROOT, *comps[k:])
        if os.path.exists(cand):
            return cand
    return str(stored)


def load_single_model(path):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = get_model(model_name='densenet169', num_classes=NUM_CLASSES,
                      device=device, dropout_rate=0.0, attention_type='axial')
    state_dict = checkpoint['model_state_dict']
    if hasattr(model, 'base_model') and not any(k.startswith('base_model.') for k in state_dict):
        model.base_model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    print("=" * 66)
    print("MIRIAD external validation: single-fold model AUC recomputation (paper §3.4 / S7)")
    print("=" * 66)

    # 1. Read the MIRIAD data
    df = pd.read_excel(MIRIAD_EXCEL_PATH)
    df = df.dropna(subset=['file_path', 'label'])
    df['label_int'] = df['label'].astype(int)
    df = df[df['label_int'].isin([0, 1])]  # HC / AD only
    test_data = []
    for _, r in df.iterrows():
        _p = locate_image(r["file_path"])
        if os.path.exists(_p):
            test_data.append({"image": _p, "label": int(r["label_int"])})
    y_true = np.array([t["label"] for t in test_data])
    print(f"Valid samples: {len(test_data)} (HC={int((y_true==0).sum())}, AD={int((y_true==1).sum())})")
    if not test_data:
        print("\nNo MIRIAD images found (configure data.miriad_root in config.yaml); skipping this script and exiting 0.")
        return

    from monai.data import Dataset
    _, test_transforms = get_data_transforms(0, a_max=1100)
    test_loader = torch.utils.data.DataLoader(
        Dataset(data=test_data, transform=test_transforms),
        batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 2. Per-fold inference, accumulating the ensemble predictions
    fold_aucs = []
    all_fold_logits = []      # (n_samples, 5, 3)
    ensemble_probs_by_fold = []
    for fold in range(1, 6):
        ckpt = os.path.join(CHECKPOINT_DIR, f"fold_{fold}_best_geo.pth")
        if not os.path.exists(ckpt):
            print(f"[WARN] checkpoint missing: {ckpt}; skipping")
            continue
        model = load_single_model(ckpt)
        fold_logits = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['image'].to(device)
                fold_logits.append(model(inputs).cpu().numpy())
        fold_logits = np.concatenate(fold_logits, axis=0)
        fold_probs = torch.softmax(torch.tensor(fold_logits), dim=1).numpy()
        auc_fold = roc_auc_score(y_true, fold_probs[:, 2])  # AD-class probability
        fold_aucs.append(auc_fold)
        all_fold_logits.append(fold_logits)
        print(f"  fold {fold}: single-fold AUC (AD vs HC) = {auc_fold:.4f}")
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 3. Ensemble AUC (equally weighted logit average)
    ensemble_logits = np.mean(all_fold_logits, axis=0)
    ensemble_probs = torch.softmax(torch.tensor(ensemble_logits), dim=1).numpy()
    auc_ens = roc_auc_score(y_true, ensemble_probs[:, 2])

    mean_fold = np.mean(fold_aucs)
    print("\n" + "=" * 66)
    print("Summary of results")
    print("=" * 66)
    print(f"Single-fold AUC list: {[round(a, 4) for a in fold_aucs]} (paper S7: [0.8459, 0.8790, 0.6928, 0.8658, 0.8554])")
    print(f"Single-fold AUC mean: {mean_fold:.4f} (paper: 0.8278)")
    print(f"Ensemble AUC: {auc_ens:.4f} (paper: 0.8790)")
    print(f"Difference between the ensemble and the single-fold mean: {(auc_ens - mean_fold) * 100:.2f} percentage points"
          f" (the paper performs no significance test on this difference and claims no ensemble gain)")
    ok = abs(mean_fold - 0.8278) < 0.01 and abs(auc_ens - 0.8790) < 0.01
    print(f"Verdict: {'✔ single-fold/ensemble AUC agrees with the paper' if ok else '✘ discrepancy found, please check'}")

    # 4. Write to disk
    out = pd.DataFrame({"fold": list(range(1, len(fold_aucs) + 1)),
                        "single_fold_auc": fold_aucs})
    out.loc[len(out)] = ["ensemble", auc_ens]
    here = os.path.dirname(os.path.abspath(__file__))
    out.to_csv(os.path.join(here, "miriad_single_fold_auc_recheck.csv"), index=False)
    print(f"\nResults saved: miriad_single_fold_auc_recheck.csv")


if __name__ == "__main__":
    main()
