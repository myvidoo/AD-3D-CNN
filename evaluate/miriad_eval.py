# -*- coding: utf-8 -*-
"""
MIRIAD external validation: 5-fold ensemble inference + binary/three-class evaluation (normalised v2).

Corresponds to the paper §2.6 / §3.4 / S7 (external-validation AUC 0.8790, sensitivity 76.1%, specificity 87.0%).

Changes relative to the older evaluate/miriad_eval.py:
  1. Data import changed from "reading the Excel index (file_path / label)" to "direct directory scan":
     *.nii.gz is glob-scanned directly under the label subdirectories {0: HC, 1: AD}, and the label is determined by the directory name.
  2. The Bootstrap 95% CI (2000 stratified bootstrap resamples, seed=42) is now written to disk.
  3. The exact-match accuracy (three-class alignment, 54/69 = 0.7826) is now written to disk instead of only printed.

All paths are provided by config.yaml: data.miriad_root / weights_dir / output_dir.
The root directory of the preprocessed MIRIAD images (containing the 0/1 label subdirectories) must first be filled in config.yaml.

Usage:
    python evaluate/miriad_eval.py
"""

import os
import sys
import glob
import gc

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config, resolve_path                                 # noqa: E402
from common.utils import NUM_CLASSES, get_data_transforms, load_ensemble_models  # noqa: E402

# Run 128 best configuration (consistent with best_run_id=128 in config.yaml)
MODEL_CONFIG = {
    "model_name": "densenet169",
    "attention_type": "axial",
    "dropout_rate": 0.0,
    "a_max": 1100,
}

# Label subdirectory → three-class model class index (MIRIAD has no MCI; only HC=0→CN (class 0), AD=1→AD (class 2))
LABEL_DIRS = {"0": 0, "1": 2}

N_BOOT = 2000
BOOT_SEED = 42


def bootstrap_auc_ci(y_true, y_score, n_boot=N_BOOT, seed=BOOT_SEED):
    """Estimate the 95% confidence interval of the AUC by 2000 stratified bootstrap resamples (same definition as for the internal test set).

    Stratification: each class is resampled with replacement separately and then merged, so that the class proportions of every bootstrap sample are unchanged.
    The 95% CI is taken from the 2.5 / 97.5 percentiles of the bootstrap AUC distribution.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    class_indices = {c: np.where(y_true == c)[0] for c in np.unique(y_true)}
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        boot_idx = []
        for idx in class_indices.values():
            boot_idx.extend(rng.choice(idx, size=len(idx), replace=True).tolist())
        rng.shuffle(boot_idx)
        boot_idx = np.asarray(boot_idx)
        yt, ys = y_true[boot_idx], y_score[boot_idx]
        if len(np.unique(yt)) < 2:
            aucs.append(0.5)  # treat as chance level when positive or negative samples are missing
        else:
            aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    lo = np.percentile(aucs, 2.5)
    hi = np.percentile(aucs, 97.5)
    return lo, hi


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    miriad_root = cfg["data"]["miriad_root"]
    weights_dir = resolve_path(cfg, "weights_dir")
    output_dir = os.path.join(cfg["output_dir"], "miriad_ensemble_eval")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Direct directory scan of the data (the directory name is the label)
    test_data, valid_paths = [], []
    for dname, cls in LABEL_DIRS.items():
        for p in sorted(glob.glob(os.path.join(miriad_root, dname, "*.nii.gz"))):
            test_data.append({"image": p, "label": cls})
            valid_paths.append(p)
    n_hc = sum(1 for d in test_data if d["label"] == 0)
    n_ad = sum(1 for d in test_data if d["label"] == 2)
    print(f"Samples from the directory scan: {len(test_data)} (HC={n_hc}, AD={n_ad})")
    if not test_data:
        print("\nNo MIRIAD images found (configure data.miriad_root in config.yaml); "
              "skipping this script and exiting 0.")
        return

    # 2. Load the 5-fold ensemble models
    checkpoint_paths = [os.path.join(weights_dir, f"fold_{i}_best_geo.pth") for i in range(1, 6)]
    models = load_ensemble_models(checkpoint_paths, MODEL_CONFIG, device)
    _, test_transforms = get_data_transforms(0, a_max=MODEL_CONFIG["a_max"])

    from monai.data import Dataset
    test_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 3. Ensemble inference (also recording the single-fold AD probabilities)
    all_y_true, all_ensemble_probs = [], []
    all_fold_probs = [[] for _ in models]
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="MIRIAD ensemble inference"):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            all_y_true.extend(labels.cpu().numpy())
            fold_logits = [model(inputs) for model in models]
            for k, logits in enumerate(fold_logits):
                all_fold_probs[k].append(torch.softmax(logits, dim=1).cpu().numpy())
            ens = torch.softmax(torch.mean(torch.stack(fold_logits), dim=0), dim=1)
            all_ensemble_probs.extend(ens.cpu().numpy())

    all_ensemble_probs = np.array(all_ensemble_probs)
    all_y_true = np.array(all_y_true)
    y_pred_3class = np.argmax(all_ensemble_probs, axis=1)

    # 4. Binary-classification view (AD vs HC, score = P(AD))
    # Note: MIRIAD has no MCI group, so in the model's three-class output anything "predicted as non-CN (i.e. MCI or AD)" is counted as AD-positive,
    #     consistent with the paper's definition (sensitivity 35/46=0.7609, specificity 20/23=0.8696).
    y_true_binary = (all_y_true == 2).astype(int)
    y_pred_binary = (y_pred_3class != 0).astype(int)   # predicted as non-CN → AD-positive
    prob_ad = all_ensemble_probs[:, 2]
    auc = roc_auc_score(y_true_binary, prob_ad)

    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm_binary.ravel()
    acc = accuracy_score(y_true_binary, y_pred_binary)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_macro = f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)

    # Exact-match accuracy (three-class alignment: HC→CN, AD→AD)
    exact_acc = accuracy_score(all_y_true, y_pred_3class)

    # Per-class OVR AUC
    auc_cn = roc_auc_score((all_y_true == 0).astype(int), all_ensemble_probs[:, 0])
    auc_ad = roc_auc_score((all_y_true == 2).astype(int), all_ensemble_probs[:, 2])

    # Bootstrap 95% CI
    ci_lo, ci_hi = bootstrap_auc_ci(y_true_binary, prob_ad)

    # 3×3 confusion matrix (rows = true HC/AD/MCI (empty))
    cm_3x3 = confusion_matrix(all_y_true, y_pred_3class, labels=[0, 1, 2])

    # Single-fold AD vs HC AUC
    fold_aucs = []
    for k in range(len(models)):
        fp_ad = np.concatenate(all_fold_probs[k], axis=0)[:, 2]
        fold_aucs.append(roc_auc_score(y_true_binary, fp_ad))
    mean_fold_auc = float(np.mean(fold_aucs))

    # 5. Print the report
    print("\n" + "=" * 62)
    print("MIRIAD external validation report (5-fold ensemble, direct directory scan)")
    print("=" * 62)
    print(f"Samples: {len(test_data)} (HC={n_hc}, AD={n_ad})")
    print(f"\n[1] Exact-match accuracy (three-class alignment): {exact_acc:.4f}  (paper: 0.7826)")
    print(f"[2] Binary classification (AD vs HC):")
    print(f"    Accuracy:    {acc:.4f}")
    print(f"    AUC:         {auc:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]  (paper: 0.8790 [0.793, 0.953])")
    print(f"    Sensitivity: {sensitivity:.4f}  (paper: 0.7609)")
    print(f"    Specificity: {specificity:.4f}  (paper: 0.8696)")
    print(f"    F1 Macro:    {f1_macro:.4f}  (paper: 0.7870)")
    print(f"\n[3] Per-class OVR AUC:  CN={auc_cn:.4f}  AD={auc_ad:.4f}")
    print(f"[4] 3×3 confusion matrix (rows = true, columns = predicted CN/MCI/AD):")
    print(cm_3x3)
    print(f"[5] Single-fold AUC: {[round(a, 4) for a in fold_aucs]}")
    print(f"    Single-fold mean: {mean_fold_auc:.4f}  (paper: 0.8278)")

    # 6. Write to disk
    pd.DataFrame({
        "Metric": ["Exact-match Accuracy", "Binary Accuracy", "AUC (AD vs HC)",
                   "Sensitivity", "Specificity", "F1 Macro"],
        "Value": [exact_acc, acc, auc, sensitivity, specificity, f1_macro]
    }).to_csv(os.path.join(output_dir, "miriad_ensemble_metrics.csv"), index=False)

    pd.DataFrame({
        "Class": ["CN (HC)", "MCI (N/A)", "AD"],
        "AUC": [auc_cn, np.nan, auc_ad]
    }).to_csv(os.path.join(output_dir, "miriad_class_auc.csv"), index=False)

    pd.DataFrame(cm_3x3, index=["True_HC", "True_MCI(N/A)", "True_AD"],
                 columns=["Pred_CN", "Pred_MCI", "Pred_AD"]).to_csv(
        os.path.join(output_dir, "miriad_confusion_matrix_3x3.csv"))

    pd.DataFrame({
        "file_path": valid_paths,
        "true_label": all_y_true,
        "pred_label": y_pred_3class,
        "prob_CN": all_ensemble_probs[:, 0],
        "prob_MCI": all_ensemble_probs[:, 1],
        "prob_AD": all_ensemble_probs[:, 2]
    }).to_csv(os.path.join(output_dir, "miriad_ensemble_detailed_predictions.csv"), index=False)

    pd.DataFrame({
        "fold": list(range(1, len(models) + 1)),
        "single_fold_auc_AD_vs_HC": fold_aucs
    }).to_csv(os.path.join(output_dir, "miriad_single_fold_auc.csv"), index=False)

    pd.DataFrame([{"single_fold_mean_auc": mean_fold_auc}]).to_csv(
        os.path.join(output_dir, "miriad_single_fold_mean_auc.csv"), index=False)

    pd.DataFrame({
        "Metric": ["AUC (AD vs HC)", "Bootstrap_CI_low", "Bootstrap_CI_high", "N_bootstrap", "Seed"],
        "Value": [auc, ci_lo, ci_hi, N_BOOT, BOOT_SEED]
    }).to_csv(os.path.join(output_dir, "miriad_bootstrap_ci.csv"), index=False)

    print(f"\nResults saved to: {output_dir}")

    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
