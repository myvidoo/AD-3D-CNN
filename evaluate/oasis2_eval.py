# -*- coding: utf-8 -*-
"""
OASIS-2 external-validation evaluation script (normalised v3).

Corresponds to the paper §2.6 / §3.4 / S7 (second external cohort: CN vs AD binary AUC 0.757, 95% CI 0.633–0.873).

Changes relative to the older script run_oasis2_newpipeline_full_evaluation.py:
  1. Data import changed to a "direct directory scan": *.nii.gz is scanned under the label subdirectories {0: CN, 0.5: MCI, 1: AD}.
  2. The main CN vs AD binary-classification results are now written to disk:
     - normalised score P(AD)/[P(AD)+P(CN)];
     - AUC + Bootstrap 95% CI (2000 stratified bootstrap resamples, seed=42);
     - one-sided Mann-Whitney U test (CN scores vs AD scores, alternative='less').
  3. The AD-group accuracy stratified by MMSE is now written to disk (MMSE linked to the demographics table via the MRI ID).

Imports go through the repository's standard modules (config / common.utils / model.densenet169_attention).
All paths are provided by config.yaml: data.oasis2_root / data.oasis2_demo / weights_dir / output_dir.
The root directory of the preprocessed OASIS-2 images (containing the 0/0.5/1 label subdirectories) must first be filled in config.yaml;
the basic-information list data/oasis2_basic_info.csv (the 147-sample subset used in this study) is provided with the repository.

Usage (same environment as the paper's OASIS-2 results, monai 1.5.2 / torch 2.13):
    python evaluate/oasis2_eval.py
"""

import os
import sys
import glob
import re
import gc

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, \
    f1_score, confusion_matrix, precision_recall_fscore_support
from scipy.stats import mannwhitneyu
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

# Label subdirectory → three-class model class index (0→CN, 0.5→MCI, 1→AD)
LABEL_DIRS = {"0": 0, "0.5": 1, "1": 2}
CLASS_NAMES = ["CN", "MCI", "AD"]

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


def extract_mri_id(file_path):
    """Extract the OASIS-2 MRI ID from the file name (e.g. OAS2_0001_MR2)."""
    m = re.search(r"OAS2_\d+_MR\d+", os.path.basename(file_path))
    return m.group(0) if m else None


def mmse_bin(m):
    """MMSE strata: <20 / 20-24 / >=25."""
    if m < 20:
        return "<20"
    if m < 25:
        return "20-24"
    return ">=25"


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oasis2_root = cfg["data"]["oasis2_root"]
    oasis2_demo = resolve_path(cfg, "data.oasis2_demo")
    weights_dir = resolve_path(cfg, "weights_dir")
    output_dir = os.path.join(cfg["output_dir"], "oasis2_ensemble_eval")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Direct directory scan of the data (the directory name is the label)
    test_data, valid_paths = [], []
    for dname, cls in LABEL_DIRS.items():
        for p in sorted(glob.glob(os.path.join(oasis2_root, dname, "*.nii.gz"))):
            test_data.append({"image": p, "label": cls})
            valid_paths.append(p)
    n_per_class = {CLASS_NAMES[i]: sum(1 for d in test_data if d["label"] == i) for i in range(3)}
    print(f"Samples from the directory scan: {len(test_data)} ({n_per_class})")
    if not test_data:
        print("\nNo OASIS-2 images found (configure data.oasis2_root in config.yaml); "
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

    # 3. Ensemble inference (recording the per-fold probabilities)
    all_y_true, all_probs = [], []
    fold_probs = [[] for _ in models]
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="OASIS-2 ensemble inference"):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            all_y_true.extend(labels.cpu().numpy())
            fold_logits = [model(inputs) for model in models]
            for k, logits in enumerate(fold_logits):
                fold_probs[k].extend(torch.softmax(logits, dim=1).cpu().numpy())
            ens = torch.softmax(torch.mean(torch.stack(fold_logits), dim=0), dim=1)
            all_probs.extend(ens.cpu().numpy())

    all_probs = np.array(all_probs, dtype=np.float64)  # promote to float64 to avoid floating-point precision loss in the normalised score
    all_y_true = np.array(all_y_true)
    y_pred = np.argmax(all_probs, axis=1)

    # 4. Three-class overall metrics
    acc = accuracy_score(all_y_true, y_pred)
    bacc = balanced_accuracy_score(all_y_true, y_pred)
    f1_macro = f1_score(all_y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(all_y_true, y_pred, average="weighted", zero_division=0)
    per_class_auc = [roc_auc_score((all_y_true == i).astype(int), all_probs[:, i]) for i in range(3)]
    auc_macro = float(np.mean(per_class_auc))
    cm3 = confusion_matrix(all_y_true, y_pred, labels=[0, 1, 2])
    prec, rec, f1s, sup = precision_recall_fscore_support(
        all_y_true, y_pred, labels=[0, 1, 2], zero_division=0)

    # 5. Main CN vs AD binary result (excluding MCI, normalised score P(AD)/[P(AD)+P(CN)])
    cn_ad_mask = np.isin(all_y_true, [0, 2])
    y_cn_ad = (all_y_true[cn_ad_mask] == 2).astype(int)          # CN=0, AD=1
    score_cn_ad = all_probs[cn_ad_mask, 2] / (
        all_probs[cn_ad_mask, 2] + all_probs[cn_ad_mask, 0])     # normalised P(AD)
    auc_cn_ad = roc_auc_score(y_cn_ad, score_cn_ad)
    ci_lo, ci_hi = bootstrap_auc_ci(y_cn_ad, score_cn_ad)
    acc_cn_ad = accuracy_score(y_cn_ad, (score_cn_ad > 0.5).astype(int))

    # One-sided Mann-Whitney U test: the CN scores should be lower than the AD scores
    n_cn = int((y_cn_ad == 0).sum())
    n_ad = int((y_cn_ad == 1).sum())
    u_stat, p_mw = mannwhitneyu(score_cn_ad[y_cn_ad == 0], score_cn_ad[y_cn_ad == 1],
                                alternative="less")
    # scipy returns U1 for the first sample (CN); the paper reports the complementary U2 = n1*n2 − U1
    u_report = n_cn * n_ad - u_stat

    # 6. Accuracy stratified by MMSE (AD group, linked to the basic-information list via the MRI ID)
    demo = pd.read_csv(oasis2_demo)
    demo["MRI ID"] = demo["MRI ID"].astype(str)
    mmse_map = dict(zip(demo["MRI ID"], demo["MMSE"]))

    ad_idx = np.where(all_y_true == 2)[0]
    ad_rows = []
    for i in ad_idx:
        mid = extract_mri_id(valid_paths[i])
        mmse = mmse_map.get(mid, np.nan)
        correct = float(all_probs[i, 2] > all_probs[i, 0])  # correct if P(AD)>P(CN)
        ad_rows.append({"mri_id": mid, "MMSE": mmse, "correct": correct})
    ad_df = pd.DataFrame(ad_rows).dropna(subset=["MMSE"])
    ad_df["MMSE_bin"] = ad_df["MMSE"].apply(mmse_bin)
    mmse_strat = ad_df.groupby("MMSE_bin", as_index=False).agg(
        n=("correct", "size"), n_correct=("correct", "sum"))
    mmse_strat["accuracy"] = mmse_strat["n_correct"] / mmse_strat["n"]

    # 7. Single-fold metrics
    single_rows = []
    for fi in range(len(models)):
        fp = np.array(fold_probs[fi])
        yp = np.argmax(fp, axis=1)
        s_auc = [roc_auc_score((all_y_true == i).astype(int), fp[:, i]) for i in range(3)]
        single_rows.append(dict(fold=fi + 1, accuracy=accuracy_score(all_y_true, yp),
                                balanced_accuracy=balanced_accuracy_score(all_y_true, yp),
                                f1_macro=f1_score(all_y_true, yp, average="macro", zero_division=0),
                                auc_CN=s_auc[0], auc_MCI=s_auc[1], auc_AD=s_auc[2],
                                auc_macro=float(np.mean(s_auc))))
    single_df = pd.DataFrame(single_rows)

    # 8. Print the report
    print("\n" + "=" * 66)
    print("OASIS-2 external validation report (5-fold ensemble, direct directory scan)")
    print("=" * 66)
    print(f"Samples: {len(test_data)} | " + " | ".join(f"{k}: {v}" for k, v in n_per_class.items()))
    print("\n[1] Three-class overall:")
    print(f"    Accuracy={acc:.4f}  BAcc={bacc:.4f}  F1_Macro={f1_macro:.4f}  AUC_Macro={auc_macro:.4f}")
    print(f"[2] Per-class AUC:  CN={per_class_auc[0]:.4f}  MCI={per_class_auc[1]:.4f}  AD={per_class_auc[2]:.4f}")
    print(f"[3] Confusion matrix (rows = true, columns = predicted CN/MCI/AD):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    True {name:3s} | {cm3[i,0]:4d}  {cm3[i,1]:4d}  {cm3[i,2]:4d}")
    print("\n[4] Main CN vs AD binary result (excluding MCI, n=94):")
    print(f"    AUC={auc_cn_ad:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]  (paper: 0.757 [0.633, 0.873])")
    print(f"    Accuracy={acc_cn_ad:.4f}  (paper: 0.660)")
    print(f"    Mann-Whitney U={u_report:.0f}, one-sided p={p_mw:.3e}  (paper: U=1160, p=1.8e-4)")
    print("\n[5] Accuracy of the AD group stratified by MMSE (P(AD)>P(CN)):")
    for _, r in mmse_strat.iterrows():
        print(f"    MMSE {r['MMSE_bin']:>5s}: {r['accuracy']*100:.0f}% ({int(r['n_correct'])}/{int(r['n'])})")
    print(f"\n[6] Single-fold AUC (CN/MCI/AD) mean±std:")
    for col in ["auc_CN", "auc_MCI", "auc_AD", "auc_macro"]:
        print(f"    {col:10s}: {single_df[col].mean():.4f} ± {single_df[col].std(ddof=1):.4f}")

    # 9. Write to disk
    pd.DataFrame({
        "Metric": ["Accuracy", "Balanced_Accuracy", "F1_Macro", "F1_Weighted", "AUC_Macro_OvR",
                   "AUC_CN", "AUC_MCI", "AUC_AD", "N_Total", "N_CN", "N_MCI", "N_AD"],
        "Value": [acc, bacc, f1_macro, f1_weighted, auc_macro,
                  per_class_auc[0], per_class_auc[1], per_class_auc[2],
                  len(test_data), n_per_class["CN"], n_per_class["MCI"], n_per_class["AD"]]
    }).to_csv(os.path.join(output_dir, "oasis2_ensemble_metrics.csv"), index=False)

    pd.DataFrame({"Class": CLASS_NAMES, "Precision": prec, "Recall": rec, "F1": f1s,
                  "Support": sup.astype(int), "AUC_OvR": per_class_auc}
                 ).to_csv(os.path.join(output_dir, "oasis2_class_metrics.csv"), index=False)

    pd.DataFrame(cm3, index=[f"True_{n}" for n in CLASS_NAMES],
                 columns=[f"Pred_{n}" for n in CLASS_NAMES]).to_csv(
        os.path.join(output_dir, "oasis2_confusion_matrix_3x3.csv"))

    pd.DataFrame({"file_path": valid_paths,
                  "true_label": all_y_true,
                  "true_label_name": [CLASS_NAMES[l] for l in all_y_true],
                  "pred_label": y_pred,
                  "pred_label_name": [CLASS_NAMES[p] for p in y_pred],
                  "prob_CN": all_probs[:, 0], "prob_MCI": all_probs[:, 1], "prob_AD": all_probs[:, 2]
                  }).to_csv(os.path.join(output_dir, "oasis2_detailed_predictions.csv"), index=False)

    pd.DataFrame({
        "Metric": ["AUC_CN_vs_AD", "Accuracy", "Bootstrap_CI_low", "Bootstrap_CI_high",
                   "MannWhitney_U", "MannWhitney_p_one_sided", "N_bootstrap", "Seed"],
        "Value": [auc_cn_ad, acc_cn_ad, ci_lo, ci_hi, u_report, p_mw, N_BOOT, BOOT_SEED]
    }).to_csv(os.path.join(output_dir, "oasis2_binary_cn_vs_ad.csv"), index=False)

    mmse_strat.to_csv(os.path.join(output_dir, "oasis2_mmse_stratified.csv"), index=False)

    single_df.to_csv(os.path.join(output_dir, "oasis2_single_fold_metrics.csv"), index=False)
    stats = single_df.drop(columns=["fold"]).agg(["mean", "std"]).T
    stats.columns = ["mean", "std"]
    stats.to_csv(os.path.join(output_dir, "oasis2_single_fold_mean_std.csv"))

    print(f"\nResults saved to: {output_dir}")

    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
