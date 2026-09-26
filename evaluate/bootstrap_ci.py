# -*- coding: utf-8 -*-
"""
Stratified bootstrap 95% confidence intervals (offline version).

Relies only on data/ensemble_test_predictions.csv (per-sample prediction probabilities for 144 samples); no imaging data or weights required.

Note: the original implementation bootstrapped the logits of the 5-fold models (per-sample softmax(mean(logits)) is exactly equivalent to the prob in
      ensemble_test_predictions.csv); this script performs the same stratified bootstrap directly on the released probabilities,
      which is mathematically equivalent.

Corresponds to the paper: the confidence-interval column of Table 2 (macro-average AUC [0.9357–0.9845], F1, Acc; per-class OVR AUC CI).

Caution: bootstrap is stochastic and the exact interval depends on the random seed; this script fixes random_state to guarantee reproducibility,
      and the reproduced values agree with the paper within bootstrap error.

Usage:
    python evaluate/bootstrap_ci.py [--n_bootstrap 2000] [--seed 42]
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config, resolve_path

CLASS_NAMES = ['CN', 'MCI', 'AD']
NUM_CLASSES = 3


def main():
    parser = argparse.ArgumentParser(description="Stratified bootstrap 95% CI")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    pred_csv = resolve_path(cfg, "data.pred_csv")
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(args.seed)

    df = pd.read_csv(pred_csv)
    y_true = df['true_label'].values
    y_proba = df[['prob_cn', 'prob_mci', 'prob_ad']].values

    class_indices = {i: np.where(y_true == i)[0] for i in range(NUM_CLASSES)}

    auc_scores, f1_scores, acc_scores = [], [], []
    class_auc_scores = {i: [] for i in range(NUM_CLASSES)}

    for _ in tqdm(range(args.n_bootstrap), desc="Stratified Bootstrap"):
        boot_indices = []
        for i in range(NUM_CLASSES):
            boot_indices.extend(np.random.choice(
                class_indices[i], size=len(class_indices[i]), replace=True))
        np.random.shuffle(boot_indices)

        boot_gt = y_true[boot_indices]
        boot_probs = y_proba[boot_indices]
        boot_preds = np.argmax(boot_probs, axis=1)

        auc_scores.append(roc_auc_score(boot_gt, boot_probs, multi_class='ovr'))
        f1_scores.append(f1_score(boot_gt, boot_preds, average='macro'))
        acc_scores.append(accuracy_score(boot_gt, boot_preds))

        for i in range(NUM_CLASSES):
            y_bin = (boot_gt == i).astype(int)
            if len(np.unique(y_bin)) > 1:
                class_auc_scores[i].append(roc_auc_score(y_bin, boot_probs[:, i]))
            else:
                class_auc_scores[i].append(0.5)

    def mean_ci(scores):
        return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

    print(f"\nStratified bootstrap 95% confidence intervals (N={args.n_bootstrap}, seed={args.seed})")
    print("=" * 60)
    m_auc, l_auc, h_auc = mean_ci(auc_scores)
    m_f1, l_f1, h_f1 = mean_ci(f1_scores)
    m_acc, l_acc, h_acc = mean_ci(acc_scores)
    print(f"Macro-average AUC: {m_auc:.4f}  (95% CI: {l_auc:.4f} – {h_auc:.4f})  paper [0.9357–0.9845]")
    print(f"Macro-average F1 : {m_f1:.4f}  (95% CI: {l_f1:.4f} – {h_f1:.4f})  paper [0.8122–0.9176]")
    print(f"Accuracy         : {m_acc:.4f}  (95% CI: {l_acc:.4f} – {h_acc:.4f})  paper [0.8125–0.9168]")

    print("\nPer-class OVR AUC 95% CI:")
    class_results = {}
    for i in range(NUM_CLASSES):
        m, l, h = mean_ci(class_auc_scores[i])
        class_results[CLASS_NAMES[i]] = {'mean': m, 'ci_low': l, 'ci_high': h}
        print(f"  {CLASS_NAMES[i]}: {m:.4f}  (95% CI: {l:.4f} – {h:.4f})")

    overall_df = pd.DataFrame({
        'Metric': ['Macro_AUC', 'Macro_F1', 'Accuracy'],
        'Mean': [m_auc, m_f1, m_acc],
        'CI_Low': [l_auc, l_f1, l_acc],
        'CI_High': [h_auc, h_f1, h_acc]
    })
    overall_df.to_csv(os.path.join(output_dir, 'stratified_bootstrap_ci_results.csv'), index=False)

    class_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'AUC_Mean': [class_results[c]['mean'] for c in CLASS_NAMES],
        'AUC_CI_Low': [class_results[c]['ci_low'] for c in CLASS_NAMES],
        'AUC_CI_High': [class_results[c]['ci_high'] for c in CLASS_NAMES]
    })
    class_df.to_csv(os.path.join(output_dir, 'class_wise_auc_bootstrap_ci.csv'), index=False)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
