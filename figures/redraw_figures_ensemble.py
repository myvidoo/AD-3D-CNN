# -*- coding: utf-8 -*-
"""
Redrawn from ensemble_test_predictions.csv:
- Figure 4 ROC curves + PR curves (left/right)  ->  Figure_4_roc_pr.png

Figure 2 (calibration) and Figure S4 (confidence) are produced by evaluate/plot_figures.py.

Outputs 600 DPI PNGs, replacing the old figures in the docx (based on the RUN128 single fold).
All titles carry the "5-Fold Ensemble" label.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

# All paths are resolved from config.yaml (reproducers only need to edit config.yaml)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config, resolve_path
from common.calibration_metrics import ece_score

_cfg = load_config()
INPUT_CSV = resolve_path(_cfg, "data.pred_csv")
OUTPUT_DIR = _cfg["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

DPI = 600
CLASS_NAMES = ['CN', 'MCI', 'AD']
CLASS_COLORS = {'CN': '#1F77B4', 'MCI': '#FF7F0E', 'AD': '#D62728'}  # blue, orange, red (keeping the style of the original figure)
MODEL_LABEL = "5-Fold Ensemble"

# ===== load data =====
df = pd.read_csv(INPUT_CSV)
y_true = df['true_label'].values
y_pred = df['pred_label'].values
y_proba = df[['prob_cn', 'prob_mci', 'prob_ad']].values
y_bin = label_binarize(y_true, classes=[0, 1, 2])
n = len(y_true)
n_correct = int((y_pred == y_true).sum())
n_wrong = n - n_correct
print(f"n={n}, correct={n_correct}, wrong={n_wrong}, ACC={n_correct/n:.4f}")

# ================================================================
# Figure 2 calibration curves (Reliability Diagrams, 1x3)
# ================================================================
def plot_roc_pr():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    # left: ROC
    ax = axes[0]
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ls = ['-', '--', '-.'][i]
        ax.plot(fpr, tpr, color=CLASS_COLORS[cls], linestyle=ls, linewidth=1.2,
                label=f'{cls} (AUC={roc_auc:.3f})')
    # micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    ax.plot(fpr_micro, tpr_micro, color='navy', linestyle=':', linewidth=1.5,
            label=f'Micro-avg (AUC={auc(fpr_micro, tpr_micro):.3f})')
    # macro-average
    all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:, i], y_proba[:, i])[0] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= 3
    # Legend AUC = arithmetic mean of the three per-class AUCs (paper macro-AUC, 0.9630),
    # not the integral of the averaged curve (0.9683) -- consistent with Table 2. (fixed 2026-09-29)
    macro_auc_mean = float(np.mean([auc(*roc_curve(y_bin[:, i], y_proba[:, i])[:2]) for i in range(3)]))
    ax.plot(all_fpr, mean_tpr, color='darkgreen', linestyle='-.', linewidth=1.2,
            label=f'Macro-avg (AUC={macro_auc_mean:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
    # optimal operating point (Youden index)
    for i, cls in enumerate(CLASS_NAMES):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        idx = int(np.argmax(tpr_i - fpr_i))
        ax.scatter(fpr_i[idx], tpr_i[idx], color=CLASS_COLORS[cls], s=30, zorder=5,
                   marker='o', edgecolors='black', linewidth=0.3)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=9)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=9)
    ax.set_title('ROC Curves with Optimal Operating Points', fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', fontsize=7, frameon=True)
    ax.grid(True, alpha=0.2, linewidth=0.5); ax.tick_params(labelsize=7)
    ax.set_aspect('equal')

    # right: PR
    ax = axes[1]
    for i, cls in enumerate(CLASS_NAMES):
        p, r, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        ax.plot(r, p, color=CLASS_COLORS[cls], linewidth=1.2, label=f'{cls} (AP={ap:.3f})')
        f1s = 2 * p * r / (p + r + 1e-10)
        idx = int(np.argmax(f1s))
        ax.scatter(r[idx], p[idx], color=CLASS_COLORS[cls], s=30, zorder=5,
                   marker='s', edgecolors='black', linewidth=0.3)
    # baseline (positive-sample proportion)
    for i, cls in enumerate(CLASS_NAMES):
        ratio = y_bin[:, i].mean()
        ax.axhline(ratio, color=CLASS_COLORS[cls], linestyle=':', linewidth=0.6, alpha=0.5)
    # iso-F1
    for f in [0.2, 0.4, 0.6]:
        xs = np.linspace(0.01, 1, 100)
        ys = f * xs / (2 * xs - f)
        valid = (ys >= 0) & (ys <= 1)
        ax.plot(xs[valid], ys[valid], color='gray', alpha=0.2, linewidth=0.5)
        ax.annotate(f'F1={f:.1f}', xy=(0.95, f*0.95/(2*0.95-f)), fontsize=6, alpha=0.4, ha='right')
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=9)
    ax.set_ylabel('Precision (PPV)', fontsize=9)
    ax.set_title('Precision-Recall Curves with Iso-F1 Lines', fontsize=10, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, frameon=True)
    ax.grid(True, alpha=0.2, linewidth=0.5); ax.tick_params(labelsize=7)
    ax.set_aspect('equal')

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'Figure_4_roc_pr.png')
    plt.savefig(p, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"saved: {p}")

# ================================================================
# Figure S4 confidence analysis (2x2 subplots)
# ================================================================
if __name__ == "__main__":
    plot_roc_pr()
    print("Figure 4 generated.")