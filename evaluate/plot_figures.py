# -*- coding: utf-8 -*-
"""
Offline plotting: Figure 2 calibration curves, Figure 3 confusion matrix, Figure S4 confidence analysis.

Relies only on data/ensemble_test_predictions.csv (per-sample prediction probabilities for 144 samples); no imaging data required.

Corresponds to the paper: Figure 2 (calibration curves), Figure 3 (confusion matrix), Figure S4 (confidence analysis).
Figure 4 (ROC+PR) is produced by figures/redraw_figures_ensemble.py.

Output file names follow the paper's figure numbering (Figure_2_*, Figure_3_*, Figure_S4_*).

Usage:
    python evaluate/plot_figures.py [--all | --calibration | --confusion | --confidence]
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config, resolve_path
from common.calibration_metrics import ece_score

DPI = 600
CLASS_NAMES = ['CN', 'MCI', 'AD']
CLASS_COLORS = {'CN': '#1F77B4', 'MCI': '#FF7F0E', 'AD': '#D62728'}
MODEL_LABEL = "5-Fold Ensemble"


def load_data(pred_csv):
    df = pd.read_csv(pred_csv)
    y_true = df['true_label'].values
    y_pred = df['pred_label'].values
    y_proba = df[['prob_cn', 'prob_mci', 'prob_ad']].values
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    return df, y_true, y_pred, y_proba, y_bin


def plot_calibration(y_bin, y_proba, output_dir):
    """Figure 2 calibration curves (fixed version: set_aspect('equal') removed, headroom left above the histograms)"""
    n = len(y_bin)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))
    fig.suptitle(f'Calibration Curves (Reliability Diagrams, {MODEL_LABEL})',
                 fontsize=11, fontweight='bold', y=0.99)
    for i, cls in enumerate(CLASS_NAMES):
        ax = axes[i]
        color = CLASS_COLORS[cls]
        prob_true, prob_pred = calibration_curve(y_bin[:, i], y_proba[:, i], n_bins=10, strategy='uniform')
        ece = ece_score(y_bin[:, i], y_proba[:, i], n_bins=10)
        bs = brier_score_loss(y_bin[:, i], y_proba[:, i])
        ax.plot(prob_pred, prob_true, 'o-', color=color, linewidth=1.5, markersize=5, label='Model')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Perfect')
        ax2 = ax.twinx()
        hist, edges = np.histogram(y_proba[:, i], bins=10, range=(0, 1))
        # 20% headroom at the top, so that the tallest bar does not touch the ceiling and mislead
        ax2.set_ylim([0, hist.max() * 1.2 if hist.max() > 0 else 1])
        ax2.bar((edges[:-1] + edges[1:]) / 2, hist, width=0.08, alpha=0.2,
                color='gray', edgecolor='gray', linewidth=0.3)
        ax2.set_ylabel('Count', fontsize=7, color='gray')
        ax2.tick_params(labelsize=6, colors='gray')
        ax.set_xlabel('Predicted Probability', fontsize=8)
        ax.set_ylabel('Fraction of Positives', fontsize=8)
        ax.set_title(f'{cls} (ECE={ece:.3f}, BS={bs:.3f})', fontsize=9, fontweight='bold')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.legend(fontsize=7, loc='upper left')
        ax.tick_params(labelsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(output_dir, 'Figure_2_calibration_curves.png')
    plt.savefig(p, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"saved: {p}")


def plot_confidence(y_true, y_pred, y_proba, output_dir):
    """Figure S4 confidence analysis (2x2 subplots)"""
    max_proba = np.max(y_proba, axis=1)
    correct_mask = (y_true == y_pred)
    n_correct = int(correct_mask.sum())
    n_wrong = len(y_true) - n_correct
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.0))

    # (a) confidence histogram
    bins = np.linspace(0, 1, 21)
    axes[0, 0].hist(max_proba[correct_mask], bins=bins, alpha=0.7,
                    label=f'Correct (n={n_correct})', color='green', edgecolor='black', linewidth=0.3)
    axes[0, 0].hist(max_proba[~correct_mask], bins=bins, alpha=0.7,
                    label=f'Incorrect (n={n_wrong})', color='red', edgecolor='black', linewidth=0.3)
    axes[0, 0].axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[0, 0].set_xlabel('Prediction Confidence (Max Probability)', fontsize=8)
    axes[0, 0].set_ylabel('Number of Samples', fontsize=8)
    axes[0, 0].set_title('(a) Confidence Distribution', fontsize=9, fontweight='bold')
    axes[0, 0].legend(fontsize=7, frameon=True)
    axes[0, 0].grid(True, alpha=0.2, axis='y')

    # (b) box plot per class
    box_data, box_labels = [], []
    for i, cls in enumerate(CLASS_NAMES):
        box_data.append(max_proba[y_true == i])
        box_labels.append(f'{cls}\n(n={int((y_true == i).sum())})')
    bp = axes[0, 1].boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                            showfliers=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='white', markersize=4))
    for patch, cls in zip(bp['boxes'], CLASS_NAMES):
        patch.set_facecolor(CLASS_COLORS[cls]); patch.set_alpha(0.6)
    axes[0, 1].set_ylabel('Prediction Confidence', fontsize=8)
    axes[0, 1].set_title('(b) Confidence by True Class', fontsize=9, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.2, axis='y')

    # (c) violin plot: correct vs incorrect
    data_for_violin, positions, colors_violin, labels_violin = [], [], [], []
    for i, cls in enumerate(CLASS_NAMES):
        cls_mask = (y_true == i)
        c_ok, c_err = cls_mask & correct_mask, cls_mask & ~correct_mask
        pos = i * 2 + 1
        if c_ok.any():
            data_for_violin.append(max_proba[c_ok]); positions.append(pos)
            colors_violin.append('lightgreen'); labels_violin.append(f'{cls}\nCorrect')
        if c_err.any():
            data_for_violin.append(max_proba[c_err]); positions.append(pos + 0.5)
            colors_violin.append('lightcoral'); labels_violin.append(f'{cls}\nError')
    vp = axes[1, 0].violinplot(data_for_violin, positions=positions, showmeans=True, showmedians=True)
    for body, color in zip(vp['bodies'], colors_violin):
        body.set_facecolor(color); body.set_alpha(0.7)
    axes[1, 0].set_xticks(positions)
    axes[1, 0].set_xticklabels(labels_violin, fontsize=6.5)
    axes[1, 0].set_ylabel('Prediction Confidence', fontsize=8)
    axes[1, 0].set_title('(c) Confidence: Correct vs Error by Class', fontsize=9, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.2, axis='y')
    axes[1, 0].set_ylim([0, 1.05])

    # (d) reliability diagram
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    accuracies, counts = [], []
    for j in range(len(bin_edges) - 1):
        mask = (max_proba >= bin_edges[j]) & (max_proba < bin_edges[j + 1])
        counts.append(int(mask.sum()))
        accuracies.append(correct_mask[mask].mean() if mask.any() else np.nan)
    axes[1, 1].bar(bin_centers, counts, width=0.08, alpha=0.3, color='gray', label='Count')
    ax2 = axes[1, 1].twinx()
    ax2.plot(bin_centers, accuracies, 'b-o', linewidth=1.5, markersize=5, label='Accuracy')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Perfect Calibration')
    ax2.set_ylim([0, 1.05])
    axes[1, 1].set_xlabel('Confidence Bin', fontsize=8)
    axes[1, 1].set_ylabel('Number of Samples', fontsize=8, color='gray')
    ax2.set_ylabel('Accuracy', fontsize=8, color='blue')
    axes[1, 1].set_title('(d) Reliability Diagram (All Classes)', fontsize=9, fontweight='bold')
    h1, l1 = axes[1, 1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left')

    plt.tight_layout()
    p = os.path.join(output_dir, 'Figure_S4_confidence.png')
    plt.savefig(p, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"saved: {p}")


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Figure 3 confusion matrix: three panels (a) Counts / (b) Row-Normalized (%) / (c) Summary"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent_row = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], cbar_kws={'label': 'Count', 'shrink': 0.7},
                annot_kws={'fontsize': 9}, vmin=0, vmax=48)
    axes[0].set_title('(a) Counts', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=9)
    axes[0].set_xlabel('Predicted Label', fontsize=9)
    axes[0].tick_params(labelsize=8)

    sns.heatmap(cm_percent_row, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], cbar_kws={'label': '% (Row)', 'shrink': 0.7},
                annot_kws={'fontsize': 9}, vmin=0, vmax=100)
    axes[1].set_title('(b) Row-Normalized (%)', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=9)
    axes[1].set_xlabel('Predicted Label', fontsize=9)
    axes[1].tick_params(labelsize=8)

    summary_text = ""
    for i, cls in enumerate(CLASS_NAMES):
        correct = cm[i, i]
        total = cm[i, :].sum()
        summary_text += f"{cls}: {correct}/{total} ({cm_percent_row[i, i]:.1f}%)\n"
    overall_acc = np.trace(cm) / cm.sum() * 100
    summary_text += f"\nOverall: {np.trace(cm)}/{cm.sum()} ({overall_acc:.1f}%)"

    axes[2].text(0.5, 0.5, summary_text, transform=axes[2].transAxes,
                 fontsize=10, verticalalignment='center', horizontalalignment='center',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[2].set_title('(c) Summary', fontsize=10, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    p = os.path.join(output_dir, 'Figure_3_confusion_matrices.png')
    plt.savefig(p, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"saved: {p}")
    print(f"Overall: {np.trace(cm)}/{cm.sum()} = {overall_acc:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Plot calibration / confidence / confusion-matrix figures")
    parser.add_argument("--all", action="store_true", help="generate all three figures")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--confidence", action="store_true")
    parser.add_argument("--confusion", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    pred_csv = resolve_path(cfg, "data.pred_csv")
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    _, y_true, y_pred, y_proba, y_bin = load_data(pred_csv)

    do_all = args.all or not (args.calibration or args.confidence or args.confusion)
    if do_all or args.calibration:
        plot_calibration(y_bin, y_proba, output_dir)
    if do_all or args.confidence:
        plot_confidence(y_true, y_pred, y_proba, output_dir)
    if do_all or args.confusion:
        plot_confusion_matrix(y_true, y_pred, output_dir)


if __name__ == "__main__":
    main()
