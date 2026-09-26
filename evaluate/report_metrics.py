# -*- coding: utf-8 -*-
"""
Offline recomputation: all performance metrics of Table 2 + confusion matrix + Brier/ECE + Youden thresholds + confidence statistics.

Relies only on the already-released per-sample prediction probabilities (data/ensemble_test_predictions.csv); no imaging data required.

Corresponds to the paper: Table 2, Table S5, Figure 3, §3.2 (probability calibration), §4.2 (confidence).

Usage:
    python evaluate/report_metrics.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config, resolve_path
from common.calibration_metrics import ece_score

CLASSES = ["CN", "MCI", "AD"]


def compute_all(df):
    y_true = df["true_label"].values
    y_pred = df["pred_label"].values
    proba = df[["prob_cn", "prob_mci", "prob_ad"]].values
    conf = proba.max(axis=1)
    correct = (y_true == y_pred)

    results = {}

    # overall
    results["accuracy"] = float(correct.mean())
    results["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    results["macro_auc"] = float(np.mean([roc_auc_score(y_true == i, proba[:, i]) for i in range(3)]))

    # per class
    per_class = {}
    for i, c in enumerate(CLASSES):
        per_class[c] = {
            "precision": float(precision_score(y_true == i, y_pred == i, zero_division=0)),
            "recall": float(recall_score(y_true == i, y_pred == i)),
            "specificity": float(recall_score(y_true != i, y_pred != i)),
            "f1": float(f1_score(y_true == i, y_pred == i)),
            "auc": float(roc_auc_score(y_true == i, proba[:, i])),
            "support": int((y_true == i).sum()),
        }
    results["per_class"] = per_class

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results["confusion_matrix"] = cm.tolist()

    # Brier / ECE (one-vs-rest macro-average; for the ECE definition see common/calibration_metrics.py)
    eces = []
    briers = []
    for i, c in enumerate(CLASSES):
        briers.append(float(brier_score_loss(y_true == i, proba[:, i])))
        eces.append(ece_score(y_true == i, proba[:, i], n_bins=10))
    results["brier_per_class"] = {c: briers[i] for i, c in enumerate(CLASSES)}
    results["ece_per_class"] = {c: eces[i] for i, c in enumerate(CLASSES)}
    results["macro_brier"] = float(np.mean(briers))
    results["macro_ece"] = float(np.mean(eces))

    # confidence
    results["mean_confidence"] = float(conf.mean())
    results["correct_confidence"] = float(conf[correct].mean())
    results["incorrect_confidence"] = float(conf[~correct].mean())
    results["confidence_gap"] = float(conf[correct].mean() - conf[~correct].mean())

    # Youden thresholds
    youden = {}
    for i, c in enumerate(CLASSES):
        fpr, tpr, thr = roc_curve(y_true == i, proba[:, i])
        j = int(np.argmax(tpr - fpr))
        youden[c] = {
            "threshold": float(thr[j]),
            "sensitivity": float(tpr[j]),
            "specificity": float(1 - fpr[j]),
            "youden_j": float(tpr[j] - fpr[j]),
        }
    results["youden"] = youden

    return results


def main():
    cfg = load_config()
    pred_csv = resolve_path(cfg, "data.pred_csv")
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(pred_csv)
    assert len(df) == 144, f"expected 144 samples, got {len(df)}"

    r = compute_all(df)

    # print
    print("=" * 64)
    print("Table 2 overall performance")
    print("=" * 64)
    print(f"Accuracy: {r['accuracy']:.4f}  (paper: 0.8681, 125/144)")
    print(f"Macro-average F1: {r['macro_f1']:.4f}  (paper: 0.8675)")
    print(f"Macro-average AUC: {r['macro_auc']:.4f}  (paper: 0.9630)")

    print("\nPer-class metrics (Table 2 of the paper):")
    print(f"{'Class':<5}{'Precision':>9}{'Recall':>9}{'Specificity':>9}{'F1':>8}{'AUC':>8}{'Support':>6}")
    for c in CLASSES:
        p = r["per_class"][c]
        print(f"{c:<5}{p['precision']:>9.4f}{p['recall']:>9.4f}{p['specificity']:>9.4f}"
              f"{p['f1']:>8.4f}{p['auc']:>8.4f}{p['support']:>6d}")

    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(np.array(r["confusion_matrix"]))

    print("\nProbability calibration (Brier/ECE, Table 2 of the paper):")
    for c in CLASSES:
        print(f"{c}: Brier={r['brier_per_class'][c]:.4f}, ECE={r['ece_per_class'][c]:.4f}")
    print(f"Macro-average Brier={r['macro_brier']:.4f}, ECE={r['macro_ece']:.4f}  (paper: 0.0748 / 0.0774)")

    print("\nConfidence statistics (Table S5 of the paper):")
    print(f"Mean confidence: {r['mean_confidence']:.4f}  (paper: 0.9717)")
    print(f"Correct-prediction confidence: {r['correct_confidence']:.4f}  (paper: 0.9838)")
    print(f"Incorrect-prediction confidence: {r['incorrect_confidence']:.4f}  (paper: 0.8923)")
    print(f"Confidence gap: {r['confidence_gap']:.4f}  (paper: 0.0914)")

    print("\nPer-class Youden thresholds (Table S5 of the paper):")
    for c in CLASSES:
        y = r["youden"][c]
        print(f"{c}: threshold={y['threshold']:.3f}, sensitivity={y['sensitivity']:.3f}, "
              f"specificity={y['specificity']:.3f}, J={y['youden_j']:.3f}")

    # save JSON
    out_json = os.path.join(output_dir, "report_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_json}")


if __name__ == "__main__":
    main()
