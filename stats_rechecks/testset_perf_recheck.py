# -*- coding: utf-8 -*-
"""
Verification script 2: ensemble test-set performance recomputation (paper §3.2, Table 2, Table S5, Figures 2–4, Figure S4)
=========================================================================
Data source: ../04_five-fold_cross-validation_and_ensemble_evaluation_paper_2.4-3.2_S3-S5/result_data/ensemble_test_predictions.csv
       (per-sample prediction probabilities of the equally weighted 5-fold logit ensemble on the fixed test set of 144 samples)
Purpose: independently recompute all metrics of the paper's Table 2 and Table S5 from the per-sample predictions, verifying traceability.
Run    : python verify_script2_testset_performance_recompute.py > verification_result_testset_performance.txt
Note   : this script is a verification script newly written during the paper-preparation stage (not the original analysis code);
        the original metrics were computed and written to disk by run_generate_ensemble_predictions.py and
        run_5fold_bootstrap_ci_with_class_ci.py and _calc_*.py.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "ensemble_test_predictions.csv")
df = pd.read_csv(CSV)
assert len(df) == 144, "expected 144 samples, got %d" % len(df)

y_true = df["true_label"].values
y_pred = df["pred_label"].values
proba = df[["prob_cn", "prob_mci", "prob_ad"]].values
classes = ["CN", "MCI", "AD"]
conf = proba.max(axis=1)
correct = (y_true == y_pred)

print("=" * 60)
print("Verification 1: overall performance (macro-average rows of Table 2 / Table S5 of the paper)")
print("=" * 60)
acc = (y_true == y_pred).mean()
print("Accuracy: %.4f (%d/144) (paper: 0.8681, 125/144)" % (acc, correct.sum()))
print("Macro-average F1: %.4f (paper: 0.8675)" % f1_score(y_true, y_pred, average="macro"))
macro_auc = np.mean([roc_auc_score(y_true == i, proba[:, i]) for i in range(3)])
print("Macro-average AUC: %.4f (paper: 0.9630)" % macro_auc)

print()
print("=" * 60)
print("Verification 2: per-class metrics (Table 2 of the paper)")
print("=" * 60)
print("%-4s %-9s %-9s %-9s %-7s %-7s %-7s" %
      ("Class", "Precision", "Recall", "Specificity", "F1", "AUC", "Support"))
for i, c in enumerate(classes):
    prec = precision_score(y_true == i, y_pred == i, zero_division=0)
    rec = recall_score(y_true == i, y_pred == i)
    spec = recall_score(y_true != i, y_pred != i)
    f1c = f1_score(y_true == i, y_pred == i)
    auc_c = roc_auc_score(y_true == i, proba[:, i])
    print("%-4s %-9.4f %-9.4f %-9.4f %-7.4f %-7.4f %-7d" %
          (c, prec, rec, spec, f1c, auc_c, (y_true == i).sum()))

print()
print("=" * 60)
print("Verification 3: confusion matrix (paper Figure 3: CN→AD 2 cases, AD→CN 0 cases, MCI→CN 5 cases, MCI→AD 4 cases)")
print("=" * 60)
cm = confusion_matrix(y_true, y_pred)
print(cm)

print()
print("=" * 60)
print("Verification 4: probability calibration (Brier/ECE columns of Table 2 / §3.2 of the paper)")
print("=" * 60)
eces = []
for i, c in enumerate(classes):
    b = brier_score_loss(y_true == i, proba[:, i])
    # ECE: 10 bins
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (proba[:, i] > lo) & (proba[:, i] <= hi)
        if m.sum() > 0:
            ece += m.sum() / len(y_true) * abs(proba[m, i].mean() - (y_true[m] == i).mean())
    eces.append(ece)
    print("%s: Brier=%.4f, ECE=%.4f (paper: CN 0.0742/0.0741, MCI 0.0966/0.1024, AD 0.0534/0.0557)"
          % (c, b, ece))
print("Macro-average Brier=%.4f, ECE=%.4f (paper: 0.0748 / 0.0774)"
      % (np.mean([brier_score_loss(y_true == i, proba[:, i]) for i in range(3)]),
         np.mean(eces)))

print()
print("=" * 60)
print("Verification 5: confidence statistics (Table S5 / §3.2 / §4.2 of the paper)")
print("=" * 60)
print("Mean confidence: %.4f (paper: 0.9717)" % conf.mean())
print("Correct-prediction confidence: %.4f (paper: 0.9838)" % conf[correct].mean())
print("Incorrect-prediction confidence: %.4f (paper: 0.8923)" % conf[~correct].mean())
print("Confidence gap: %.4f (paper: 0.0914)" % (conf[correct].mean() - conf[~correct].mean()))

print()
print("=" * 60)
print("Verification 6: per-class optimal Youden thresholds (Table S5 of the paper)")
print("=" * 60)
for i, c in enumerate(classes):
    fpr, tpr, thr = roc_curve(y_true == i, proba[:, i])
    j = np.argmax(tpr - fpr)
    print("%s: threshold=%.3f, sensitivity=%.3f, specificity=%.3f, J=%.3f"
          " (paper: CN 0.983/0.854/0.990/0.844; MCI 0.052/0.938/0.917/0.854; AD 0.287/0.958/0.938/0.896)"
          % (c, thr[j], tpr[j], 1 - fpr[j], tpr[j] - fpr[j]))
