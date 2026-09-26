# -*- coding: utf-8 -*-
"""Probability calibration metrics: a canonical implementation of ECE (expected calibration error).

This module is the **single source of truth** for the ECE implementation and is shared by the
following scripts:
    evaluate/report_metrics.py        Brier and ECE for Table 2 / Table S5
    evaluate/plot_figures.py          Figure 2 calibration-curve title
    figures/redraw_figures_ensemble.py Figure 4 ROC/PR redraw
    smoke_tests/smoke_offline_recheck.py offline anchor recomputation

[Fix background (2026-09-21)]
The scripts above each previously inlined their own ECE implementation, and they
shared the same defect:

    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy='uniform')
    bin_counts = np.histogram(p, bins=10, range=(0, 1))[0]        # 10 equal-width bin counts
    if len(prob_true) < len(bin_counts):                          # when empty bins exist
        bin_counts = np.histogram(p, bins=len(prob_true), range=(0,1))[0]   # ← wrong
    ece = np.sum(bin_counts / n * np.abs(prob_true - prob_pred))

`sklearn.calibration_curve(strategy='uniform')` **returns only non-empty bins**
(mean predicted probability, positive-class frequency), and their element order corresponds to
the positions of the original 10 bins "after skipping empty bins". The code above, however,
when empty bins exist, **re-bins** the sample counts into len(prob_true) equal-width bins; the
bin edges change accordingly, so that the "per-bin counts" and the "per-bin calibration
deviation" become misaligned and the ECE sum loses its meaning.

Affected values (before fix → after fix):
    per-class ECE  CN 0.0723→0.0741, MCI 0.1061→0.1024, AD 0.0531→0.0557
    macro-average ECE 0.0772→0.0774
See BUG-01 in the authors' code-statistics methodology audit report
(authors' workspace; not distributed with this package).
"""

import numpy as np


def ece_score(y_true, y_prob, n_bins=10):
    """Expected calibration error, ECE (as defined in supplementary S8 of the paper).

    The predicted probabilities are divided into n_bins equal-width bins and summed with
    weighting by the number of samples in each bin:

        ECE = Σ_m (n_m / N) · |acc(B_m) − conf(B_m)|

    where n_m is the number of samples in the m-th bin, acc(B_m) is the true positive-class
    frequency within that bin, and conf(B_m) is the mean predicted probability within that
    bin. Empty bins have n_m = 0 and do not contribute.

    Bin-edge convention: intervals are left-closed and right-open; p = 0 falls into the first
    bin and p = 1 into the last bin.

    Args:
        y_true: true binary labels (0/1), a one-dimensional array-like.
        y_prob: predicted probability of the positive class, a one-dimensional array-like of
            the same length as y_true.
        n_bins: number of bins, default 10 (the paper's definition).

    Returns:
        float: the ECE value, in the range [0, 1]; returns nan when the input is empty.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n = y_true.size
    if n == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-12                      # so that p = 1.0 falls into the last bin rather than out of range
    binids = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)

    total = 0.0
    for b in range(n_bins):
        m = binids == b
        n_b = int(m.sum())
        if n_b == 0:                        # empty bin: n_m = 0, no contribution
            continue
        acc = float(y_true[m].mean())
        conf = float(y_prob[m].mean())
        total += n_b / n * abs(acc - conf)
    return float(total)


def argmax_ece_score(proba, y_true, n_bins=10, strategy="uniform"):
    """Top-label (argmax) ECE.

    The unit of observation is "each sample's maximum predicted probability and its
    corresponding class": the true label is taken as "whether the prediction equals the true
    class", and the confidence as that sample's maximum probability.
    This metric corresponds directly to the single probability value used in clinical risk
    communication.

    Args:
        proba: (N, K) predicted probabilities for each class.
        y_true: (N,) true class labels (integer-encoded).
        n_bins: number of bins, default 10.
        strategy: 'uniform' = equal-width binning (the paper's main definition);
                  'quantile' = equal-frequency (adaptive) binning.

    Returns:
        float: top-label ECE.
    """
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true).ravel()
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(float)
    n = conf.size
    if n == 0:
        return float("nan")

    if strategy == "quantile":
        total = 0.0
        for idx in np.array_split(np.argsort(conf), n_bins):
            if len(idx) == 0:
                continue
            total += len(idx) / n * abs(correct[idx].mean() - conf[idx].mean())
        return float(total)

    return ece_score(correct, conf, n_bins=n_bins)
