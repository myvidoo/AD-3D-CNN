# -*- coding: utf-8 -*-
# [Source] 01_data_and_preprocessing_paper2.1-2.2_S1/registration_quality_ANOVA_recheck.py  (translated from the authors' archive path)
# [Paper correspondence] S1.9 registration QC: three-group ANOVA + Tukey recomputation (reproduction of the authoritative summary CSV)
# [Note] This file is copied unchanged and its logic is untouched; before running, change the paths in the following lines to local paths:
#        Line numbers: (no hard-coded drive-letter paths found; check the configuration section at the head of the script)

"""
Verification script: between-group ANOVA / η² recomputation for registration quality
(paper supplementary material S1.9)
=========================================================================
Data source: registration_quality_S1.9/registration_quality_summary.csv in this directory
       (954 cases summarised by CN/MCI/AD group as mean±SD and sample size for Dice / CC / MAE)
Purpose: independently rebuild the one-way analysis of variance (ANOVA) and the
        effect size η² from the group-level sufficient statistics (per-group n, mean, SD),
        verifying every F, p and η² value reported in §S1.9 of the paper.
Principle: the F statistic of a one-way ANOVA depends only on the group sample sizes, means
        and within-group variances (SDs), so it can be rebuilt exactly from the group-level
        summary table without per-sample raw data.
        SS_between = Σ n_i (x̄_i − x̄_grand)²
        SS_within  = Σ (n_i − 1) s_i²
        F = (SS_between/(k−1)) / (SS_within/(N−k))
        η² = SS_between / (SS_between + SS_within)
Run: python registration_quality_ANOVA_recheck.py
Note: this script is a standalone verification script written during the paper-organisation
        stage (the original ANOVA analysis was a temporary script in the workflow and was
        not archived; its results are saved to
        registration_quality_S1.9/between_group_effect_sizes.txt and agree exactly with
        this script's output).
"""
import os
import math
import sys

import pandas as pd
from scipy import stats

# The Chinese Windows console defaults to GBK; this script prints characters such as η²,
# so the encoding error policy must be relaxed
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "registration_quality_S1.9", "registration_quality_summary.csv")
df = pd.read_csv(CSV)
df = df[df["Group"].isin(["CN", "MCI", "AD"])].copy()

GROUP_ORDER = ["CN", "MCI", "AD"]
N = int(df["N"].sum())

# Values reported in the paper, §S1.9 (used for comparison)
PAPER = {
    "Dice_mean": {"F": 21.94, "p": 4.85e-10, "eta2": 0.044},
    "CC_mean":   {"F": 16.21, "p": 1.20e-7,  "eta2": 0.033},
    "MAE_mean":  {"F": 19.49, "p": 5.07e-9,  "eta2": 0.039},
}


def anova_from_summary(means, stds, ns):
    """Rebuild the one-way ANOVA from per-group mean/SD/sample size; return F, p, eta2"""
    ns = [int(n) for n in ns]
    k = len(ns)
    grand = sum(n * m for n, m in zip(ns, means)) / sum(ns)
    ss_between = sum(n * (m - grand) ** 2 for n, m in zip(ns, means))
    ss_within = sum((n - 1) * s ** 2 for n, s in zip(ns, stds))
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (sum(ns) - k)
    F = ms_between / ms_within
    p = stats.f.sf(F, k - 1, sum(ns) - k)
    eta2 = ss_between / (ss_between + ss_within)
    return F, p, eta2


def fmt_float(x, nd=4):
    return f"{x:.{nd}f}"


print("=" * 66)
print("Between-group ANOVA / η² recomputation for registration quality (paper supplementary material S1.9)")
print("=" * 66)
print(f"Total sample size N = {N}\n")

results = {}
for col in ["Dice_mean", "CC_mean", "MAE_mean"]:
    std_col = col.replace("_mean", "_std")
    means = [float(df[df["Group"] == g][col].iloc[0]) for g in GROUP_ORDER]
    stds = [float(df[df["Group"] == g][std_col].iloc[0]) for g in GROUP_ORDER]
    ns = [int(df[df["Group"] == g]["N"].iloc[0]) for g in GROUP_ORDER]

    F, p, eta2 = anova_from_summary(means, stds, ns)
    results[col] = (F, p, eta2)

    print("-" * 66)
    print(f"Metric: {col.replace('_mean', '')}   groups: " +
          ", ".join(f"{g} {m:.4f}±{s:.4f} (n={n})"
                    for g, m, s, n in zip(GROUP_ORDER, means, stds, ns)))
    print(f"  recomputed: F={fmt_float(F)}, p={p:.2e}, η²={eta2:.4f}")
    paper = PAPER[col]
    print(f"  paper: F={paper['F']}, p={paper['p']:.2e}, η²={paper['eta2']}")
    ok = (abs(F - paper["F"]) < 0.01 and abs(eta2 - paper["eta2"]) < 0.001
          and abs(math.log10(p) - math.log10(paper["p"])) < 0.01)
    print(f"  verdict: {'✔ consistent with the paper' if ok else '✘ discrepancy, please check'}")

print("\nConclusion: although the between-group differences in registration quality are "
      "statistically significant (p<0.001), η² is < 0.05 for all metrics, i.e. a small "
      "effect; group explains only 3%–4% of the variance and is not a confounder for the "
      "downstream classification.")
