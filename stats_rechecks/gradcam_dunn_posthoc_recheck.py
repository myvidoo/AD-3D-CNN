# -*- coding: utf-8 -*-
"""
v41 revision auxiliary statistics: the MTL mass fractions of the three groups (CN/MCI/AD, whole test set grouped by true label)
Kruskal-Wallis post-hoc pairwise comparisons — Dunn's test (with ties correction) + Bonferroni correction.

Data source: 05_explainability_GradCAM_..._S6/result_data/per_sample_results_final.csv
Definition: mtl_mass_frac (same source as the main-text KW; KW over the whole test set, n=144, n=48 per group)
        [v45 correction] The ROI masks have been corrected according to the Harvard-Oxford atlas rule "image value = XML index + 1",
        and this script reads the corrected per_sample_results_final.csv.
Recompute: also recompute KW H/p and the group means±SD, cross-checked against statistical_results.json.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference_results", "gradcam_v3", "per_sample_results_final.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradcam_dunn_posthoc_results.json")

df = pd.read_csv(CSV)
CLASSES = ["CN", "MCI", "AD"]
groups = {c: df.loc[df["true_name"] == c, "mtl_mass_frac"].dropna().values for c in CLASSES}
# SD uses ddof=1, consistent with statistical_results.json (pandas .std()) and the main-text definition
for c in CLASSES:
    print(f"{c}: n={len(groups[c])}  mean={groups[c].mean()*100:.3f}%  sd={np.std(groups[c], ddof=1)*100:.3f}%")

# ---- KW recomputation ----
H, p_kw = stats.kruskal(*[groups[c] for c in CLASSES])
print(f"\nKruskal-Wallis: H={H:.4f}, p={p_kw:.6g}  (n_total={sum(len(groups[c]) for c in CLASSES)})")

# ---- Dunn's test (Bonferroni) ----
data = np.concatenate([groups[c] for c in CLASSES])
labels = np.concatenate([[c] * len(groups[c]) for c in CLASSES])
N = len(data)
ranks = stats.rankdata(data)  # average ranks, ties handled
grp_idx = {c: np.where(labels == c)[0] for c in CLASSES}
R_bar = {c: ranks[idx].mean() for c, idx in grp_idx.items()}
n_i = {c: len(idx) for c, idx in grp_idx.items()}

# tie correction: T = sum(t^3 - t) over tied-rank groups
unique, counts = np.unique(data, return_counts=True)
tie_t = int(np.sum(counts[counts > 1] ** 3 - counts[counts > 1]))
denom_var = (N * (N + 1) / 12.0) - (tie_t / (12.0 * (N - 1)))  # pooled variance (null)

pairs = [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]
res = {"kw": {"H": float(H), "p_value": float(p_kw), "n_total": int(N),
              "n_per_group": {c: int(n_i[c]) for c in CLASSES},
              "group_mean_pct": {c: float(groups[c].mean() * 100) for c in CLASSES},
              "group_sd_pct": {c: float(np.std(groups[c], ddof=1) * 100) for c in CLASSES}},
       "dunn_bonferroni": {}}
print("\nDunn's post-hoc (Bonferroni, k=3):")
for a, b in pairs:
    z = abs(R_bar[a] - R_bar[b]) / np.sqrt(denom_var * (1.0 / n_i[a] + 1.0 / n_i[b]))
    p_raw = 2 * (1 - stats.norm.cdf(z))
    p_adj = min(1.0, p_raw * 3)
    res["dunn_bonferroni"][f"{a}_vs_{b}"] = {
        "mean_diff_pct": float((groups[a].mean() - groups[b].mean()) * 100),
        "z": float(z), "p_raw": float(p_raw), "p_bonf": float(p_adj)}
    print(f"  {a} vs {b}: mean difference={(groups[a].mean()-groups[b].mean())*100:+.3f}pp  "
          f"z={z:.3f}  p_raw={p_raw:.4g}  p_bonf={p_adj:.4g}")

# Reference: pairwise Mann-Whitney U (uncorrected and Bonferroni-corrected) — used as a robustness cross-check
res["pairwise_mwu_reference"] = {}
print("\nReference: pairwise Mann-Whitney U")
for a, b in pairs:
    u, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
    res["pairwise_mwu_reference"][f"{a}_vs_{b}"] = {
        "u": float(u), "p_raw": float(p), "p_bonf": float(min(1.0, p * 3))}
    print(f"  {a} vs {b}: U={u:.1f}  p_raw={p:.4g}  p_bonf={min(1.0, p*3):.4g}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"\nSaved -> {OUT}")
