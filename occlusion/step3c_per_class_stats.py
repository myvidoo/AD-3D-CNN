# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 3c: stratification by true class + class-level directionality analysis (CPU, seconds).

Corresponds to the paper: supplementary S6 (AD-group recall drops; AD McNemar p=0.25, not significant).

Motivation: the AD row of the baseline confusion matrix = [0, 3, 45] (no AD→CN errors), and after
      occlusion the AD row = [3, 3, 42]
      → a net change of 0 in overall accuracy masks a **class asymmetry** (AD worsens, CN slightly improves).

Answers three questions:
  Q1 Is the Δp_true after main-mask occlusion stratified by true class (CN / MCI / AD)?
  Q2 Does "AD-specific worsening" hold (a significant drop in AD-class recall)?
  Q3 Is the sign of Δp_true consistent within each class, and is the small n sufficient to support the conclusion?

Definition: consistent with step3 (the non-saturated n=80 as the main analysis, with all samples n=144 reported in parallel).
⚠ This script only performs description and tests; it makes no causal claim.

Run: python occlusion/step3c_per_class_stats.py"""

# Source: 27_occlusion_experiment/step3c_perclass.py (the computation logic is unchanged line by line; only the
# "environment bootstrap layer" was changed to be provided uniformly by occlusion_config; see
# occlusion/README.md for the refactoring notes).

import json
import sys
import numpy as np
from scipy import stats as st

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from occlusion_config import (  # noqa: E402
    OUT, setup_stdout,
)

setup_stdout()

raw = np.load(OUT / 'occlusion_raw.npz', allow_pickle=True)
cond_names = [str(x) for x in raw['cond_names']]
file_ids = [str(x) for x in raw['file_ids']]
true_lab = raw['true_labels']
cond_idx = raw['cond_idx']
P0 = raw['p0'].astype(np.float64)
P1 = raw['p_after'].astype(np.float64)

uniq = sorted(set(file_ids))
N = len(uniq)
i2r = {f: i for i, f in enumerate(uniq)}


def mat_of(c):
    rows = {i: None for i in range(N)}
    for k, f in enumerate(file_ids):
        if cond_names[cond_idx[k]] == c:
            rows[i2r[f]] = P1[k]
    return np.array([rows[i] for i in range(N)])


P0m = np.array([P0[[k for k in range(len(file_ids))
                   if cond_names[cond_idx[k]] == cond_names[0] and file_ids[k] == f][0]]
                for f in uniq])
TL = np.array([true_lab[[k for k in range(len(file_ids)) if file_ids[k] == f][0]]
               for f in uniq])
IDX = np.arange(N)
CLS = ['CN', 'MCI', 'AD']

SAT = P0m.max(1) > 0.999999
NS = ~SAT
print("=" * 84)
print("Step 3c  stratification by true class + class-level directionality")
print("=" * 84)
print("all samples n=%d, non-saturated n=%d, saturated n=%d" % (N, NS.sum(), SAT.sum()))

MAIN = 'MTL_d100_F1const'
Pa = mat_of(MAIN)
pred0, pred1 = P0m.argmax(1), Pa.argmax(1)
d_all = P0m[IDX, TL] - Pa[IDX, TL]

R = {'scope': {'n_all': int(N), 'n_nonsat': int(NS.sum()), 'n_sat': int(SAT.sum())},
     'per_class_delta': {}, 'per_class_flip': {}, 'ad_specificity': {}}

# ==================== Q1 Δp_true stratified by true class ====================
print("\n" + "-" * 84)
print("Q1  Δp_true stratified by true class (main mask %s)" % MAIN)
print("-" * 84)
print("%-5s %5s %5s | %14s %14s | %8s %8s %10s" %
      ("class", "n_all", "n_nonsat", "nonsat median Δ", "nonsat mean Δ", "n+/n-", "Wilcoxon p", "effect r"))
for ci, cn in enumerate(CLS):
    m_all = TL == ci
    m_ns = m_all & NS
    d = d_all[m_ns]
    if len(d) == 0 or not np.any(d != 0):
        W, p = np.nan, np.nan
    else:
        W, p = st.wilcoxon(d)
    r = np.nan if not np.isfinite(p) else abs(st.norm.ppf(max(p / 2, 1e-300))) / np.sqrt(len(d))
    npos, nneg = int((d > 0).sum()), int((d < 0).sum())
    med, mean = float(np.median(d)), float(d.mean())
    print("%-5s %5d %5d | %+14.5f %+14.5f | %4d/%4d %10.4g %10.3f"
          % (cn, int(m_all.sum()), int(m_ns.sum()), med, mean, npos, nneg, p, r))
    R['per_class_delta'][cn] = dict(
        n_all=int(m_all.sum()), n_nonsat=int(m_ns.sum()),
        median_nonsat=med, mean_nonsat=mean, n_pos=npos, n_neg=nneg,
        W=float(W) if np.isfinite(W) else None,
        p=float(p) if np.isfinite(p) else None,
        effect_r=float(r) if np.isfinite(r) else None)

# between-class comparison (non-saturated, Kruskal-Wallis + pairwise Mann-Whitney)
groups = [d_all[(TL == ci) & NS] for ci in range(3)]
H, pH = st.kruskal(*groups)
print("\nbetween-class difference (non-saturated): Kruskal-Wallis H = %.3f, p = %.4g" % (H, pH))
R['per_class_delta']['_KW'] = dict(H=float(H), p=float(pH))
for i in range(3):
    for j in range(i + 1, 3):
        U, pu = st.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
        print("  %s vs %s : U = %.0f, p = %.4g" % (CLS[i], CLS[j], U, pu))
        R['per_class_delta']['_pair_%s_%s' % (CLS[i], CLS[j])] = \
            dict(U=float(U), p=float(pu))

# ==================== Q2 AD-specific worsening ====================
print("\n" + "-" * 84)
print("Q2  class-level flip direction (all-samples definition; flips are unaffected by saturation)")
print("-" * 84)
print("%-5s %6s | %8s %8s %8s | %8s %8s %8s" %
      ("true class", "n", "baseline recall", "occluded recall", "Δrecall", "flip-to-correct", "flip-to-wrong", "net"))
for ci, cn in enumerate(CLS):
    m = TL == ci
    n = int(m.sum())
    rec0 = float(np.mean(pred0[m] == TL[m]))
    rec1 = float(np.mean(pred1[m] == TL[m]))
    fix = int(np.sum(m & (pred0 != TL) & (pred1 == TL)))
    brk = int(np.sum(m & (pred0 == TL) & (pred1 != TL)))
    print("%-5s %6d | %8.4f %8.4f %+8.4f | %8d %8d %+8d"
          % (cn, n, rec0, rec1, rec1 - rec0, fix, brk, fix - brk))
    R['per_class_flip'][cn] = dict(n=n, recall_base=rec0, recall_occ=rec1,
                                   recall_delta=rec1 - rec0,
                                   n_fix=fix, n_break=brk, net=fix - brk)

# whether the change in AD-class recall is significant: McNemar exact test (paired within AD only)
m_ad = TL == 2
b = int(np.sum(m_ad & (pred0 == 2) & (pred1 != 2)))   # correct→wrong
c = int(np.sum(m_ad & (pred0 != 2) & (pred1 == 2)))   # wrong→correct
if b + c > 0:
    p_mc = float(st.binomtest(b, b + c, 0.5).pvalue)
else:
    p_mc = float('nan')
print("\nAD-class McNemar exact test (correct→wrong %d vs wrong→correct %d): p = %.4g" % (b, c, p_mc))
print("→ %s" % ("significant" if p_mc < 0.05 else "⚠️ not significant (insufficient sample size, b+c=%d)" % (b + c)))
R['ad_specificity'] = dict(ad_break=int(b), ad_fix=int(c), mcnemar_p=p_mc,
                           significant=bool(p_mc < 0.05) if np.isfinite(p_mc) else False)

# McNemar over all classes (overall symmetry)
b_all = int(np.sum((pred0 == TL) & (pred1 != TL)))
c_all = int(np.sum((pred0 != TL) & (pred1 == TL)))
p_all = float(st.binomtest(b_all, b_all + c_all, 0.5).pvalue) if b_all + c_all else float('nan')
print("McNemar over all cases (correct→wrong %d vs wrong→correct %d): p = %.4g" % (b_all, c_all, p_all))
R['ad_specificity']['all_break'] = b_all
R['ad_specificity']['all_fix'] = c_all
R['ad_specificity']['all_mcnemar_p'] = p_all

# ==================== Q3 do the controls show the same between-class asymmetry ====================
print("\n" + "-" * 84)
print("Q3  does the change in AD recall also appear in the controls (ruling out a \"general perturbation\" explanation)")
print("-" * 84)
print("%-22s %10s %10s %10s %10s" % ("condition", "AD recall Δ", "CN recall Δ", "MCI recall Δ", "AD McNemar p"))
R['controls_ad_asymmetry'] = {}
for cond in cond_names:
    Pc = mat_of(cond)
    pc_ = Pc.argmax(1)
    row = []
    for ci in range(3):
        m = TL == ci
        row.append(float(np.mean(pc_[m] == TL[m])) - float(np.mean(pred0[m] == TL[m])))
    bb = int(np.sum(m_ad & (pred0 == 2) & (pc_ != 2)))
    cc = int(np.sum(m_ad & (pred0 != 2) & (pc_ == 2)))
    pm = float(st.binomtest(bb, bb + cc, 0.5).pvalue) if bb + cc else float('nan')
    print("%-22s %+10.4f %+10.4f %+10.4f %10.4g" % (cond, row[2], row[0], row[1], pm))
    R['controls_ad_asymmetry'][cond] = dict(ad_delta=row[2], cn_delta=row[0], mci_delta=row[1],
                                            mcnemar_p=pm, ad_break=bb, ad_fix=cc)

# ==================== conclusion ====================
print("\n" + "=" * 84)
print("interpretation")
print("=" * 84)
ad = R['per_class_flip']['AD']
if np.isfinite(p_mc) and p_mc < 0.05:
    print("AD-class recall drops significantly (%.4f → %.4f, p=%.4g), suggesting MTL occlusion has the largest effect on AD discrimination."
          % (ad['recall_base'], ad['recall_occ'], p_mc))
else:
    print("⚠️ the change in AD-class recall (%.4f → %.4f, %d cases correct→wrong) **does not reach significance** (McNemar p=%.4g, b+c=%d)."
          % (ad['recall_base'], ad['recall_occ'], b, p_mc, b + c))
    print("   If this claim is needed, the sample should be enlarged, or S6 should explicitly flag it as \"exploratory, insufficient sample size\".")

json.dump(R, open(OUT / 'perclass_stats.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2, default=float)
print("\nsaved: results/perclass_stats.json")
print("=" * 84)
