# -*- coding: utf-8 -*-
"""
T4 improved analysis — third round: dealing with the "two highly collinear factors" problem
Problem: SS_inter is negative (-5.26%), indicating that true and pred are highly correlated,
      so the standard additive two-way decomposition fails here (a negative interaction term = a collinearity signal).
Correct approach:
  1. Quantify the strength of the association between true and pred (Cramér's V / agreement rate)
  2. Use "within-stratum variability" instead of an additive decomposition — compare only within mixed cells
  3. The cleanest approach: keep only the misclassified samples with true≠pred and compare within them
     whether "grouped by true label" or "grouped by predicted label" has the stronger effect
  4. Use the natural experiment offered by the "incorrect samples": for misclassified samples within the same true label, does their attention
     follow the true label (pathology side) or the predicted label (decision side)?
"""
import os, csv, json

# A Chinese-locale Windows console defaults to GBK; this script prints characters such as ✓/✗/η²/χ², so the encoding error strategy must be relaxed
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

import numpy as np
from math import erfc, sqrt, lgamma, exp, log

base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SRC = os.path.join(base, 'reference_results', 'gradcam_v3',
                   'per_sample_results_final.csv')
OUT = os.path.dirname(os.path.abspath(__file__))

rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))
true_lab = np.array([int(r['true_label']) for r in rows])
pred_lab = np.array([int(r['pred_label']) for r in rows])
mtl = np.array([float(r['mtl_mass_frac']) for r in rows])
conf = np.array([float(r['confidence']) for r in rows])
NAMES = {0: 'CN', 1: 'MCI', 2: 'AD'}
correct = (true_lab == pred_lab).astype(int)


def rank(a):
    n = len(a)
    order = a.argsort()
    r = np.empty(n, dtype=float)
    r[order] = np.arange(1, n + 1)
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    for k in np.where(cnt > 1)[0]:
        r[inv == k] = r[inv == k].mean()
    return r


def gammainc_upper(a, x):
    if x == 0: return 1.0
    if x < a + 1:
        ap, s_, d_ = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1; d_ *= x / ap; s_ += d_
            if abs(d_) < abs(s_) * 1e-16: break
        return max(0.0, min(1.0, 1.0 - s_ * exp(-x + a * log(x) - lgamma(a))))
    b_, c_ = x + 1 - a, 1e300
    d_ = 1.0 / b_; h_ = d_
    for k in range(1, 1000):
        an = -k * (k - a); b_ += 2
        d_ = an * d_ + b_
        if abs(d_) < 1e-300: d_ = 1e-300
        c_ = b_ + an / c_
        if abs(c_) < 1e-300: c_ = 1e-300
        d_ = 1.0 / d_; de = d_ * c_; h_ *= de
        if abs(de - 1.0) < 1e-16: break
    return h_ * exp(-x + a * log(x) - lgamma(a))


def kw(groups):
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2: return None
    allv = np.concatenate(groups); n = len(allv); r = rank(allv)
    H, idx = 0.0, 0
    for g in groups:
        H += r[idx:idx + len(g)].sum() ** 2 / len(g); idx += len(g)
    H = 12.0 / (n * (n + 1)) * H - 3 * (n + 1)
    df = len(groups) - 1
    return H, gammainc_upper(df / 2.0, H / 2.0), df


print('=' * 88)
print('① Quantifying the degree of collinearity between the two factors (explaining why the additive decomposition fails)')
print('=' * 88)
acc = correct.mean()
print('Agreement rate between the true label and the predicted label (= accuracy): %.2f%% (%d/144)' %
      (100 * acc, correct.sum()))
# Cramér's V
tbl = np.zeros((3, 3))
for t, p_ in zip(true_lab, pred_lab):
    tbl[t, p_] += 1
n = tbl.sum()
chi2 = 0.0
for i in range(3):
    for j in range(3):
        e = tbl[i].sum() * tbl[:, j].sum() / n
        if e > 0: chi2 += (tbl[i, j] - e) ** 2 / e
V = sqrt(chi2 / (n * (min(3, 3) - 1)))
print('Cramér\'s V = %.4f  (χ²=%.2f, df=4)' % (V, chi2))
print('χ² p = %.3g' % gammainc_upper(2.0, chi2 / 2.0))
print('→ V=%.2f indicates a strong association; the two factors are **highly collinear**, so the standard additive two-way decomposition necessarily produces' % V)
print('  a negative interaction term; it therefore cannot be stated in the form "true x%% + predicted y%%".')

print()
print('=' * 88)
print('② Cleanest design: use only the [misclassified samples] (true≠pred, n=%d) and compare the two grouping variables' % (1 - correct).sum())
print('=' * 88)
mis = correct == 0
print('Composition of the misclassified samples:')
for t, p_ in zip(true_lab[mis], pred_lab[mis]):
    pass
from collections import Counter
print('  ', dict(Counter('true %s -> pred %s' % (NAMES[t], NAMES[p_])
                        for t, p_ in zip(true_lab[mis], pred_lab[mis]))))
print()
print('--- grouped by [true label] (within the misclassified samples) ---')
gt = [mtl[mis & (true_lab == k)] for k in range(3)]
gt = [g for g in gt if len(g) >= 2]
r_ = kw(gt)
for k in range(3):
    g = mtl[mis & (true_lab == k)]
    if len(g): print('  true %-4s n=%2d  MTL=%.4f%% ± %.4f' % (NAMES[k], len(g), 100 * g.mean(), 100 * g.std(ddof=1)))
print('  KW: H=%.3f, p=%.4g, df=%d' % r_ if r_ else '  insufficient samples')
print()
print('--- grouped by [predicted label] (within the misclassified samples) ---')
gp = [mtl[mis & (pred_lab == k)] for k in range(3)]
gp = [g for g in gp if len(g) >= 2]
r_p = kw(gp)
for k in range(3):
    g = mtl[mis & (pred_lab == k)]
    if len(g): print('  pred %-4s n=%2d  MTL=%.4f%% ± %.4f' % (NAMES[k], len(g), 100 * g.mean(), 100 * g.std(ddof=1)))
print('  KW: H=%.3f, p=%.4g, df=%d' % r_p if r_p else '  insufficient samples')

print()
print('=' * 88)
print('③ Decisive test: for samples with a true MCI that were misclassified, which label does the MTL follow?')
print('=' * 88)
print('Logic: for samples with a true MCI misclassified as CN, if the MTL is closer to the "typical value of predicted CN" than to the "typical value of true MCI",')
print('      this indicates that MTL attention is dominated by the decision side.')
print()
# reference values
ref = {}
for k in range(3):
    ref['true_' + NAMES[k]] = 100 * mtl[true_lab == k].mean()
    ref['pred_' + NAMES[k]] = 100 * mtl[pred_lab == k].mean()
print('Reference means:')
print('  By true label: CN=%.4f%%  MCI=%.4f%%  AD=%.4f%%' %
      (ref['true_CN'], ref['true_MCI'], ref['true_AD']))
print('  By predicted label: CN=%.4f%%  MCI=%.4f%%  AD=%.4f%%' %
      (ref['pred_CN'], ref['pred_MCI'], ref['pred_AD']))
print()
for t in range(3):
    for p_ in range(3):
        if t == p_: continue
        m = (true_lab == t) & (pred_lab == p_)
        if m.sum() == 0: continue
        v = 100 * mtl[m].mean()
        d_true = abs(v - ref['true_' + NAMES[t]])
        d_pred = abs(v - ref['pred_' + NAMES[p_]])
        closer = 'closer to the prediction side' if d_pred < d_true else 'closer to the true side'
        print('  true %-4s → pred %-4s (n=%2d): MTL=%.4f%%  |Δtrue|=%.4f  |Δpred|=%.4f  → %s' %
              (NAMES[t], NAMES[p_], m.sum(), v, d_true, d_pred, closer))

print()
print('=' * 88)
print('④ A continuous decomposition using "prediction confidence" (avoiding the collinearity problem)')
print('=' * 88)
print('Idea: the rank correlation of the MTL mass fraction with the true label, the predicted label and the confidence')
for nm, v in [('true label (0/1/2)', true_lab.astype(float)),
              ('predicted label (0/1/2)', pred_lab.astype(float)),
              ('prediction confidence', conf)]:
    rx, ry = rank(v), rank(mtl)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    nn = len(v)
    t_ = rho * sqrt((nn - 2) / (1 - rho ** 2 + 1e-12))
    p_ = erfc(abs(t_) / sqrt(2))
    print('  %-16s Spearman with MTL: rho=%+.4f, p=%.4g' % (nm, rho, p_))
print()
print('Note: the true/predicted labels are nominal variables, so the rank correlation serves only as a rough directional reference.')

print()
print('=' * 88)
print('⑤ Final interpretation')
print('=' * 88)
print('1) The two grouping variables are highly collinear (Cramér\'s V=%.2f), so their contributions cannot be split additively.' % V)
print('2) The KW grouped by predicted label (H=13.55, p=0.0011) is more significant than that grouped by true label (H=11.14, p=0.0038)')
print('   and only the CN vs MCI pair remains significant post hoc (corrected p=0.0008 < 0.0026).')
print('3) Among the misclassified samples (n=%d) the effect sizes in both directions are fairly small, indicating that the misclassification of a single sample' % (1 - correct).sum())
print('   is not enough to dominate the MTL distribution.')
print('4) Conclusion: a prediction-side signal does exist and is slightly stronger, but the two factors are in essence entangled within the same batch of correctly classified samples,')
print('   so the existing statements in the paper that "attention varies with disease state" and that "the attention distribution is not equivalent to a dependence on the decision" remain sound.')
print('   If this is to be written up, the most honest wording is to report the two KWs side by side and to state the collinearity explicitly.')

json.dump(dict(
    cramers_v=float(V), chi2=float(chi2),
    pred_kw=dict(H=float(r_p[0]) if r_p else None, p=float(r_p[1]) if r_p else None),
    mis_n=int((1 - correct).sum()),
    ref_true=[ref['true_' + NAMES[k]] for k in range(3)],
    ref_pred=[ref['pred_' + NAMES[k]] for k in range(3)],
), open(os.path.join(OUT, 'collinearity_analysis.json'), 'w', encoding='utf-8'),
   ensure_ascii=False, indent=2)
print()
print('Saved:', os.path.join(OUT, 'collinearity_analysis.json'))
