# -*- coding: utf-8 -*-
"""
T4 improved analysis — second round: true label × predicted label two-way decomposition
Answering the central objection from external comments:
  "the between-group difference in T4 = true-label effect + predicted-class effect + sampling fluctuation; the three are entangled and cannot be separated"
Methods:
  A. Cross-tabulation reading: the MTL mean of each cell, to see the separate contribution of each factor
  B. Two-way variance decomposition: MTL ~ true + pred (an approximate non-parametric two-way decomposition using a rank transform)
  C. Stratified comparison: hold one factor fixed and vary the other
  D. A weight check using "prediction confidence"
"""
import os, csv, json
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
NAMES = {0: 'CN', 1: 'MCI', 2: 'AD'}


def rank(a):
    n = len(a)
    order = a.argsort()
    r = np.empty(n, dtype=float)
    r[order] = np.arange(1, n + 1)
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    for k in np.where(cnt > 1)[0]:
        r[inv == k] = r[inv == k].mean()
    return r


print('=' * 88)
print('A. Cross-tabulation reading: MTL mass-fraction mean (%) for the 9 cells, with n in parentheses')
print('=' * 88)
print('%-10s %14s %14s %14s %14s' % ('true\\pred', 'pred CN', 'pred MCI', 'pred AD', 'row mean'))
print('-' * 88)
cell = {}
for i in range(3):
    line = '%-10s' % ('true ' + NAMES[i])
    rowvals = []
    for j in range(3):
        m = (true_lab == i) & (pred_lab == j)
        v = mtl[m]
        cell[(i, j)] = (len(v), 100 * v.mean() if len(v) else float('nan'))
        if len(v):
            line += ' %9.4f(%2d)' % (100 * v.mean(), len(v))
            rowvals.append(v)
        else:
            line += ' %14s' % ('—( 0)')
    line += ' %11.4f' % (100 * mtl[true_lab == i].mean())
    print(line)
print('-' * 88)
line = '%-10s' % 'column mean'
for j in range(3):
    v = mtl[pred_lab == j]
    line += ' %9.4f(%2d)' % (100 * v.mean(), len(v))
print(line)

print()
print('=' * 88)
print('B. Key stratification: hold the true label fixed and examine the effect of the predicted label')
print('=' * 88)
print('(If, within the same true group, the MTL means still differ across predicted labels → the prediction side carries an independent signal)')
print()
for i in range(3):
    m_i = true_lab == i
    print('True %s group (n=%d):' % (NAMES[i], m_i.sum()))
    for j in range(3):
        m = m_i & (pred_lab == j)
        if m.sum() == 0:
            continue
        print('    Predicted as %-4s : n=%2d  MTL=%.4f%% ± %.4f' %
              (NAMES[j], m.sum(), 100 * mtl[m].mean(), 100 * mtl[m].std(ddof=1)))
    # within-group KW (if the subgroup sample sizes suffice)
    sub = [mtl[m_i & (pred_lab == j)] for j in range(3)]
    sub = [s for s in sub if len(s) >= 3]
    if len(sub) >= 2:
        allv = np.concatenate(sub)
        r = rank(allv)
        n = len(allv)
        H = 0.0; idx = 0
        for s in sub:
            H += r[idx:idx + len(s)].sum() ** 2 / len(s); idx += len(s)
        H = 12.0 / (n * (n + 1)) * H - 3 * (n + 1)
        # chi2 sf
        def _gu(a, x):
            if x == 0: return 1.0
            if x < a + 1:
                ap, s_, d_ = a, 1.0 / a, 1.0 / a
                for _ in range(500):
                    ap += 1; d_ *= x / ap; s_ += d_
                    if abs(d_) < abs(s_) * 1e-15: break
                return max(0.0, 1.0 - s_ * exp(-x + a * log(x) - lgamma(a)))
            b_, c_ = x + 1 - a, 1e300
            d_ = 1.0 / b_; h_ = d_
            for k in range(1, 500):
                an = -k * (k - a); b_ += 2
                d_ = an * d_ + b_
                if abs(d_) < 1e-300: d_ = 1e-300
                c_ = b_ + an / c_
                if abs(c_) < 1e-300: c_ = 1e-300
                d_ = 1.0 / d_; de = d_ * c_; h_ *= de
                if abs(de - 1.0) < 1e-15: break
            return h_ * exp(-x + a * log(x) - lgamma(a))
        p_in = _gu((len(sub) - 1) / 2.0, H / 2.0)
        print('    >> within-group KW: H=%.3f, p=%.4g  (df=%d)' % (H, p_in, len(sub) - 1))
    print()

print('=' * 88)
print('C. Key stratification: hold the predicted label fixed and examine the effect of the true label')
print('=' * 88)
print('(If, within the same predicted group, the MTL means still differ across true labels → the true side carries an independent signal)')
print()
for j in range(3):
    m_j = pred_lab == j
    print('Predicted %s group (n=%d):' % (NAMES[j], m_j.sum()))
    for i in range(3):
        m = m_j & (true_lab == i)
        if m.sum() == 0:
            continue
        print('    True %-4s : n=%2d  MTL=%.4f%% ± %.4f' %
              (NAMES[i], m.sum(), 100 * mtl[m].mean(), 100 * mtl[m].std(ddof=1)))
    print()

print('=' * 88)
print('D. Two-way variance decomposition (approximate two-way ANOVA in rank space)')
print('=' * 88)
# two-way decomposition on the rank-transformed values: SS_true, SS_pred, SS_resid
y = rank(mtl)
n = len(y)
gm = y.mean()
SS_total = ((y - gm) ** 2).sum()
# true main effect
SS_true = sum(len(y[true_lab == i]) * (y[true_lab == i].mean() - gm) ** 2 for i in range(3))
# pred main effect
SS_pred = sum(len(y[pred_lab == j]) * (y[pred_lab == j].mean() - gm) ** 2 for j in range(3))
# cells (interaction + residual)
SS_cell = 0.0
for i in range(3):
    for j in range(3):
        m = (true_lab == i) & (pred_lab == j)
        if m.sum():
            SS_cell += (m.sum()) * (y[m].mean() - gm) ** 2
SS_inter = SS_cell - SS_true - SS_pred
SS_within = SS_total - SS_cell
print('SS_total   = %10.2f' % SS_total)
print('SS_true    = %10.2f  (%5.2f%% of total)' % (SS_true, 100 * SS_true / SS_total))
print('SS_pred    = %10.2f  (%5.2f%% of total)' % (SS_pred, 100 * SS_pred / SS_total))
print('SS_inter   = %10.2f  (%5.2f%% of total)' % (SS_inter, 100 * SS_inter / SS_total))
print('SS_within  = %10.2f  (%5.2f%% of total)' % (SS_within, 100 * SS_within / SS_total))
print()
print('→ In rank space, the true label explains %.2f%%, the predicted label explains %.2f%%, the interaction %.2f%%, and within-group %.2f%%' %
      (100 * SS_true / SS_total, 100 * SS_pred / SS_total,
       100 * SS_inter / SS_total, 100 * SS_within / SS_total))
print()
if SS_pred > SS_true:
    print('*** The variance contribution of the predicted label (%.2f%%) exceeds that of the true label (%.2f%%) ***' % (
        100 * SS_pred / SS_total, 100 * SS_true / SS_total))
    print('    → consistent with "MTL attention is driven mainly by the model decision side"; grouping T4 by true label')
    print('      indeed did not use the strongest signal as the grouping variable.')
else:
    print('The variance contribution of the true label is larger (%.2f%% vs %.2f%%).' % (
        100 * SS_true / SS_total, 100 * SS_pred / SS_total))

print()
print('=' * 88)
print('E. Comparison with "correctness": ranking of the variance contributions of the three grouping variables')
print('=' * 88)
correct = np.array([1 if t == p else 0 for t, p in zip(true_lab, pred_lab)])
for name, g in [('true label (true)', true_lab), ('predicted label (pred)', pred_lab), ('correctness (correct)', correct)]:
    cats = np.unique(g)
    SS = sum((g == c).sum() * (y[g == c].mean() - gm) ** 2 for c in cats)
    print('  %-18s (k=%d)  SS=%9.2f   %5.2f%% of total' % (name, len(cats), SS, 100 * SS / SS_total))

res = dict(
    crosstab_cells={f'{i}_{j}': dict(n=v[0], mtl_mean_pct=v[1]) for (i, j), v in cell.items()},
    variance_decomp=dict(SS_total=float(SS_total), SS_true=float(SS_true),
                         SS_pred=float(SS_pred), SS_inter=float(SS_inter),
                         SS_within=float(SS_within),
                         pct_true=float(100 * SS_true / SS_total),
                         pct_pred=float(100 * SS_pred / SS_total),
                         pct_inter=float(100 * SS_inter / SS_total),
                         pct_within=float(100 * SS_within / SS_total)),
)
with open(os.path.join(OUT, 'two_way_decomposition.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print()
print('Saved:', os.path.join(OUT, 'two_way_decomposition.json'))
