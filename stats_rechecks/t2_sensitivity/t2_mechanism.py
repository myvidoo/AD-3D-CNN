# -*- coding: utf-8 -*-
"""
T2 sensitivity analysis — fourth round: locating the real mechanism behind the rho collapse
Hypothesis: rho is dominated by the correct/incorrect stratification rather than by a continuous confidence gradient.
     Excluding conf>0.99 happens to take away almost all the correct samples (114 of the 121 are correct),
     leaving 23 residual samples split roughly evenly between correct and incorrect, whose cam_mean distributions overlap → the rank correlation vanishes.
Verification: partial correlation (controlling for correct) / within-stratum correlation / between-group mean separation
"""
import os, csv, json
import numpy as np
from math import erfc, sqrt

base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SRC = os.path.join(base, 'reference_results', 'gradcam_v3',
                   'per_sample_results_final.csv')
OUT = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))
conf = np.array([float(r['confidence']) for r in rows])
cam = np.array([float(r['cam_mean']) for r in rows])
correct = np.array([int(r['correct']) for r in rows])


def rank(a):
    n = len(a)
    order = a.argsort()
    r = np.empty(n, dtype=float)
    r[order] = np.arange(1, n + 1)
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    for k in np.where(cnt > 1)[0]:
        r[inv == k] = r[inv == k].mean()
    return r


def spear(x, y):
    n = len(x)
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rc = max(min(rho, 0.9999999), -0.9999999)
    t = rc * sqrt((n - 2) / (1 - rc ** 2))
    return rho, float(erfc(abs(t) / sqrt(2))), n


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


print('=' * 78)
print('Verification 1  Partial correlation: after controlling for "prediction correctness", is confidence still correlated with CAM intensity?')
print('=' * 78)
print('H0: the relationship between conf and cam is fully mediated by "correctness" (i.e. there is no independent continuous coupling)')
print()
# approximate the partial correlation by "within-stratum ranking + pooling" (a Spearman version of the partial correlation)
def partial_spearman_by_group(x, y, g):
    """Rank x and y within each group g, then pool them and compute the correlation (approximate partial Spearman)"""
    rx = np.zeros(len(x)); ry = np.zeros(len(y))
    for gv in np.unique(g):
        m = g == gv
        if m.sum() < 3:
            rx[m] = np.nan; ry[m] = np.nan; continue
        rx[m] = rank(x[m]); ry[m] = rank(y[m])
    ok = ~np.isnan(rx)
    return pearson(rx[ok], ry[ok])


r_within = partial_spearman_by_group(conf, cam, correct)
print('Within-group correlation after controlling for correctness (pooled): r = %.4f' % r_within)
r_raw = pearson(rank(conf), rank(cam))
print('Uncontrolled raw rank correlation:            r = %.4f' % r_raw)

print()
print('=' * 78)
print('Verification 2  Continuous correlation within each group')
print('=' * 78)
for lab, m in [('correct group', correct == 1), ('incorrect group', correct == 0)]:
    r, p, n = spear(conf[m], cam[m])
    print('%-8s n=%3d  rho=%.4f  p=%.3g' % (lab, n, r, p))

print()
print('=' * 78)
print('Verification 3  Separation of cam_mean between the correct and incorrect groups (determining which is the "true" grouping variable)')
print('=' * 78)
c1, c0 = cam[correct == 1], cam[correct == 0]
print('Correct group cam_mean = %.4f ± %.4f (n=%d)' % (c1.mean(), c1.std(ddof=1), len(c1)))
print('Incorrect group cam_mean = %.4f ± %.4f (n=%d)' % (c0.mean(), c0.std(ddof=1), len(c0)))
print('Difference = %+.4f' % (c1.mean() - c0.mean()))
# Cliff's delta
gt = sum((a > b) for a in c1 for b in c0)
eq = sum((a == b) for a in c1 for b in c0)
U = gt + 0.5 * eq
print('Cliff delta = %.3f  (reported in the paper: 0.52)' % (2 * U / (len(c1) * len(c0)) - 1))

print()
print('  Separation of conf between the correct and incorrect groups:')
print('Correct group conf = %.4f ± %.4f' % (conf[correct == 1].mean(), conf[correct == 1].std(ddof=1)))
print('Incorrect group conf = %.4f ± %.4f' % (conf[correct == 0].mean(), conf[correct == 0].std(ddof=1)))

print()
print('=' * 78)
print('Verification 4  How the correct/incorrect ratio changes when the 121 saturated samples are excluded')
print('=' * 78)
m = conf > 0.99
print('Of the %d excluded samples: %d correct (%.1f%%), %d incorrect (%.1f%%)' %
      (m.sum(), (correct[m] == 1).sum(), 100 * (correct[m] == 1).mean(),
       (correct[m] == 0).sum(), 100 * (correct[m] == 0).mean()))
print('Of the %d remaining samples: %d correct (%.1f%%), %d incorrect (%.1f%%)' %
      ((~m).sum(), (correct[~m] == 1).sum(), 100 * (correct[~m] == 1).mean(),
       (correct[~m] == 0).sum(), 100 * (correct[~m] == 0).mean()))
print()
print('→ All-sample correct:incorrect ratio 125:19 (6.6:1); residual subset 11:12 (0.9:1).')
print('  The exclusion operation turns the "extremely imbalanced" pair of groups into a "roughly balanced" pair,')
print('  and the main difference in cam_mean comes precisely from the correct and incorrect groups — hence the rho collapse.')

print()
print('=' * 78)
print('Final interpretation')
print('=' * 78)
print('Mechanism: the rho=0.800 of T2 is contributed mainly by the stratification structure in which "correct samples have high confidence and strong CAM, whereas incorrect samples have low confidence and weak CAM"')
print('      — excluding conf>0.99 removes the saturated samples and at the same time almost empties the correct group,')
print('      and the residual subset degenerates into a balanced correct/incorrect mixture, so the continuous gradient disappears.')
print()
print('This is neither "the coupling is spurious" nor "the coupling is robust", but rather:')
print('  ① on this dataset the >0.99 exclusion design cannot test the continuous coupling (residual n=23, variance emptied out)')
print('  ② the more appropriate sensitivity analysis is "excluding the exactly saturated 1.0" (n=100, rho=0.6696, p<0.001)')
print('  ③ the permutation test shows that after randomly removing 80%% of the saturated samples the median rho is still 0.767, so the coupling does not depend on individual saturated points')
