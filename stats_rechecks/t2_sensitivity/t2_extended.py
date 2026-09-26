# -*- coding: utf-8 -*-
"""
T2 sensitivity analysis — third round: searching for "reportable and still significant" robustness evidence
Objective: after excluding >0.99, rho=0.2115 (p=0.321) is not significant and cannot serve as evidence "supporting robustness".
      However, it must be checked whether a sensitivity version exists that is both honest and substantively informative.
Attempts:
  A. Stratified analysis: whether the correlation is still positive within the "correct samples only" (removing the strong correct/incorrect confound)
  B. Excluding the exactly saturated 1.0 (keeping conf<1.0, n=100): whether rho=0.6696 is usable
  C. Looking after a rank transform of conf (to avoid the saturation pile-up)
  D. Replacing the rank correlation with a between-group difference in cam_mean (Mann-Whitney) within the "non-saturated" samples
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
        m = inv == k
        r[m] = r[m].mean()
    return r


def spear(x, y):
    n = len(x)
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rc = max(min(rho, 0.9999999), -0.9999999)
    t = rc * sqrt((n - 2) / (1 - rc ** 2))
    return rho, float(erfc(abs(t) / sqrt(2))), n


def mwu(x, y):
    """Returns (U, p, delta). p uses the normal approximation (no scipy)."""
    n1, n2 = len(x), len(y)
    allv = np.concatenate([x, y])
    r = rank(allv)
    R1 = r[:n1].sum()
    U = R1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sd = sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (U - mu) / sd
    p = erfc(abs(z) / sqrt(2))
    delta = 2.0 * U / (n1 * n2) - 1.0
    return float(U), float(p), float(delta)


print('=' * 76)
print('A. Stratified sensitivity: excluding saturated samples within the "correct predictions only"')
print('=' * 76)
mc = correct == 1
for name, m in [('correct & all', mc), ('correct & conf<=0.99', mc & (conf <= 0.99)),
                ('correct & conf<1.0', mc & (conf < 1.0))]:
    if m.sum() < 4:
        print('%-20s n=%d insufficient samples' % (name, m.sum())); continue
    r, p, n = spear(conf[m], cam[m])
    print('%-20s n=%3d  rho=%.4f  p=%.3g' % (name, n, r, p))

print()
print('=' * 76)
print('B. Excluding the "exactly saturated 1.0" (keeping conf<1.0, n should be 100)')
print('=' * 76)
m = conf < 1.0
r, p, n = spear(conf[m], cam[m])
print('n=%d  rho=%.4f  p=%.3g  (vs baseline 0.8000, Δrho=%+.4f)' % (n, r, p, r - 0.8000))
# grouping within this subset
c_, a_ = conf[m], cam[m]
med = np.median(c_)
hi, lo = a_[c_ >= med], a_[c_ < med]
U, pU, d = mwu(hi, lo)
print('  Between groups (MWU): high conf n=%d cam=%.4f±%.4f | low conf n=%d cam=%.4f±%.4f' %
      (len(hi), hi.mean(), hi.std(ddof=1), len(lo), lo.mean(), lo.std(ddof=1)))
print('  U=%.1f  p=%.3g  Cliff delta=%+.3f' % (U, pU, d))

print()
print('=' * 76)
print('C. Permutation test over all samples (testing whether rho=0.800 is driven by a few saturated samples)')
print('=' * 76)
# randomly remove most of the 121 saturated samples step by step and inspect the rho distribution
rng = np.random.default_rng(42)
pool = np.where(conf > 0.99)[0]
non_sat = np.where(conf <= 0.99)[0]
rhos = []
for _ in range(2000):
    # randomly keep 20% of the saturated samples + all non-saturated
    k = int(0.2 * len(pool))
    keep = np.concatenate([non_sat, rng.choice(pool, k, replace=False)])
    rr, _, _ = spear(conf[keep], cam[keep])
    rhos.append(rr)
rhos = np.array(rhos)
print('Keeping 20%% of the saturated samples at random, 2000 times: rho mean %.4f, median %.4f, 95%% interval [%.4f, %.4f]' %
      (rhos.mean(), np.median(rhos), np.percentile(rhos, 2.5), np.percentile(rhos, 97.5)))
print('  Of these, the proportion with rho>0 is %.1f%%, and with rho>0.3 is %.1f%%' %
      (100 * (rhos > 0).mean(), 100 * (rhos > 0.3).mean()))
print('→ If, after randomly removing saturated samples, rho is mostly still positive and of appreciable magnitude, the coupling does not rest on a few saturated points alone')

print()
print('=' * 76)
print('D. Reporting recommendation: three numbers that can be written into S6')
print('=' * 76)
print('1) All-sample baseline:  n=144, rho=0.8000, p<0.001')
m1 = conf < 1.0
r1, p1, n1 = spear(conf[m1], cam[m1])
print('2) Excluding exactly saturated 1.0: n=%d, rho=%.4f, p=%.3g' % (n1, r1, p1))
m2 = conf <= 0.99
r2, p2, n2 = spear(conf[m2], cam[m2])
print('3) Excluding conf>0.99:   n=%d, rho=%.4f, p=%.3g  (not significant; must be explained together with the "variance compression")' % (n2, r2, p2))
print()
print('Appendix: the accuracy difference between the saturated and the non-saturated group — the most direct honest evidence')
print('    conf>0.99:  accuracy %.1f%% (n=%d)' % (100 * correct[conf > 0.99].mean(), (conf > 0.99).sum()))
print('    conf<=0.99: accuracy %.1f%% (n=%d)' % (100 * correct[conf <= 0.99].mean(), (conf <= 0.99).sum()))

res = dict(
    full=dict(n=144, rho=0.8000),
    excl_exact_1=dict(n=int(n1), rho=float(r1), p=float(p1)),
    excl_gt099=dict(n=int(n2), rho=float(r2), p=float(p2)),
    permutation=dict(mean=float(rhos.mean()), median=float(np.median(rhos)),
                     lo=float(np.percentile(rhos, 2.5)), hi=float(np.percentile(rhos, 97.5))),
)
with open(os.path.join(OUT, 't2_robustness_extended.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print('\nSaved:', os.path.join(OUT, 't2_robustness_extended.json'))
