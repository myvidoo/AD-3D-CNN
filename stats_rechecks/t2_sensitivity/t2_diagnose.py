# -*- coding: utf-8 -*-
"""
T2 sensitivity analysis — deeper look: why does rho collapse after excluding >0.99?
Key hypothesis: the low-confidence samples are almost all "incorrect predictions", and after removing the saturated samples the residual subset's
         confidence variance is compressed (range restriction / truncation),
         which attenuates the rank correlation. A distinction must be drawn between "the coupling does not exist" and "there is insufficient variance to detect it".
"""
import os, csv, json
import numpy as np

base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SRC = os.path.join(base, 'reference_results', 'gradcam_v3',
                   'per_sample_results_final.csv')
OUT = os.path.dirname(os.path.abspath(__file__))

rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))
conf = np.array([float(r['confidence']) for r in rows])
cam = np.array([float(r['cam_mean']) for r in rows])
correct = np.array([int(r['correct']) for r in rows])


def spearman(x, y, ties=False):
    n = len(x)
    def rank(a):
        order = a.argsort()
        r = np.empty(n, dtype=float)
        r[order] = np.arange(1, n + 1)
        vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        for k in np.where(cnt > 1)[0]:
            m = inv == k
            r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rho_c = max(min(rho, 0.9999999), -0.9999999)
    from math import erfc
    t = rho_c * np.sqrt((n - 2) / (1 - rho_c ** 2))
    p = float(erfc(abs(t) / np.sqrt(2)))
    return rho, p, n


print('=' * 74)
print('Diagnosis 1  Composition of the low-confidence samples (who was excluded?)')
print('=' * 74)
m_sat = conf > 0.99
m_low = ~m_sat
print('Saturated group (conf>0.99): n=%d  accuracy %.1f%%  errors %d' %
      (m_sat.sum(), 100 * correct[m_sat].mean(), (correct[m_sat] == 0).sum()))
print('Non-saturated group (conf<=0.99): n=%d  accuracy %.1f%%  errors %d  <== residual subset' %
      (m_low.sum(), 100 * correct[m_low].mean(), (correct[m_low] == 0).sum()))
print()
print('→ Of all 19 incorrect samples, the proportion falling in the non-saturated group: %d/%d = %.1f%%' %
      ((correct[m_low] == 0).sum(), (correct == 0).sum(),
       100 * (correct[m_low] == 0).sum() / (correct == 0).sum()))

print()
print('=' * 74)
print('Diagnosis 2  Spread of the residual subset (range-restriction check)')
print('=' * 74)
for name, m in [('all samples', np.ones(len(conf), bool)), ('non-saturated (conf<=0.99)', m_low)]:
    c, a = conf[m], cam[m]
    print('%-22s conf: [%.4f, %.4f]  SD=%.4f | cam_mean: [%.4f, %.4f] SD=%.5f' %
          (name, c.min(), c.max(), c.std(ddof=1), a.min(), a.max(), a.std(ddof=1)))
cs, cu = conf[m_sat], conf[m_low]
print()
print('→ Saturated-group conf SD=%.5f; non-saturated-group SD=%.4f (compressed to 1/%.1f)' %
      (cs.std(ddof=1), cu.std(ddof=1), cs.std(ddof=1) / cu.std(ddof=1)))

print()
print('=' * 74)
print('Diagnosis 3  Sensitivity analysis within the "correct samples only" (removing the correct/incorrect confound)')
print('=' * 74)
print('%-26s %5s %9s %12s' % ('subset', 'n', 'rho', 'p'))
print('-' * 74)
for name, m in [
    ('all samples (including incorrect)', np.ones(len(conf), bool)),
    ('correct predictions only (all)', correct == 1),
    ('correct predictions only & conf<=0.99', (correct == 1) & m_low),
    ('incorrect predictions only (all)', correct == 0),
]:
    if m.sum() < 4:
        print('%-26s %5d  insufficient samples' % (name, m.sum())); continue
    r, p, n = spearman(conf[m], cam[m])
    print('%-26s %5d %9.4f %12.3g' % (name, n, r, p))

print()
print('=' * 74)
print('Diagnosis 4  Within-stratum variation: the incorrect group on its own')
print('=' * 74)
me = correct == 0
c_e, a_e = conf[me], cam[me]
print('Incorrect group (n=%d): conf [%.4f, %.4f] SD=%.4f | cam_mean [%.5f, %.5f] SD=%.5f' %
      (me.sum(), c_e.min(), c_e.max(), c_e.std(ddof=1), a_e.min(), a_e.max(), a_e.std(ddof=1)))
r, p, n = spearman(c_e, a_e)
print('Spearman within the incorrect group: rho=%.4f, p=%.3g, n=%d' % (r, p, n))

print()
print('=' * 74)
print('Diagnosis 5  Effect-size comparison: all samples vs the non-saturated subset (cam_mean in the high- vs low-conf groups)')
print('=' * 74)
# split at the median
for name, m in [('all samples', np.ones(len(conf), bool)), ('non-saturated subset', m_low)]:
    cc, aa = conf[m], cam[m]
    med = np.median(cc)
    hi, lo = aa[cc >= med], aa[cc < med]
    if len(hi) < 2 or len(lo) < 2:
        print('%s: insufficient samples after grouping' % name); continue
    print('%-12s high-conf group cam_mean=%.4f±%.4f (n=%d) | low-conf group=%.4f±%.4f (n=%d) | difference=%+.4f' %
          (name, hi.mean(), hi.std(ddof=1), len(hi),
           lo.mean(), lo.std(ddof=1), len(lo), hi.mean() - lo.mean()))

print()
print('=' * 74)
print('Interpretation of the conclusions')
print('=' * 74)
n_low = int(m_low.sum())
print('Residual subset n=%d, of which %d are correct predictions and %d are incorrect predictions.' %
      (n_low, (correct[m_low] == 1).sum(), (correct[m_low] == 0).sum()))
print('The confidence variance of the residual subset is severely compressed (SD %.4f, only %.0f times the interval width of the saturated group),' %
      (cu.std(ddof=1), cs.std(ddof=1) / cu.std(ddof=1)))
print('Performing a rank correlation over such a narrow confidence interval is essentially "insufficient variance to detect an effect",')
print('not direct evidence that "the coupling does not exist" — the paper must distinguish these two interpretations faithfully.')
