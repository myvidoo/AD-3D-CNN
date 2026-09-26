# -*- coding: utf-8 -*-
"""
Cross-check: recompute the Spearman p value with a higher-precision method, confirming that the normal approximation introduces no material bias
Method: the exact t-distribution CDF (implemented via the continued fraction of the incomplete beta function) instead of the normal approximation
"""
import os, csv, json
import numpy as np

base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SRC = os.path.join(base, 'reference_results', 'gradcam_v3',
                   'per_sample_results_final.csv')
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


def betacf(a, b, x, itmax=300, eps=3e-16):
    """Incomplete beta function via the Lentz continued-fraction method"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    """Regularised incomplete beta function I_x(a,b)"""
    from math import lgamma, exp, log
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1 - x) / b


def t_sf(t, df):
    """t-distribution right-tail probability P(T>t)"""
    x = df / (df + t * t)
    p = 0.5 * betai(df / 2.0, 0.5, x)
    return p if t > 0 else 1.0 - p


def spearman_exact(x, y):
    """Returns (rho, p_two_sided, n), with p from the exact t-distribution CDF"""
    n = len(x)
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rc = max(min(rho, 0.9999999), -0.9999999)
    t = rc * np.sqrt((n - 2) / (1 - rc ** 2))
    p = 2.0 * t_sf(abs(t), n - 2)
    return rho, min(p, 1.0), n


def spearman_normal(x, y):
    from math import erfc, sqrt
    n = len(x)
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rc = max(min(rho, 0.9999999), -0.9999999)
    t = rc * sqrt((n - 2) / (1 - rc ** 2))
    return rho, float(erfc(abs(t) / sqrt(2))), n


print('=' * 82)
print('p-value precision cross-check: exact t-distribution CDF vs normal approximation')
print('=' * 82)
print('%-30s %5s %9s %16s %16s' % ('subset', 'n', 'rho', 'p (exact t)', 'p (normal approx.)'))
print('-' * 82)
cases = [
    ('all-sample baseline', np.ones(len(conf), bool)),
    ('excluding exactly saturated 1.0', conf < 1.0),
    ('excluding conf>0.99', conf <= 0.99),
    ('correct predictions only', correct == 1),
    ('incorrect predictions only', correct == 0),
]
for name, m in cases:
    if m.sum() < 4:
        print('%-30s %5d  insufficient samples' % (name, m.sum())); continue
    r1, p1, n1 = spearman_exact(conf[m], cam[m])
    r2, p2, _ = spearman_normal(conf[m], cam[m])
    print('%-30s %5d %9.4f %16.4g %16.4g' % (name, n1, r1, p1, p2))

print()
print('→ If the two columns agree to 4 decimal places, the normal approximation used in this report introduces no material bias.')
