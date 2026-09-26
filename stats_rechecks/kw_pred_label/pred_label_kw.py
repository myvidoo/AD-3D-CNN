# -*- coding: utf-8 -*-
"""
T4 improved analysis: rerun KW by [predicted label], with a true × predicted two-way decomposition
Purpose:
  1. Reproduce the baseline T4 (grouped by true label, H=11.14, p=0.0038)
  2. Rerun KW grouped by predicted label — isolating the "prediction-side" signal
  3. Two-way decomposition: MTL mass fraction ~ true label + predicted label
  4. A cross-tabulation showing how entangled the two factors are
Data source: 05_explainability_GradCAM_.../result_data/per_sample_results_final.csv (144 per-sample records)
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
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))
true_lab = np.array([int(r['true_label']) for r in rows])
pred_lab = np.array([int(r['pred_label']) for r in rows])
correct = np.array([int(r['correct']) for r in rows])
mtl = np.array([float(r['mtl_mass_frac']) for r in rows])
hipp = np.array([float(r['hipp_mass_frac']) for r in rows])
conf = np.array([float(r['confidence']) for r in rows])
NAMES = {0: 'CN', 1: 'MCI', 2: 'AD'}

# ============ exact t-distribution CDF (continued fraction of the incomplete beta function) ============
def betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d; h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1.0) < eps: break
    return h


def betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1 - x) / b


def chi2_sf(x, df):
    """Chi-square right-tail probability P(X>x) = the upper tail of I_{1}(df/2, x/2) = 1 - I(x/2, df/2, ...)"""
    # P(X>x) = 1 - P(df/2, x/2), where P is the regularised lower incomplete gamma -> use the beta relation
    # chi2 sf = betai(df/2, 0.5, ...) is not direct; use the series: P(X>x)=Q(df/2, x/2)
    # use Gammainc via series / continued fraction
    return _gammainc_upper(df / 2.0, x / 2.0)


def _gammainc_upper(a, x):
    """Upper incomplete gamma Q(a,x)"""
    if x < 0 or a <= 0: return float('nan')
    if x == 0: return 1.0
    if x < a + 1:
        # series expansion for P(a,x), then 1-P
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-16: break
        p = s * exp(-x + a * log(x) - lgamma(a))
        return max(0.0, min(1.0, 1.0 - p))
    else:
        # continued fraction for Q(a,x)
        b, c = x + 1 - a, 1e300
        d = 1.0 / b; h = d
        for i in range(1, 1000):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            if abs(d) < 1e-300: d = 1e-300
            c = b + an / c
            if abs(c) < 1e-300: c = 1e-300
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < 1e-16: break
        return h * exp(-x + a * log(x) - lgamma(a))


def rank(a):
    n = len(a)
    order = a.argsort()
    r = np.empty(n, dtype=float)
    r[order] = np.arange(1, n + 1)
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    for k in np.where(cnt > 1)[0]:
        r[inv == k] = r[inv == k].mean()
    return r


def kruskal_wallis(*groups):
    """Returns (H, p, df, n, tie_correction_applied)"""
    allv = np.concatenate(groups)
    n = len(allv)
    r = rank(allv)
    idx = 0
    H = 0.0
    for g in groups:
        ng = len(g)
        Rg = r[idx:idx + ng].sum()
        H += Rg ** 2 / ng
        idx += ng
    H = 12.0 / (n * (n + 1)) * H - 3 * (n + 1)
    # ties correction
    vals, cnt = np.unique(allv, return_counts=True)
    ties = cnt[cnt > 1]
    if len(ties) > 0:
        T = sum(t ** 3 - t for t in ties)
        C = 1 - T / (n ** 3 - n)
        H_corr = H / C
    else:
        H_corr = H
    df = len(groups) - 1
    p = chi2_sf(H_corr, df)
    return H, H_corr, p, df, n, len(ties) > 0


def mwu(x, y):
    n1, n2 = len(x), len(y)
    allv = np.concatenate([x, y])
    r = rank(allv)
    U = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sd = sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (U - mu) / sd
    p = erfc(abs(z) / sqrt(2))
    return float(U), float(p), float(2 * U / (n1 * n2) - 1)


def eta2_epsilon2(H, n, k):
    eps2 = H / (n - 1)
    eta2 = (H - k + 1) / (n - k)
    return eps2, eta2


print('=' * 84)
print('Step 0  Data overview')
print('=' * 84)
print('n =', len(rows))
tbl = np.zeros((3, 3), dtype=int)
for t, p_ in zip(true_lab, pred_lab):
    tbl[t, p_] += 1
print('True (rows) × predicted (columns) cross-tabulation:')
print('%-8s %8s %8s %8s %8s' % ('', 'pred CN', 'pred MCI', 'pred AD', 'Total'))
for i in range(3):
    print('%-8s %8d %8d %8d %8d' % ('true ' + NAMES[i], tbl[i, 0], tbl[i, 1], tbl[i, 2], tbl[i].sum()))
print('%-8s %8d %8d %8d %8d' % ('Total', tbl[:, 0].sum(), tbl[:, 1].sum(), tbl[:, 2].sum(), tbl.sum()))

# =======================================================
print()
print('=' * 84)
print('Step 1  Reproduce the baseline T4 (grouped by true label)')
print('=' * 84)
g_true = [mtl[true_lab == k] for k in range(3)]
H, Hc, p, df, n, tied = kruskal_wallis(*g_true)
eps2, eta2 = eta2_epsilon2(Hc, n, 3)
print('H(uncorrected) = %.4f   H(ties-corrected) = %.4f   p = %.4g   df = %d' % (H, Hc, p, df))
print('Reported in the paper: H=11.14, p=0.0038, η²=0.078')
print('This run: eps² = H/(n-1) = %.4f' % eps2)
print('This run: η² = (H-k+1)/(n-k) = %.4f' % eta2)
for k in range(3):
    print('  %-4s n=%2d  MTL = %.4f%% ± %.4f' % (NAMES[k], len(g_true[k]),
          100 * g_true[k].mean(), 100 * g_true[k].std(ddof=1)))
chk = '✓ reproduction matches' if abs(Hc - 11.14) < 0.05 and abs(p - 0.0038) < 5e-4 else '✗ needs checking'
print('Verdict:', chk)

# =======================================================
print()
print('=' * 84)
print('Step 2  [CORE] rerun KW grouped by predicted label')
print('=' * 84)
g_pred = [mtl[pred_lab == k] for k in range(3)]
H2, Hc2, p2, df2, n2, tied2 = kruskal_wallis(*g_pred)
eps2_2, eta2_2 = eta2_epsilon2(Hc2, n2, 3)
print('H(uncorrected) = %.4f   H(ties-corrected) = %.4f   p = %.4g' % (H2, Hc2, p2))
print('eps² = %.4f   η² = %.4f' % (eps2_2, eta2_2))
for k in range(3):
    g = g_pred[k]
    print('  pred %-4s n=%2d  MTL = %.4f%% ± %.4f' % (NAMES[k], len(g),
          100 * g.mean(), 100 * g.std(ddof=1)))
print()
print('Comparison:')
print('  By true label H=%.2f, p=%.4g, eps²=%.4f' % (Hc, p, eps2))
print('  By predicted label H=%.2f, p=%.4g, eps²=%.4f' % (Hc2, p2, eps2_2))

# pairwise post-hoc (Dunn, Bonferroni)
def dunn(groups, names):
    """Dunn post-hoc test + Bonferroni; returns the list of pairwise results"""
    allv = np.concatenate(groups)
    n = len(allv)
    r = rank(allv)
    # compute the ties-corrected variance term
    vals, cnt = np.unique(allv, return_counts=True)
    ties = cnt[cnt > 1]
    T = sum(t ** 3 - t for t in ties)
    sig2 = n * (n + 1) / 12.0
    if T > 0:
        sig2 = (n * (n + 1) / 12.0) - T / (12.0 * (n - 1))
    idx, means = 0, []
    for g in groups:
        means.append(r[idx:idx + len(g)].mean())
        idx += len(g)
    out = []
    npairs = 0
    res = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            npairs += 1
            se = sqrt(sig2 * (1.0 / len(groups[i]) + 1.0 / len(groups[j])))
            z = abs(means[i] - means[j]) / se
            p_raw = erfc(z / sqrt(2))
            res.append((names[i], names[j], z, p_raw))
    return res, npairs


print()
print('--- Dunn post-hoc grouped by predicted label (Bonferroni correction ×3) ---')
res, npairs = dunn(g_pred, [NAMES[k] for k in range(3)])
for a, b, z, pr in res:
    print('  pred %-4s vs pred %-4s : z=%.3f  p_raw=%.4g  p_adj=%.4g  %s' %
          (a, b, z, pr, min(pr * npairs, 1.0), 'significant' if pr * npairs < 0.05 else 'n.s.'))

print()
print('--- Dunn post-hoc grouped by true label (Bonferroni correction ×3, baseline reproduction) ---')
res_t, _ = dunn(g_true, [NAMES[k] for k in range(3)])
for a, b, z, pr in res_t:
    print('  true %-4s vs true %-4s : z=%.3f  p_raw=%.4g  p_adj=%.4g  %s' %
          (a, b, z, pr, min(pr * npairs, 1.0), 'significant' if pr * npairs < 0.05 else 'n.s.'))
print('  (The paper reports: only MCI>CN is significant, corrected p=0.0026; this example uses an absolute-value test, whereas the paper used a directional test)')

data = dict(
    baseline_true=dict(H=float(Hc), p=float(p), eps2=float(eps2), eta2=float(eta2),
                       means=[float(100 * g.mean()) for g in g_true],
                       sds=[float(100 * g.std(ddof=1)) for g in g_true],
                       ns=[int(len(g)) for g in g_true]),
    predicted=dict(H=float(Hc2), p=float(p2), eps2=float(eps2_2), eta2=float(eta2_2),
                   means=[float(100 * g.mean()) for g in g_pred],
                   sds=[float(100 * g.std(ddof=1)) for g in g_pred],
                   ns=[int(len(g)) for g in g_pred],
                   dunn=[dict(a=a, b=b, z=float(z), p_raw=float(pr),
                              p_adj=float(min(pr * npairs, 1.0))) for a, b, z, pr in res]),
    crosstab=tbl.tolist(),
)
with open(os.path.join(OUT, 'pred_label_kw_results.json'), 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print()
print('Saved:', os.path.join(OUT, 'pred_label_kw_results.json'))
