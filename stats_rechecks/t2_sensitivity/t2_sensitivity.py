# -*- coding: utf-8 -*-
"""
T2 sensitivity analysis: recompute the confidence-CAM intensity Spearman correlation after excluding the confidence>0.99 subsample
Objectives:
  1. First reproduce the baseline ρ=0.800 (n=144, p<0.001) — a determinism check
  2. Recompute Spearman after excluding the confidence>0.99 subsample
  3. Report the number of samples removed, the change in ρ, the p value and the robustness interpretation
Data source: 05_explainability_GradCAM_.../result_data/per_sample_results_final.csv (n=144, authoritative archive)
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

# ---------- paths ----------
base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SRC = os.path.join(base, 'reference_results', 'gradcam_v3',
                   'per_sample_results_final.csv')
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# ---------- read the data ----------
rows = []
with open(SRC, encoding='utf-8-sig') as f:
    for d in csv.DictReader(f):
        rows.append(d)
print('Samples read:', len(rows))

conf = np.array([float(r['confidence']) for r in rows])
cam = np.array([float(r['cam_mean']) for r in rows])
correct = np.array([int(r['correct']) for r in rows])
true_lab = np.array([int(r['true_label']) for r in rows])
pred_lab = np.array([int(r['pred_label']) for r in rows])

# ---------- statistics functions (scipy preferred, manual fallback) ----------
try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
print('scipy available:', HAVE_SCIPY)


def spearman(x, y):
    """Returns (rho, p, n). Uses scipy when available, otherwise a manual t approximation."""
    n = len(x)
    if n < 4:
        return float('nan'), float('nan'), n
    if HAVE_SCIPY:
        rho, p = stats.spearmanr(x, y)
        return float(rho), float(p), n
    # manual: rank correlation = Pearson correlation on ranks
    def rank(a):
        order = a.argsort()
        r = np.empty(n, dtype=float)
        r[order] = np.arange(1, n + 1)
        # handle ties (average ranks)
        vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        for k in np.where(cnt > 1)[0]:
            m = inv == k
            r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rho = max(min(rho, 0.9999999), -0.9999999)
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    # two-sided p (t distribution approximated by the normal)
    from math import erfc
    p = float(erfc(abs(t) / np.sqrt(2)))
    return rho, p, n


def cliff_delta(x, y):
    """Cliff's delta (x vs y) via the rank-sum relation: delta = 2U/(n1*n2) - 1"""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float('nan')
    if HAVE_SCIPY:
        U, _ = stats.mannwhitneyu(x, y, alternative='two-sided')
    else:
        gt = sum((a > b) for a in x for b in y)
        lt = sum((a < b) for a in x for b in y)
        U = gt + 0.5 * sum((a == b) for a in x for b in y)
    return 2.0 * U / (n1 * n2) - 1.0


# =======================================================
# Step 1: reproduce the baseline
# =======================================================
print()
print('=' * 72)
print('Step 1  Reproduce the baseline (paper §3.3: rho=0.800, p<0.001, n=144)')
print('=' * 72)
rho0, p0, n0 = spearman(conf, cam)
print('n = %d' % n0)
print('Spearman rho = %.4f' % rho0)
print('p = %.3g' % p0)
chk = '✓ reproduction matches' if abs(rho0 - 0.800) < 0.002 else '✗ does not match 0.800 (difference %.4f)' % (rho0 - 0.800)
print(chk)

# =======================================================
# Step 2: goodness-of-fit description (degree of saturation)
# =======================================================
print()
print('=' * 72)
print('Step 2  Description of the degree of probability saturation')
print('=' * 72)
print('confidence description: mean %.4f  median %.4f  min %.4f  max %.4f' %
      (conf.mean(), np.median(conf), conf.min(), conf.max()))
for th in [0.99, 0.95, 0.90, 0.80]:
    m = conf > th
    print('  confidence > %.2f : %3d samples (%.1f%%)' % (th, m.sum(), 100 * m.mean()))
print('  confidence == 1.0 exactly saturated: %d samples (%.1f%%)' %
      ((conf >= 1.0).sum(), 100 * (conf >= 1.0).mean()))
print('  Accuracy among samples with peak >0.99: %.1f%% (%d/%d)' %
      (100 * correct[conf > 0.99].mean(), correct[conf > 0.99].sum(), (conf > 0.99).sum()))

# =======================================================
# Step 3: sensitivity analysis — exclusion by threshold
# =======================================================
print()
print('=' * 72)
print('Step 3  Sensitivity analysis: excluding the confidence>threshold subsample')
print('=' * 72)
print('%-12s %6s %10s %12s %10s' % ('threshold', 'n left', 'rho', 'p', 'Δrho'))
print('-' * 72)
results = []
for th in [1.00, 0.99, 0.95, 0.90, 0.80]:
    keep = conf <= th if th < 1.0 else conf < 1.0
    n_k = int(keep.sum())
    if n_k < 4:
        continue
    rho_k, p_k, _ = spearman(conf[keep], cam[keep])
    d_rho = rho_k - rho0
    results.append(dict(threshold=th, n=n_k, rho=rho_k, p=p_k, d_rho=d_rho))
    print('%-12s %6d %10.4f %12.3g %+10.4f' %
          ('%.2f' % th, n_k, rho_k, p_k, d_rho))

# primary criterion: the >0.99 exclusion (the one the paper is concerned with)
keep99 = conf <= 0.99
rho99, p99, n99 = spearman(conf[keep99], cam[keep99])
print()
print('>>> Main result (excluding confidence>0.99): n=%d, rho=%.4f, p=%.3g, Δrho=%+.4f' %
      (n99, rho99, p99, rho99 - rho0))

# =======================================================
# Step 4: robustness interpretation
# =======================================================
print()
print('=' * 72)
print('Step 4  Interpretation')
print('=' * 72)
same_sign = (rho99 > 0) and (rho0 > 0)
still_sig = p99 < 0.05
print('Direction consistent (both positive):', same_sign)
print('Still significant after exclusion (p<0.05):', still_sig, '(p=%.3g)' % p99)
if still_sig and same_sign:
    print('→ Conclusion: the confidence-CAM coupling in T2 still holds after excluding the probability-saturated samples,')
    print('  i.e. the correlation is not driven by extreme saturated probabilities. The sensitivity analysis supports the robustness of T2.')
else:
    print('→ Conclusion: after excluding the saturated samples the correlation is no longer significant, so T2 must be stated more cautiously in the paper.')

# Supplementary: whether it still holds after stratifying by correct/incorrect (extra robustness)
print()
print('Group-wise robustness (supplementary):')
for lab, m in [('correct predictions', correct == 1), ('incorrect predictions', correct == 0)]:
    if m.sum() >= 4:
        r_, p_, n_ = spearman(conf[m], cam[m])
        print('  %s (n=%d): rho=%.4f, p=%.3g' % (lab, n_, r_, p_))

# ---------- output ----------
# `source` is recorded as a package-relative path on purpose. Writing the resolved absolute path here
# would make the JSON (a) machine-specific and therefore not byte-reproducible on any other machine,
# and (b) leak the running user's local directory layout into a distributed artifact.
SOURCE_REL = os.path.relpath(SRC, base).replace(os.sep, "/")
res = dict(
    source=SOURCE_REL,
    baseline=dict(n=n0, rho=rho0, p=p0),
    saturation=dict(
        gt099=int((conf > 0.99).sum()),
        eq1=int((conf >= 1.0).sum()),
        mean_conf=float(conf.mean()),
    ),
    sensitivity=results,
    primary=dict(threshold=0.99, n=n99, rho=rho99, p=p99, d_rho=rho99 - rho0,
                 still_significant=bool(p99 < 0.05)),
    scipy_available=HAVE_SCIPY,
)
with open(os.path.join(OUT, 't2_sensitivity_results.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print()
print('Saved:', os.path.join(OUT, 't2_sensitivity_results.json'))
