# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 4a: diagnosis of the full-rerun difference (CPU, seconds; reads existing artifacts only).

Corresponds to the paper: the source of evidence for supplementary S6 item (1) "tiered reproducibility verification".

Background (step2b full-rerun result): p0 max|Δ| ≈ 9.4×10⁻⁴, p_after max|Δ| ≈ 1.3×10⁻³,
mask_vox exactly identical, argmax mismatches 0/1440.
This script determines the **nature and scope** of that difference:
  A internal consistency of p0 (the 10 conditions of one sample should give the same p0; and whether it equals baseline_p0.json)
  B the sample-level / condition-level structure of the difference (is it concentrated in a few samples)
  C the relation between the difference and the baseline probability level (is it concentrated in the near-saturated floating-point sensitive region)
  D fidelity at the Δp_true level (old vs new: absolute difference, rank correlation, sign agreement rate)
  E stability of argmax / the discrete decision
  F stability of the saturation stratification ← determines whether all "non-saturated definitions" can be reused

Input: occlusion_raw.npz + occlusion_raw_rerun.npz + baseline_p0.json
Output: <work>/results/rerun_diag.json

Run: python occlusion/step4a_rerun_diag.py"""

# Source: 27_occlusion_experiment/step4a_rerun_diag.py (the computation logic is unchanged line by line; only the
# "environment bootstrap layer" was changed to be provided uniformly by occlusion_config; see
# occlusion/README.md for the refactoring notes).

import json
import sys
import numpy as np
import scipy.stats as st

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from occlusion_config import (  # noqa: E402
    OUT, SAT_THR, setup_stdout,
)

setup_stdout()

old = np.load(OUT / 'occlusion_raw.npz', allow_pickle=True)
new = np.load(OUT / 'occlusion_raw_rerun.npz', allow_pickle=True)

COND = [str(c) for c in old['cond_names']]
K = len(COND)
FI_ROW = [str(f) for f in old['file_ids']]      # row by row (1440)
TL_ROW = np.asarray(old['true_labels']).astype(int)
CI = np.asarray(old['cond_idx']).astype(int)
NROWS = len(CI)
R = {}

print("=" * 84)
print("Step 4a  diagnosis of the full-rerun difference")
print("=" * 84)

# the row order must be the flattened (sample × condition), otherwise reshape does not hold
if NROWS % K:
    raise SystemExit("the number of rows %d is not a multiple of the number of conditions %d, aborting" % (NROWS, K))
M = NROWS // K
expect_ci = np.tile(np.arange(K), M)
if not (CI == expect_ci).all():
    raise SystemExit("the row order is not the flattened (sample×condition), reshape does not hold, aborting")
CI_NEW = np.asarray(new['cond_idx']).astype(int)
FI_ROW_NEW = [str(f) for f in new['file_ids']]
if not (CI_NEW == expect_ci).all() or FI_ROW_NEW != FI_ROW:
    raise SystemExit("the rerun file has an inconsistent row/sample order, aborting")
if not (np.asarray(new['true_labels']).astype(int) == TL_ROW).all():
    raise SystemExit("the rerun file has inconsistent true labels, aborting")
# shrink to sample level
FI = [FI_ROW[i * K] for i in range(M)]
TL = TL_ROW[::K]
for i in range(M):
    blk = FI_ROW[i * K:(i + 1) * K]
    if len(set(blk)) != 1 or len(set(TL_ROW[i * K:(i + 1) * K].tolist())) != 1:
        raise SystemExit("within sample block %d the file_id/label is not unique, aborting" % i)

print("grid: %d samples × %d conditions = %d rows | condition names and row order: consistent"
      % (M, K, NROWS))

P0O = old['p0'].astype(np.float64).reshape(M, K, 3)
P0N = new['p0'].astype(np.float64).reshape(M, K, 3)
PAO = old['p_after'].astype(np.float64).reshape(M, K, 3)
PAN = new['p_after'].astype(np.float64).reshape(M, K, 3)

# ---------------------------------------------------------------- A p0 consistency
print()
print("--- A  internal consistency of p0 ---")
w_o = float(np.abs(P0O - P0O[:, :1, :]).max())
w_n = float(np.abs(P0N - P0N[:, :1, :]).max())
print("  maximum spread of p0 across the 10 conditions of one sample:  OLD %.3e | NEW %.3e" % (w_o, w_n))

bp = json.load(open(OUT / 'baseline_p0.json', encoding='utf-8'))
bp_map = {str(r['file_id']): np.array(r['p0'], dtype=np.float64) for r in bp}
seteq = (set(bp_map) == set(FI))
B0 = np.stack([bp_map[f] for f in FI]) if seteq else None
d_bo = float(np.abs(B0 - P0O[:, 0, :]).max()) if seteq else float('nan')
d_bn = float(np.abs(B0 - P0N[:, 0, :]).max()) if seteq else float('nan')
print("  p0 vs baseline_p0.json        :  OLD %.3e | NEW %.3e  (sample sets identical=%s)"
      % (d_bo, d_bn, seteq))
R['p0_within_sample_disp'] = dict(old=w_o, new=w_n)
R['p0_vs_baseline_old'] = d_bo
R['p0_vs_baseline_new'] = d_bn
R['baseline_sample_set_match'] = bool(seteq)

# ------------------------------------------------------------ B structure of the difference
print()
print("--- B  sample-level / condition-level structure of the difference ---")
d0_s = np.abs(P0O - P0N).max(axis=(1, 2))   # maximum difference per sample
da_s = np.abs(PAO - PAN).max(axis=(1, 2))
q = lambda x: dict(min=float(x.min()), p25=float(np.percentile(x, 25)),
                   median=float(np.median(x)), p75=float(np.percentile(x, 75)),
                   max=float(x.max()), mean=float(x.mean()))
R['p0_per_sample'] = q(d0_s)
R['p_after_per_sample'] = q(da_s)
for nm, x in (('p0', d0_s), ('p_after', da_s)):
    s = q(x)
    print("  %-8s per-sample max|Δ|: min %.3e | p25 %.3e | median %.3e | p75 %.3e | max %.3e"
          % (nm, s['min'], s['p25'], s['median'], s['p75'], s['max']))
worst = np.argsort(-d0_s)[:5]
R['p0_worst5'] = [dict(file_id=FI[i], true_label=int(TL[i]), max_diff=float(d0_s[i]))
                  for i in worst.tolist()]
print("  the 5 samples with the largest p0 difference:")
for i in worst:
    print("    %-28s true class=%d  max|Δp0|=%.3e" % (FI[i], TL[i], d0_s[i]))
print("  number of samples with a difference > 1e-06: p0 %d/%d | p_after %d/%d"
      % (int((d0_s > 1e-6).sum()), M, int((da_s > 1e-6).sum()), M))

per_cond = {}
print("  per-condition max|Δ| against the Δp_true mean:")
IDX = np.arange(M)
t3 = np.broadcast_to(TL[:, None, None], (M, K, 1))


def dtrue(P0, PA):
    """Δp_true = p0[true class] − p_after[true class]; positive = the true-class probability drops after occlusion"""
    x = np.take_along_axis(P0, t3, axis=2)[:, :, 0]
    y = np.take_along_axis(PA, t3, axis=2)[:, :, 0]
    return x - y


D_OLD = dtrue(P0O, PAO)
D_NEW = dtrue(P0N, PAN)
for j, c in enumerate(COND):
    rec = dict(max_abs_diff=float(np.abs(PAO[:, j, :] - PAN[:, j, :]).max()),
               d_old_mean=float(D_OLD[:, j].mean()),
               d_new_mean=float(D_NEW[:, j].mean()),
               d_max_abs_diff=float(np.abs(D_OLD[:, j] - D_NEW[:, j]).max()))
    per_cond[c] = rec
    print("    %-24s max|Δp_after|=%.3e  Δp_true mean %.6f → %.6f  (|ΔΔ|max %.3e)"
          % (c, rec['max_abs_diff'], rec['d_old_mean'], rec['d_new_mean'], rec['d_max_abs_diff']))
R['per_condition'] = per_cond

# ------------------------------------------------- C coupling of the difference to the probability level
print()
print("--- C  which probability interval the difference concentrates in ---")
pv = P0O.ravel()
dv = np.abs(P0O - P0N).ravel()
edges = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, SAT_THR, 1.0000001]
names = ['[0,0.01)', '[0.01,0.1)', '[0.1,0.5)', '[0.5,0.9)',
         '[0.9,0.99)', '[0.99,0.999)', '[0.999,0.999999]', '(0.999999,1]']
bins = np.digitize(pv, edges) - 1
R['diff_by_p0_bin'] = []
for b, nm in enumerate(names):
    s = (bins == b)
    if s.sum() == 0:
        continue
    rec = dict(bin=nm, n=int(s.sum()), mean=float(dv[s].mean()),
               median=float(np.median(dv[s])), max=float(dv[s].max()))
    R['diff_by_p0_bin'].append(rec)
    print("  p0∈%-18s n=%5d   mean|Δ|=%.3e  median|Δ|=%.3e  max|Δ|=%.3e"
          % (nm, s.sum(), rec['mean'], rec['median'], rec['max']))

# ------------------------------------------------------- D fidelity at the Δp_true level
print()
print("--- D  fidelity at the Δp_true level (old vs new) ---")
dd = np.abs(D_OLD - D_NEW)
rho_res = st.spearmanr(D_OLD.ravel(), D_NEW.ravel())
R['d_true'] = dict(max_abs_diff=float(dd.max()), mean=float(dd.mean()),
                   median=float(np.median(dd)), p99=float(np.percentile(dd, 99)),
                   spearman=float(rho_res[0]), spearman_p=float(rho_res[1]),
                   sign_agreement=float((np.sign(D_OLD) == np.sign(D_NEW)).mean()))
print("  |ΔΔp_true| : mean %.3e | median %.3e | p99 %.3e | max %.3e"
      % (dd.mean(), np.median(dd), np.percentile(dd, 99), dd.max()))
print("  Spearman(Δ old, Δ new) = %.6f  (p = %.3e)" % (rho_res[0], rho_res[1]))
print("  sign agreement rate = %.4f  (%d / %d)"
      % (R['d_true']['sign_agreement'], int((np.sign(D_OLD) == np.sign(D_NEW)).sum()), D_OLD.size))

# ----------------------------------------------- E discrete decision / argmax stability
print()
print("--- E  stability of argmax and the predicted labels ---")
mis_p0 = int((P0O.argmax(2) != P0N.argmax(2)).sum())
mis_pa = int((PAO.argmax(2) != PAN.argmax(2)).sum())
conf_p0 = int((P0O.argmax(1) != P0N.argmax(1)).sum())
conf_pa = int((PAO.argmax(1) != PAN.argmax(1)).sum())
R['argmax'] = dict(cell_p0=mis_p0, cell_p_after=mis_pa, total_cells=M * K,
                   sample_p0=conf_p0, sample_p_after=conf_pa)
print("  cell-level argmax mismatches: p0 %d/%d | p_after %d/%d" % (mis_p0, M * K, mis_pa, M * K))
print("  sample-level mismatch of the 10-condition prediction vector: p0 %d/%d | p_after %d/%d" % (conf_p0, M, conf_pa, M))

# ------------------------------------------------- F stability of the saturation stratification ★ key
print()
print("--- F  stability of the saturation stratification (SAT_THR = %.6f, based on the baseline p0 only) ---" % SAT_THR)
m0_o = P0O[:, 0, :].max(1)
m0_n = P0N[:, 0, :].max(1)
sat_o = m0_o > SAT_THR
sat_n = m0_n > SAT_THR
fl = np.where(sat_o != sat_n)[0]
R['saturation'] = dict(n_sat_old=int(sat_o.sum()), n_sat_new=int(sat_n.sum()),
                       n_nonsat_old=int((~sat_o).sum()), n_nonsat_new=int((~sat_n).sum()),
                       n_flip=int(fl.size),
                       flips=[dict(file_id=FI[i], max_p_old=float(m0_o[i]),
                                   max_p_new=float(m0_n[i])) for i in fl.tolist()])
print("  saturated n: OLD %d / NEW %d   →   non-saturated n: OLD %d / NEW %d"
      % (sat_o.sum(), sat_n.sum(), (~sat_o).sum(), (~sat_n).sum()))
print("  number of samples whose stratum flipped = %d" % fl.size)
for i in fl:
    print("    %-28s baseline top1: %.9f → %.9f" % (FI[i], m0_o[i], m0_n[i]))
near = ((m0_o > 0.99) & (~sat_o)).sum()
print("  samples with baseline top1 ∈ (0.99, %.6f] (the critical band): %d; their max|Δp0| = %.3e"
      % (SAT_THR, near, (d0_s[(m0_o > 0.99) & (~sat_o)].max() if near else float('nan'))))

with open(OUT / 'rerun_diag.json', 'w', encoding='utf-8') as f:
    json.dump(R, f, ensure_ascii=False, indent=2)

print()
print("=" * 84)
print("saved:", OUT / 'rerun_diag.json')
print("=" * 84)
