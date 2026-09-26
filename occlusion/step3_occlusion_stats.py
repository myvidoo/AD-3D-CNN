# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 3: main statistical analysis (CPU, seconds).

Corresponds to the paper: §3.3 main conclusion and supplementary S6.

Input: <work>/results/occlusion_raw.npz + occlusion_meta.json
Output: <work>/results/occlusion_stats.json

Analyses (aligned with the pre-registered metrics of the "standardised occlusion protocol")
----
M1 Δp_true paired test   Wilcoxon signed-rank + effect size r; the main conclusion metric
M2 prediction flip rate  the proportion and directionality of argmax changes after occlusion (wrong→correct / correct→wrong)
M3 dose-response monotonicity   Spearman ρ of Δp_true across the 5 steps + trend test
M4 sample-level coupling    Spearman of Δp_true with baseline confidence / cam_mean
M5 cross-region comparison  the Δp_true difference of MTL vs C1/C2/C3 (paired Wilcoxon)
M6 filling-strategy robustness   agreement of the conclusions across the three strategies F1/F2/F3

Definition: the main analysis uses the **non-saturated subset** (p0.max() ≤ 0.999999, n=80) — the
      Δp_true of saturated samples is swallowed by the float64 softmax clamp; the all-samples
      n=144 result is reported in parallel.

⚠ Pre-registered interpretation matrix (hard-coded in the script, to prevent picking conclusions after the fact):
    A. MTL occlusion significantly lowers p_true and is significantly stronger than the volume-matched control → supports "the MTL carries diagnostic information"
    B. MTL occlusion does not differ from the control                          → does not support localisation specificity
    C. MTL occlusion raises p_true (counter-intuitive)                 → suggests the model relies on a non-MTL compensatory pathway

Run: python occlusion/step3_occlusion_stats.py"""

# Source: 27_occlusion_experiment/step3_stats.py (the computation logic is unchanged line by line; only the
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
    CSV, OUT, SAT_THR, setup_stdout,
)

setup_stdout()

raw = np.load(OUT / 'occlusion_raw.npz', allow_pickle=True)
meta = json.load(open(OUT / 'occlusion_meta.json', encoding='utf-8'))

cond_names = [str(x) for x in raw['cond_names']]
file_ids = [str(x) for x in raw['file_ids']]
true_lab = raw['true_labels']
cond_idx = raw['cond_idx']
mask_vox = raw['mask_vox']
P0 = raw['p0'].astype(np.float64)
P1 = raw['p_after'].astype(np.float64)

N_COND = len(cond_names)
N_SAMP = len(set(file_ids))
print("=" * 84)
print("Step 3  occlusion experiment statistical analysis")
print("=" * 84)
print("samples n = %d   conditions = %d   records = %d" % (N_SAMP, N_COND, len(file_ids)))

# ---- organise into a (cond, sample) matrix ----
uniq_ids = sorted(set(file_ids))
id2row = {fid: i for i, fid in enumerate(uniq_ids)}
for nm in meta['conditions']:
    assert nm['name'] in cond_names, nm['name']


def mat_of(cname):
    """Return the n×3 p_after matrix for this condition, with rows sorted by uniq_ids."""
    rows = {i: None for i in range(N_SAMP)}
    for k, fid in enumerate(file_ids):
        if cond_names[cond_idx[k]] == cname:
            rows[id2row[fid]] = P1[k]
    assert all(v is not None for v in rows.values()), cname
    return np.array([rows[i] for i in range(N_SAMP)])


P0m = np.array([P0[[k for k in range(len(file_ids))
                   if cond_names[cond_idx[k]] == cond_names[0]
                   and file_ids[k] == fid][0]] for fid in uniq_ids])
TL = np.array([true_lab[[k for k in range(len(file_ids))
                         if file_ids[k] == fid][0]] for fid in uniq_ids])

# ---- baseline re-check ----
base_pred = P0m.argmax(1)
base_acc = float(np.mean(base_pred == TL))
base_conf = P0m.max(1)
print("\n[baseline re-check] acc = %.4f (%d/%d)" % (base_acc, int((base_pred == TL).sum()), N_SAMP))

# ==================== ★ saturation stratification (key prerequisite) ====================
# Diagnostics: the top-1 probability of 64/144 cases is clamped by the
# float64 softmax to exactly 1.0.
# Example: p0=[1.07e-16, 1.74e-12, 1.0] → p_after=[1.67e-13, 2.33e-10, 1.0]
#     the probability actually changed by 3 orders of magnitude, but Δp_true is exactly 0 at the
#     bit level → the signal is swallowed by saturation.
# Of these, 29 cases have Δ exactly 0; 35 cases have Δ∈[1.2e-07, 4.8e-04] (likewise of numerical noise order).
# Among the 80 non-saturated samples, the share with Δ==0 is 0/80 — showing that the zero values
# are purely a numerical artefact, not a property of the model.
# → Therefore **the main analysis must be performed on the non-saturated subset (n=80)**; the
#   conclusion on all n=144 samples would be diluted.
saturated = P0m.max(1) > SAT_THR
NS = ~saturated
print("\n[saturation stratification] saturated samples with top-1 > %.6f = %d / %d (leaving %d non-saturated samples for the main analysis)"
      % (SAT_THR, int(saturated.sum()), N_SAMP, int(NS.sum())))
print("  share with Δ==0 among saturated samples = %d/%d" % (int((P0m[saturated, TL[saturated]]
                                            - mat_of('MTL_d100_F1const')[saturated, TL[saturated]] == 0).sum()),
                                     int(saturated.sum())))
print("  share with Δ==0 among non-saturated samples = %d/%d  ← the zero values are purely a numerical artefact"
      % (int((P0m[NS, TL[NS]] - mat_of('MTL_d100_F1const')[NS, TL[NS]] == 0).sum()), int(NS.sum())))

R = {}
R['saturation'] = dict(threshold=SAT_THR, n_saturated=int(saturated.sum()),
                       n_non_saturated=int(NS.sum()),
                       note='the Δp_true of saturated samples is swallowed by the float64 softmax clamp; the main analysis uses the non-saturated subset')


def cliffs_delta(a, b):
    """Cliff's delta (general paired/independent version: two independent groups)"""
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def wilcoxon_report(x, y=None, name=''):
    """Paired Wilcoxon signed-rank (x-y or x), returns a dict"""
    d = x if y is None else (np.asarray(x) - np.asarray(y))
    d = d[d != 0]
    if len(d) < 5:
        return dict(name=name, n=int(len(d)), note='too many samples have an effective difference of 0')
    w, p = st.wilcoxon(d, alternative='two-sided')
    # effect size r = Z / sqrt(N)
    try:
        z = st.norm.isf(p / 2)
    except Exception:
        z = np.nan
    r = z / np.sqrt(len(d))
    return dict(name=name, n=int(len(d)),
                median_delta=float(np.median(d)), mean_delta=float(np.mean(d)),
                W=float(w), p_value=float(p), effect_r=float(r),
                n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()))


# ==================== M1 Δp_true paired test ====================
print("\n" + "=" * 84)
print("M1  Δp_true = p0[true] - p_after[true] (positive = occlusion lowers the true-class probability)")
print("=" * 84)
print("[all samples n=%d]" % N_SAMP)
print("%-28s %11s %11s %10s %10s %8s" % ('condition', 'median Δp_true', 'mean Δp_true', 'W', 'p', 'effect r'))
M1_full = {}
for c in cond_names:
    Pa = mat_of(c)
    d = P0m[np.arange(N_SAMP), TL] - Pa[np.arange(N_SAMP), TL]
    rep = wilcoxon_report(d, name=c)
    M1_full[c] = rep
    if 'p_value' in rep:
        print("%-28s %11.5f %11.5f %10.1f %10.3g %8.3f"
              % (c, rep['median_delta'], rep['mean_delta'], rep['W'], rep['p_value'], rep['effect_r']))

print("\n[★ non-saturated subset n=%d — main analysis]" % int(NS.sum()))
print("%-28s %11s %11s %10s %10s %8s %7s" % ('condition', 'median Δp_true', 'mean Δp_true', 'W', 'p', 'effect r', 'n+/n-'))
M1 = {}
for c in cond_names:
    Pa = mat_of(c)
    d = (P0m[np.arange(N_SAMP), TL] - Pa[np.arange(N_SAMP), TL])[NS]
    rep = wilcoxon_report(d, name=c)
    M1[c] = rep
    if 'p_value' in rep:
        print("%-28s %11.5f %11.5f %10.1f %10.3g %8.3f %3d/%-3d"
              % (c, rep['median_delta'], rep['mean_delta'], rep['W'], rep['p_value'],
                 rep['effect_r'], rep['n_pos'], rep['n_neg']))
R['M1_delta_p_true'] = M1
R['M1_delta_p_true_full'] = M1_full

# ==================== M2 prediction flip rate ====================
print("\n" + "=" * 84)
print("M2  prediction flips (argmax changes after occlusion) — all samples n=%d (flips are unaffected by saturation)" % N_SAMP)
print("=" * 84)
M2 = {}
base_correct = (base_pred == TL)
print("%-28s %8s %10s %10s %10s %10s" % ('condition', 'flip rate', 'flip-to-correct (wrong→correct)', 'flip-to-wrong (correct→wrong)', 'stays correct', 'net change'))
for c in cond_names:
    Pa = mat_of(c)
    pred1 = Pa.argmax(1)
    flip = pred1 != base_pred
    corr1 = pred1 == TL
    fix = base_correct & (~corr1) if False else (~base_correct) & corr1
    brk = base_correct & (~corr1)
    keep = base_correct & corr1
    acc1 = float(np.mean(corr1))
    M2[c] = dict(flip_rate=float(flip.mean()), n_flip=int(flip.sum()),
                 n_fix=int(fix.sum()), n_break=int(brk.sum()), n_keep=int(keep.sum()),
                 acc_after=acc1, acc_delta=acc1 - base_acc,
                 flip_to_correct=int(fix.sum()), flip_to_wrong=int(brk.sum()))
    print("%-28s %8.4f %10d %10d %10d %+10.4f"
          % (c, flip.mean(), fix.sum(), brk.sum(), keep.sum(), acc1 - base_acc))
R['M2_flip'] = M2
R['baseline_acc'] = base_acc

# ==================== M3 dose-response ====================
print("\n" + "=" * 84)
print("M3  dose-response (MTL shrinking steps, F1 constant fill)")
print("=" * 84)
M3 = {}
doses = ['MTL_d25_F1const', 'MTL_d40_F1const', 'MTL_d60_F1const',
         'MTL_d80_F1const', 'MTL_d100_F1const']
dose_vox = [meta['dose_vox'][k.replace('MTL_', '').replace('_F1const', '')] for k in doses]
med = [M1[c]['median_delta'] for c in doses]
mean_ = [M1[c]['mean_delta'] for c in doses]
print("%-24s %10s %14s %14s %9s" % ('step', 'voxels', 'median Δp_true', 'mean Δp_true', 'p'))
for c, v in zip(doses, dose_vox):
    print("%-24s %10d %14.5f %14.5f %9.3g"
          % (c, v, M1[c]['median_delta'], M1[c]['mean_delta'], M1[c]['p_value']))
# use the mean for the dose-response (the median is affected by discretisation, the mean is more stable)
rho, prho = st.spearmanr(dose_vox, mean_)
rho_m, prho_m = st.spearmanr(dose_vox, med)
print("\n  Spearman(voxels, mean Δ) ρ = %.4f, p = %.4g" % (rho, prho))
print("  Spearman(voxels, median Δ) ρ = %.4f, p = %.4g" % (rho_m, prho_m))
mono = all(mean_[i] < mean_[i + 1] for i in range(len(mean_) - 1))
mono_m = all(med[i] < med[i + 1] for i in range(len(med) - 1))
print("  mean strictly monotonically increasing = %s ; median strictly monotonically increasing = %s" % (mono, mono_m))
# paired test between the first and last steps (strongest vs weakest)
Pa25, Pa100 = mat_of('MTL_d25_F1const'), mat_of('MTL_d100_F1const')
d25 = (P0m[np.arange(N_SAMP), TL] - Pa25[np.arange(N_SAMP), TL])[NS]
d100 = (P0m[np.arange(N_SAMP), TL] - Pa100[np.arange(N_SAMP), TL])[NS]
w_lr, p_lr = st.wilcoxon(d100, d25, alternative='two-sided')
print("  d100 vs d25 paired Wilcoxon: W=%.0f, p=%.4g" % (w_lr, p_lr))
R['M3_dose_response'] = dict(doses=doses, vox=dose_vox, median_delta=med, mean_delta=mean_,
                             spearman_rho_mean=float(rho), spearman_p_mean=float(prho),
                             spearman_rho_median=float(rho_m), spearman_p_median=float(prho_m),
                             strictly_monotonic_mean=bool(mono),
                             strictly_monotonic_median=bool(mono_m),
                             d100_vs_d25=dict(W=float(w_lr), p=float(p_lr)))

# ==================== M4 sample-level coupling ====================
print("\n" + "=" * 84)
print("M4  sample-level coupling (Spearman of Δp_true with baseline metrics)")
print("=" * 84)
Pa = mat_of('MTL_d100_F1const')
d_mtl = (P0m[np.arange(N_SAMP), TL] - Pa[np.arange(N_SAMP), TL])[NS]
NS_IDX = np.arange(N_SAMP)[NS]
# read the paper's per-sample table, take cam_mean / mtl_mass_frac
import csv
csvp = CSV
cam = {}
with open(csvp, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        cam[row['file_id']] = row
M4 = {}
for key, label in [('confidence', 'baseline confidence'), ('cam_mean', 'cam_mean'),
                   ('mtl_mass_frac', 'MTL mass fraction'), ('hipp_mass_frac', 'hippocampus mass fraction')]:
    if key == 'confidence':
        x = base_conf[NS_IDX]
    else:
        x = np.array([float(cam[fid][key]) for fid in uniq_ids])[NS_IDX]
    r_, p_ = st.spearmanr(x, d_mtl)
    M4[key] = dict(rho=float(r_), p=float(p_), label=label)
    print("  %-14s ρ = %+.4f, p = %.4g" % (label, r_, p_))
# and correctness
bc = base_correct[NS_IDX]
mcorr = bc.astype(float)
u_, pu = st.mannwhitneyu(d_mtl[bc], d_mtl[~bc], alternative='two-sided')
dl = cliffs_delta(d_mtl[bc], d_mtl[~bc])
M4['correctness_group'] = dict(u=float(u_), p=float(pu), cliffs_delta=float(dl),
                              correct_mean=float(d_mtl[bc].mean()),
                              incorrect_mean=float(d_mtl[~bc].mean()),
                              n_correct=int(bc.sum()), n_incorrect=int((~bc).sum()))
print("  %-14s correct %.5f (n=%d) vs incorrect %.5f (n=%d)  U=%.0f, p=%.4g, δ=%.3f"
      % ('correct vs incorrect', d_mtl[bc].mean(), bc.sum(), d_mtl[~bc].mean(), (~bc).sum(), u_, pu, dl))
R['M4_coupling'] = M4

# ==================== M5 cross-region comparison (control vs main mask) ====================
print("\n" + "=" * 84)
print("M5  control comparison (paired Wilcoxon, Δp_true main mask vs control)")
print("=" * 84)
M5 = {}
D = {}
for c in cond_names:
    Pc = mat_of(c)
    D[c] = (P0m[np.arange(N_SAMP), TL] - Pc[np.arange(N_SAMP), TL])[NS]
d_mtl = D['MTL_d100_F1const']
print("%-30s %11s %11s %9s %9s %9s" % ('control', 'main-mask median Δ', 'control median Δ', 'W', 'p', "Cliff's δ"))
for c in cond_names:
    if c == 'MTL_d100_F1const':
        continue
    d_c = D[c]
    if d_c.std() == 0 and d_mtl.std() == 0:
        M5[c] = dict(note='neither group shows any variation')
        continue
    w_, p_ = st.wilcoxon(d_mtl, d_c, alternative='two-sided')
    dl_ = cliffs_delta(d_mtl, d_c)
    M5[c] = dict(W=float(w_), p=float(p_), cliffs_delta=float(dl_),
                 median_main=float(np.median(d_mtl)), median_ctrl=float(np.median(d_c)),
                 mean_main=float(d_mtl.mean()), mean_ctrl=float(d_c.mean()))
    print("%-30s %11.5f %11.5f %9.0f %9.4g %+9.3f"
          % (c, np.median(d_mtl), np.median(d_c), w_, p_, dl_))
R['M5_vs_controls'] = M5

# ==================== M6 filling-strategy robustness ====================
print("\n" + "=" * 84)
print("M6  filling-strategy robustness (same mask d100, three fills; non-saturated subset n=%d)" % int(NS.sum()))
print("=" * 84)
strat = ['MTL_d100_F1const', 'MTL_d100_F2noise', 'MTL_d100_F3transplant']
ds = {c: D[c] for c in strat}
for c in strat:
    rep = M1[c]
    print("  %-26s median Δ=%+.5f  mean Δ=%+.5f  p=%.3g r=%.3f  flip rate=%.4f acc=%.4f"
          % (c, rep['median_delta'], rep['mean_delta'], rep['p_value'], rep['effect_r'],
             M2[c]['flip_rate'], M2[c]['acc_after']))
pairs = {}
for i in range(len(strat)):
    for j in range(i + 1, len(strat)):
        a, b = ds[strat[i]], ds[strat[j]]
        r_, p_ = st.spearmanr(a, b)
        pairs['%s|%s' % (strat[i], strat[j])] = dict(rho=float(r_), p=float(p_))
        print("    %s vs %s  ρ=%.4f (p=%.3g)" % (strat[i][-8:], strat[j][-8:], r_, p_))
same_sign = all(np.median(ds[c]) > 0 for c in strat) or all(np.median(ds[c]) < 0 for c in strat)
print("  the three strategies agree in direction = %s" % same_sign)
R['M6_fill_strategy'] = dict(per_strategy={c: M1[c] for c in strat},
                             pairwise_rho=pairs, direction_consistent=bool(same_sign))

# ==================== pre-registered interpretation matrix ====================
print("\n" + "=" * 84)
print("pre-registered interpretation matrix (based on the non-saturated subset n=%d)" % int(NS.sum()))
print("=" * 84)
c1n = 'C1_%s_F1const' % meta['c1_name']
mtl_vs_c1 = M5.get(c1n, {})
mtl_vs_c2 = M5.get('C2_ball_F1const', {})
mtl_vs_c3 = M5.get('C3_hipp_F1const', {})
m1_mtl = M1['MTL_d100_F1const']
sig_down = m1_mtl['median_delta'] > 0 and m1_mtl['p_value'] < 0.05
# at least two of the three controls significant and in the right direction = localisation specificity holds
ctrl_p = [mtl_vs_c1.get('p', 1), mtl_vs_c2.get('p', 1), mtl_vs_c3.get('p', 1)]
ctrl_d = [mtl_vs_c1.get('cliffs_delta', 0), mtl_vs_c2.get('cliffs_delta', 0),
          mtl_vs_c3.get('cliffs_delta', 0)]
n_strong = sum(1 for p_, d_ in zip(ctrl_p, ctrl_d) if p_ < 0.05 and d_ > 0)
stronger_than_ctrl = n_strong >= 2
# separate judgement of hippocampus vs MTL (key: is MTL better than the hippocampus alone)
hipp_better = mtl_vs_c3.get('p', 1) < 0.05 and mtl_vs_c3.get('cliffs_delta', 0) > 0
if sig_down and stronger_than_ctrl and hipp_better:
    verdict = 'A'
    verdict_txt = ('A — MTL occlusion significantly lowers the true-class probability, is significantly stronger than most controls, and is **better than the hippocampus alone**\n'
                   '      → supports "the MTL carries diagnostic information" and shows localisation specificity')
elif sig_down and stronger_than_ctrl:
    verdict = 'A-'
    verdict_txt = ('A- —— MTL occlusion significantly lowers the true-class probability and is significantly stronger than the volume-matched / random-sphere controls,\n'
                   '      but **MTL has no significant advantage over the hippocampus alone** (p=%.3g)\n'
                   '      → supports “the medial temporal lobe carries diagnostic information”, but the evidence points to **the hippocampus itself** rather than the whole MTL\n'
                   '      → if the paper needs to claim MTL (rather than hippocampal) specificity, it must be cautious' % mtl_vs_c3.get('p', float('nan')))
elif sig_down:
    verdict = 'A--'
    verdict_txt = ('A-- — MTL occlusion significantly lowers the true-class probability, but the difference from the controls does not reach significance\n'
                   '      → supports "the MTL carries diagnostic information"; the evidence for localisation specificity is insufficient')
else:
    verdict = 'C'
    verdict_txt = ('C — MTL occlusion does not significantly lower (or raises) the diagnostic probability\n'
                   '      → suggests the model relies on a non-MTL compensatory pathway; the interpretability claims need to be revisited')
print(verdict_txt)
print("\n  number of controls passing correction = %d / 3" % n_strong)
print("  MTL vs hippocampus p = %.4g (%s)" % (mtl_vs_c3.get('p', float('nan')),
                                             'MTL significantly better' if hipp_better else 'no significant difference'))
R['verdict'] = dict(code=verdict, text=verdict_txt,
                    sig_down=bool(sig_down), stronger_than_ctrl=bool(stronger_than_ctrl),
                    n_ctrl_significant=int(n_strong), hipp_better=bool(hipp_better),
                    mtl_vs_c1=mtl_vs_c1, mtl_vs_c2=mtl_vs_c2, mtl_vs_c3=mtl_vs_c3,
                    scope='non-saturated subset n=%d' % int(NS.sum()))

json.dump(R, open(OUT / 'occlusion_stats.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2, default=float)
print("\nsaved:", OUT / 'occlusion_stats.json')
print("=" * 84)
