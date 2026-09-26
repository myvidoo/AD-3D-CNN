# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 4b: rerun consistency -- the same set of statistics on two data sources (CPU, seconds).

Corresponds to the paper: supplementary S6 item (1).

Background: the step2b full rerun (occlusion_raw_rerun.npz) and the existing occlusion_raw.npz
      differ by ≤ 1.3×10⁻³ at the probability level (step4a has confirmed that the rank structure
      and the discrete decisions are fully preserved).
This script recomputes **every statistic on which the submitted conclusions depend** on both data
sources, compares them side by side item by item, and separately checks "whether any significance
conclusion flips".

It also: checks the donor identity sample by sample between the genuinely written donor_map.json and the rng reconstruction.

Input: occlusion_raw.npz / occlusion_raw_rerun.npz / donor_map.json + the Grad-CAM table
Output: <work>/results/rerun_consistency.json (redirecting stdout gives step4b_consistency.log)

Run: python occlusion/step4b_consistency_recompute.py > step4b_consistency.log"""

# Source: 27_occlusion_experiment/step4b_recompute.py (the computation logic is unchanged line by line; only the
# "environment bootstrap layer" was changed to be provided uniformly by occlusion_config; see
# occlusion/README.md for the refactoring notes).

import json
import sys
import itertools
import csv
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

MAIN = 'MTL_d100_F1const'
F3 = 'MTL_d100_F3transplant'
CLS = ['CN', 'MCI', 'AD']
DOSE = ['MTL_d25_F1const', 'MTL_d40_F1const', 'MTL_d60_F1const',
        'MTL_d80_F1const', 'MTL_d100_F1const']          # voxel counts strictly increasing
DOSE_IDX = [4, 3, 2, 1, 0]                               # indices within COND_ORDER
CTRLS = ['C1_cort_23_F1const', 'C2_ball_F1const', 'C3_hipp_F1const']

# ------------------------------------------------------------------ the CAM table
cam = {}
with open(CSV, encoding='utf-8-sig') as fh:
    for r in csv.DictReader(fh):
        cam[r['file_id']] = float(r['mtl_mass_frac'])


# ------------------------------------------------------------------ utilities
def ent(p):
    p = np.clip(np.asarray(p, float), 1e-300, 1.0)
    return -(p * np.log(p)).sum(-1)


def eff(d):
    """The full effect-size set for Δp_true (bounded rank-biserial; Cohen d_z; Rosenthal r retired and retained for reference)"""
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n < 2:
        return dict(n=n, mean=float('nan'), median=float('nan'), W=float('nan'),
                    p=float('nan'), r_rb=float('nan'), d_z=float('nan'), r_ros=float('nan'))
    w, p = st.wilcoxon(d, alternative='two-sided')
    z = float(st.norm.isf(p / 2))
    rk = st.rankdata(np.abs(d))
    Tp, Tn = float(rk[d > 0].sum()), float(rk[d < 0].sum())
    return dict(n=n, mean=float(d.mean()), median=float(np.median(d)),
                W=float(w), p=float(p),
                r_rb=(Tp - Tn) / (Tp + Tn) if (Tp + Tn) else float('nan'),
                d_z=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) else float('nan'),
                r_ros=z / np.sqrt(n))


def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def partial_spearman(x, y, z):
    rx, ry, rz = st.rankdata(x), st.rankdata(y), st.rankdata(z)
    rxy, rxz, ryz = (np.corrcoef(rx, ry)[0, 1], np.corrcoef(rx, rz)[0, 1],
                     np.corrcoef(ry, rz)[0, 1])
    pr = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    n = len(x)
    t = pr * np.sqrt((n - 3) / max(1 - pr ** 2, 1e-12))
    return float(pr), float(2 * st.t.sf(abs(t), n - 3)), float(rxz), float(ryz)


def exact_perm_spearman(x, y):
    """Exact two-sided permutation p over all 5-point permutations (120)"""
    obs = abs(float(st.spearmanr(x, y)[0]))
    vals = [abs(float(st.spearmanr(list(p), y)[0])) for p in itertools.permutations(x)]
    vals = np.asarray(vals)
    return obs, float((vals >= obs - 1e-12).mean()), len(vals)


# ------------------------------------------------------------------ loading
def load(fname):
    raw = np.load(OUT / fname, allow_pickle=True)
    cn = [str(x) for x in raw['cond_names']]
    fi = [str(x) for x in raw['file_ids']]
    tl_row = np.asarray(raw['true_labels'])
    ci = np.asarray(raw['cond_idx'])
    MV = np.asarray(raw['mask_vox'])
    K = len(cn)
    P0, P1 = raw['p0'].astype(np.float64), raw['p_after'].astype(np.float64)
    metas, seen = [], set()
    for f in fi:
        if f not in seen:
            seen.add(f); metas.append(f)
    N, i2r = len(metas), {f: i for i, f in enumerate(metas)}
    first = {}
    for k, f in enumerate(fi):
        if f not in first:
            first[f] = k
    P0m = np.array([P0[first[f]] for f in metas])
    TL = np.array([int(tl_row[first[f]]) for f in metas])

    def mat(c):
        rows = [None] * N
        for k, f in enumerate(fi):
            if cn[ci[k]] == c:
                rows[i2r[f]] = P1[k]
        return np.array(rows)

    IDX = np.arange(N)
    SAT = P0m.max(1) > SAT_THR
    D = {c: P0m[IDX, TL] - mat(c)[IDX, TL] for c in cn}
    dose_vox = [int(MV[DOSE_IDX[j]]) for j in range(5)]      # first sample block (identical across samples)
    for f in metas:                                          # consistency self-check over all sample blocks
        b = first[f]
        if [int(MV[b + DOSE_IDX[j]]) for j in range(5)] != dose_vox:
            raise SystemExit("the dose-step voxel counts are inconsistent across samples, aborting")
    return dict(name=fname, conds=cn, metas=metas, i2r=i2r, N=N, P0m=P0m, TL=TL,
                SAT=SAT, NS=~SAT, mat=mat, D=D, dose_vox=dose_vox)


ds_old = load('occlusion_raw.npz')
ds_new = load('occlusion_raw_rerun.npz')
assert ds_old['metas'] == ds_new['metas'], "the two data sources have an inconsistent sample order"

print("=" * 100)
print("Step 4b  rerun consistency: the same set of statistics × two data sources")
print("=" * 100)
print("sample order consistent ✅ | dose-step voxel counts %s" % ds_old['dose_vox'])
print("non-saturated n: OLD %d / NEW %d" % (ds_old['NS'].sum(), ds_new['NS'].sum()))

# ------------------------------------------------------------- donor-identity check
donor_real = json.load(open(OUT / 'donor_map.json', encoding='utf-8'))
donor_real = {k: int(v['donor_label']) for k, v in donor_real.items()}
LBL = {f: int(ds_old['TL'][ds_old['i2r'][f]]) for f in ds_old['metas']}
by_label = {}
for i, f in enumerate(ds_old['metas']):
    by_label.setdefault(LBL[f], []).append(i)
rng = np.random.default_rng(42)
COND_META = [('d100', 'const'), ('d80', 'const'), ('d60', 'const'), ('d40', 'const'),
             ('d25', 'const'), ('d100', 'noise'), ('d100', 'transplant'),
             ('c1', 'const'), ('ball', 'const'), ('hipp', 'const')]
NV = int(ds_old['dose_vox'][-1])
donor_recon = {}
for f in ds_old['metas']:
    for _, fl in COND_META:
        if fl == 'noise':
            rng.normal(0.0, 1.0, size=NV)
        elif fl == 'transplant':
            pool = [j for lb, js in by_label.items() if lb != LBL[f] for j in js]
            donor_recon[f] = LBL[ds_old['metas'][pool[rng.integers(0, len(pool))]]]

print()
print("--- donor-identity check (genuinely written vs rng reconstruction) ---")
assert set(donor_real) == set(donor_recon), "the recipient sets are inconsistent"
mism = [f for f in donor_real if donor_real[f] != donor_recon[f]]
print("  recipients genuinely written = %d | recipients from the rng reconstruction = %d | per-sample mismatches = %d"
      % (len(donor_real), len(donor_recon), len(mism)))
cnt = {}
for f, d in donor_real.items():
    cnt['%s<-%s' % (CLS[LBL[f]], CLS[d])] = cnt.get('%s<-%s' % (CLS[LBL[f]], CLS[d]), 0) + 1
print("  true donor distribution: %s" % cnt)
DONOR = donor_real            # ★ always use the identity actually written to disk
DONOR_SRC = 'donor_map.json (actually written to disk)'


# ------------------------------------------------------------------ computation
def compute(ds):
    P0m, TL, NS, SAT, mat, D = (ds['P0m'], ds['TL'], ds['NS'], ds['SAT'],
                                ds['mat'], ds['D'])
    N = ds['N']
    R = {'scope': dict(n_all=N, n_sat=int(SAT.sum()), n_nonsat=int(NS.sum()),
                       dose_vox=list(ds['dose_vox']))}
    dM = D[MAIN]
    pred0, pred1 = P0m.argmax(1), mat(MAIN).argmax(1)

    R['S1_main_all'] = eff(dM)
    R['S1_main_nonsat'] = eff(dM[NS])
    for ci, cn in enumerate(CLS):
        R['S2_class_' + cn] = eff(dM[(TL == ci) & NS])

    dm_dose = [float(D[c][NS].mean()) for c in DOSE]
    med_dose = [float(np.median(D[c][NS])) for c in DOSE]
    rho, p_asym = st.spearmanr(ds['dose_vox'], dm_dose)
    obs, p_ex, nper = exact_perm_spearman(ds['dose_vox'], dm_dose)
    rho_m, p_asm = st.spearmanr(ds['dose_vox'], med_dose)
    obs_m, p_mex, _ = exact_perm_spearman(ds['dose_vox'], med_dose)
    adj = [float(D[DOSE[j + 1]][NS].mean() - D[DOSE[j]][NS].mean()) for j in range(4)]
    adj_p = []
    for j in range(4):
        x, y = D[DOSE[j + 1]][NS], D[DOSE[j]][NS]
        diff = x - y
        adj_p.append(float(st.wilcoxon(diff[diff != 0]).pvalue) if np.any(diff != 0)
                     else float('nan'))
    R['S3_dose'] = {'mean_' + DOSE[j][4:7]: dm_dose[j] for j in range(5)}
    R['S3_dose'].update({'med_' + DOSE[j][4:7]: med_dose[j] for j in range(5)})
    R['S3_dose'].update(dict(
        rho=float(rho), p_asym=float(p_asym), p_exact=float(p_ex), n_perm=int(nper),
        rho_median=float(rho_m), p_asym_median=float(p_asm), p_exact_median=float(p_mex),
        **{'step_%d_%d_mean' % (j, j + 1): adj[j] for j in range(4)},
        **{'step_%d_%d_p' % (j, j + 1): adj_p[j] for j in range(4)}))

    for cc in CTRLS:
        diff = dM[NS] - D[cc][NS]
        w, p = st.wilcoxon(diff[diff != 0])
        rk = st.rankdata(np.abs(diff[diff != 0]))
        Tp, Tn = float(rk[diff[diff != 0] > 0].sum()), float(rk[diff[diff != 0] < 0].sum())
        R['S4_vs_' + cc] = dict(mean_diff=float(diff.mean()), mean_ctrl=float(D[cc][NS].mean()),
                                p=float(p), r_rb=(Tp - Tn) / (Tp + Tn))

    ok, bad = NS & (pred0 == TL), NS & (pred0 != TL)
    R['S5_correct'] = eff(dM[ok])
    R['S5_wrong'] = eff(dM[bad])
    U, pu = st.mannwhitneyu(dM[ok], dM[bad], alternative='two-sided')
    R['S5_group'] = dict(U=float(U), p=float(pu), cliff=float(cliffs(dM[ok], dM[bad])))
    R['S5_net_acc'] = dict(n_ok=(pred0 == TL).sum(), n_fixed=int((bad & (pred1 == TL)).sum()),
                           n_broken=int((ok & (pred1 != TL)).sum()))
    for ci, cn in enumerate(CLS):
        m = (TL == ci) & NS
        R['S5_class_' + cn] = dict(n_ok=int((m & (pred0 == TL)).sum()),
                                   n_bad=int((m & (pred0 != TL)).sum()),
                                   **{('ok_' + k): v for k, v in eff(dM[m & (pred0 == TL)]).items()},
                                   **{('bad_' + k): v for k, v in eff(dM[m & (pred0 != TL)]).items()})

    H0, H1 = ent(P0m), ent(mat(MAIN))
    dH = H1 - H0
    fl = NS & (pred0 != pred1)
    fx = NS & (pred0 != TL) & (pred1 == TL)
    bk = NS & (pred0 == TL) & (pred1 != TL)
    w, p = st.wilcoxon(dH[NS][dH[NS] != 0])
    R['S6_entropy_main'] = dict(mean_ns=float(dH[NS].mean()), p=float(p),
                                n_down=int((dH[NS] < 0).sum()), n_up=int((dH[NS] > 0).sum()),
                                mean_all=float(dH.mean()))
    R['S6_flip'] = dict(n=int(fl.sum()), mean=float(dH[fl].mean()) if fl.sum() else float('nan'))
    R['S6_fix'] = dict(n=int(fx.sum()), mean=float(dH[fx].mean()) if fx.sum() else float('nan'))
    R['S6_brk'] = dict(n=int(bk.sum()), mean=float(dH[bk].mean()) if bk.sum() else float('nan'))
    for ci, cn in enumerate(CLS):
        m = (TL == ci) & NS
        R['S6_class_' + cn] = dict(n=int(m.sum()), mean=float(dH[m].mean()))
    a2c = np.where((TL == 2) & (pred0 == 2) & (pred1 == 0))[0]
    R['S6_ad2cn'] = {'case%d' % j: dict(n=int(a2c.size), file_id=ds['metas'][i],
                                        H0=float(H0[i]), H1=float(H1[i]), dH=float(dH[i]),
                                        conf0=float(P0m[i].max()), conf1=float(mat(MAIN)[i].max()))
                     for j, i in enumerate(a2c)}

    MF = np.array([cam.get(f, np.nan) for f in ds['metas']])
    R['S7_cam_match'] = dict(n_hit=int(np.isfinite(MF).sum()), n_total=N)
    for ci, cn in list(enumerate(CLS)) + [(None, 'ALL')]:
        m = (NS & np.isfinite(MF)) if ci is None else ((TL == ci) & NS & np.isfinite(MF))
        if m.sum() < 4:
            continue
        rho, pv = st.spearmanr(MF[m], dM[m])
        key = 'S7_coupling_ALL' if ci is None else 'S7_coupling_' + cn
        R[key] = dict(n=int(m.sum()), rho=float(rho), p=float(pv))
        if cn == 'AD':
            pr, pp, rxz, ryz = partial_spearman(MF[m], dM[m], P0m[m, TL[m]] - 0.5)
            R['S7_AD_ceiling'] = dict(partial_rho=pr, partial_p=pp,
                                      rho_mass_p0=rxz, rho_dp_p0=ryz)
    mmci = (TL == 1) & NS
    dmr = (mat(MAIN) - P0m)[mmci]
    pdCN = float(st.wilcoxon(dmr[:, 0][dmr[:, 0] != 0]).pvalue) if np.any(dmr[:, 0] != 0) else float('nan')
    pdAD = float(st.wilcoxon(dmr[:, 2][dmr[:, 2] != 0]).pvalue) if np.any(dmr[:, 2] != 0) else float('nan')
    R['S7_mci_redist'] = dict(n=int(mmci.sum()), mean_dp=float(dM[mmci].mean()),
                              sd=float(dM[mmci].std(ddof=1)),
                              n_pos=int((dM[mmci] > 0).sum()), n_neg=int((dM[mmci] < 0).sum()),
                              dCN=float(dmr[:, 0].mean()), dMCI=float(dmr[:, 1].mean()),
                              dAD=float(dmr[:, 2].mean()), p_dCN=pdCN, p_dAD=pdAD)

    for ci, cn in enumerate(CLS):
        m = (TL == ci) & NS
        d1, d3 = D[MAIN][m], D[F3][m]
        dp = d3 - d1
        R['S8_receptor_' + cn] = dict(
            n=int(m.sum()), F1_mean=float(d1.mean()), F3_mean=float(d3.mean()),
            F1_p=float(st.wilcoxon(d1[d1 != 0]).pvalue) if np.any(d1 != 0) else float('nan'),
            F3_p=float(st.wilcoxon(d3[d3 != 0]).pvalue) if np.any(d3 != 0) else float('nan'),
            paired_p=float(st.wilcoxon(dp[dp != 0]).pvalue) if np.any(dp != 0) else float('nan'))
    AX0 = P0m[:, 1] + 2.0 * P0m[:, 2]

    def axis_of(c):
        Pc = mat(c)
        return Pc[:, 1] + 2.0 * Pc[:, 2]

    da3 = axis_of(F3) - AX0
    R['S8_axis_baseline'] = {cn: float(AX0[TL == ci].mean()) for ci, cn in enumerate(CLS)}
    R['S8_daxis'] = {c: dict(all=float((axis_of(c) - AX0)[NS].mean()),
                             **{cn: float((axis_of(c) - AX0)[(TL == ci) & NS].mean())
                                for ci, cn in enumerate(CLS)})
                     for c in (MAIN, 'MTL_d100_F2noise', F3)}

    def sev_stat(assign):
        vals = [np.sign(assign[f] - LBL[f]) * da3[ds['i2r'][f]]
                for f in ds['metas'] if NS[ds['i2r'][f]] and assign[f] != LBL[f]]
        return float(np.mean(vals)), len(vals)

    stat3, n3 = sev_stat(DONOR)
    sv = [np.sign(DONOR[f] - LBL[f]) * da3[ds['i2r'][f]]
          for f in ds['metas'] if NS[ds['i2r'][f]] and DONOR[f] != LBL[f]]
    w3, p3 = st.wilcoxon([v for v in sv if v != 0])
    da1 = axis_of(MAIN) - AX0
    stat1 = float(np.mean([-da1[ds['i2r'][f]] for f in ds['metas'] if NS[ds['i2r'][f]]]))
    v1 = [-da1[ds['i2r'][f]] for f in ds['metas'] if NS[ds['i2r'][f]] and da1[ds['i2r'][f]] != 0]
    w1, p1 = st.wilcoxon(v1)
    rngp = np.random.default_rng(20260916)          # ★ fixed, so both data sources share the same permutation sequence
    nperm = 2000
    null = np.empty(nperm)
    for t in range(nperm):
        a = {}
        for f in ds['metas']:
            other = [g for g in ds['metas'] if LBL[g] != LBL[f]]
            a[f] = LBL[other[rngp.integers(0, len(other))]]
        null[t] = sev_stat(a)[0]
    R['S8_joint'] = dict(stat_f1=stat1, p_f1=float(p1), stat_f3=stat3, p_f3=float(p3),
                         n=n3, perm_p=float((null >= stat3).mean()),
                         perm_mean=float(null.mean()), perm_sd=float(null.std()),
                         perm_p95=float(np.percentile(null, 95)), n_perm=nperm)

    for ci, cn in enumerate(CLS):
        ids = [f for f in ds['metas'] if LBL[f] == ci]
        for dl in range(3):
            sel = [ds['i2r'][f] for f in ids if DONOR[f] == dl and NS[ds['i2r'][f]]]
            if len(sel) < 3:
                continue
            R['S8_cell_r%s_d%s' % (cn, CLS[dl])] = dict(
                n=len(sel), dCN=float((mat(F3) - P0m)[sel, 0].mean()),
                dMCI=float((mat(F3) - P0m)[sel, 1].mean()),
                dAD=float((mat(F3) - P0m)[sel, 2].mean()), dp=float(D[F3][sel].mean()))
    return R


R_OLD = compute(ds_old)
R_NEW = compute(ds_new)


# ------------------------------------------------------------- comparison and verdict
def leaves(a, b, path=''):
    out = []
    if isinstance(a, dict):
        for k in a:
            out += leaves(a[k], b.get(k) if isinstance(b, dict) else None, path + '/' + k)
    elif isinstance(a, (int, float, np.integer, np.floating)) and not isinstance(a, bool):
        out.append((path, float(a), (float(b) if isinstance(b, (int, float, np.integer,
                      np.floating)) else float('nan'))))
    return out


LV = leaves(R_OLD, R_NEW)
print()
print("=" * 100)
print("Part one  significance/verdict flip checks (determining whether any conclusion changes)")
print("=" * 100)
CHECKS = [
    ('saturated stratum n_sat (must = 64)', '/scope/n_sat', 'count'),
    ('non-saturated n (must = 80)', '/scope/n_nonsat', 'count'),
    ('main-definition non-saturated Wilcoxon p (should be < 0.05)', '/S1_main_nonsat/p', 'sig'),
    ('main-definition non-saturated rank-biserial (should be in 0.2–0.5)', '/S1_main_nonsat/r_rb', 'val'),
    ('AD stratum non-saturated p (should be < 0.05)', '/S2_class_AD/p', 'sig'),
    ('AD stratum rank-biserial (should = 1.000)', '/S2_class_AD/r_rb', 'val'),
    ('MCI stratum p (should be ≫ 0.05, negative)', '/S2_class_MCI/p', 'ns'),
    ('CN stratum p (should be ≫ 0.05, negative)', '/S2_class_CN/p', 'ns'),
    ('dose-response ρ (mean Δ, about 0.70)', '/S3_dose/rho', 'val'),
    ('dose-response exact permutation p (mean Δ, should be > 0.05)', '/S3_dose/p_exact', 'ns'),
    ('dose-response ρ (median Δ, about 0.90)', '/S3_dose/rho_median', 'val'),
    ('dose-response exact permutation p (median Δ, should be > 0.05)', '/S3_dose/p_exact_median', 'ns'),
    ('MTL vs C1 upper-bound p', '/S4_vs_C1_cort_23_F1const/p', 'val'),
    ('MTL vs C2 upper-bound p', '/S4_vs_C2_ball_F1const/p', 'val'),
    ('MTL vs C3 upper-bound p', '/S4_vs_C3_hipp_F1const/p', 'val'),
    ('correct group mean Δp (about +0.071)', '/S5_correct/mean', 'val'),
    ('incorrect group mean Δp (about −0.164)', '/S5_wrong/mean', 'val'),
    ('between correct/incorrect groups p (should be < 0.001)', '/S5_group/p', 'sig'),
    ('net accuracy change (must = 0)', '/S5_net_acc/n_fixed', 'count'),
    ('net accuracy: number broken (must = 5)', '/S5_net_acc/n_broken', 'count'),
    ('entropy: non-saturated p (about 0.078, not significant)', '/S6_entropy_main/p', 'ns'),
    ('entropy: flipped-sample ΔH (about −0.158)', '/S6_flip/mean', 'val'),
    ('E3 MCI ρ (about −0.019, negative)', '/S7_coupling_MCI/rho', 'val'),
    ('E3 MCI p (should be ≫ 0.05)', '/S7_coupling_MCI/p', 'ns'),
    ('E3 AD ρ (about −0.448)', '/S7_coupling_AD/rho', 'val'),
    ('E3 AD p (should be < 0.05)', '/S7_coupling_AD/p', 'sig'),
    ('E3 AD partial correlation p (should be > 0.05, ceiling)', '/S7_AD_ceiling/partial_p', 'ns'),
    ('MCI redistribution ΔAD (should be < 0, significant)', '/S7_mci_redist/dAD', 'val'),
    ('E2 recipient CN: F3−F1 paired p (should be < 0.05)', '/S8_receptor_CN/paired_p', 'sig'),
    ('E2 joint statistic F3 (about 0.124)', '/S8_joint/stat_f3', 'val'),
    ('E2 permutation p (should be < 0.05)', '/S8_joint/perm_p', 'sig'),
    ('disease-axis baseline CN (about 0.177)', '/S8_axis_baseline/CN', 'val'),
]
D = {p: (a, b) for p, a, b in LV}
bad_flip = []
for label, path, kind in CHECKS:
    if path not in D:
        print("  %-44s ★ path missing" % label); continue
    a, b = D[path]
    if kind == 'count':
        ok = (a == b)
        note = "count-type values must be strictly equal"
    elif kind == 'sig':
        ok = ((a < 0.05) == (b < 0.05))
        note = "significance direction"
    elif kind == 'ns':
        ok = ((a > 0.05) == (b > 0.05))
        note = "non-significance direction"
    else:
        rel = abs(a - b) / (abs(a) + 1e-300)
        ok = (rel <= 0.05) or (abs(a - b) <= 1e-9)
        note = "relative tolerance 5%% (conclusion magnitude/sign preserved)"
    if not ok:
        bad_flip.append(label)
    print("  %-44s OLD %+12.6g  NEW %+12.6g  |Δ| %-10.3g %s  (%s)"
          % (label, a, b, abs(a - b), "✅" if ok else "★flipped", note))

IDS_O = [c['file_id'] for c in R_OLD['S6_ad2cn'].values()]
IDS_N = [c['file_id'] for c in R_NEW['S6_ad2cn'].values()]
same_set = (IDS_O == IDS_N)
if not same_set:
    bad_flip.append('AD→CN flipped-sample set')
print("  %-44s %s  (OLD n=%d / NEW n=%d)"
      % ('the AD→CN flipped-sample set is identical case by case (by file_id)',
         "✅" if same_set else "★mismatch", len(IDS_O), len(IDS_N)))

print()
print("=" * 100)
print("Part two  count/integer-valued quantities that disagree (if any, these are genuine divergences)")
print("=" * 100)
int_keys = [k for k in D if any(t in k for t in ('/n', '_n', 'n_', 'dose_vox'))
            and not any(t in k for t in ('mean', 'rho', 'stat', 'perm_p', 'perm_mean'))]
n_bad = 0
for k in int_keys:
    a, b = D[k]
    if abs(a - b) > 1e-9:
        n_bad += 1
        print("  ★ %-52s OLD %.6g  NEW %.6g" % (k, a, b))
print("  %d count/integer-valued quantities in total, %d disagreeing" % (len(int_keys), n_bad))

print()
print("=" * 100)
print("Part three  Top 25 largest deviations among continuous quantities (relative deviation)")
print("=" * 100)
cont = [(k, a, b) for k, (a, b) in D.items()
        if abs(a) > 1e-12 and not any(t in k for t in ('/n', '_n', 'n_', 'dose_vox'))]
cont.sort(key=lambda t: -abs(t[1] - t[2]) / (abs(t[1]) + 1e-300))
for k, a, b in cont[:25]:
    rel = abs(a - b) / (abs(a) + 1e-300)
    print("  %-52s OLD %+14.8g  NEW %+14.8g  rel %-10.3g" % (k, a, b, rel))
rel_all = [abs(a - b) / (abs(a) + 1e-300) for _, a, b in cont]
print("  %d continuous quantities in total: rel median %.3g | rel p95 %.3g | rel max %.3g"
      % (len(cont), float(np.median(rel_all)), float(np.percentile(rel_all, 95)),
         float(np.max(rel_all))))

json.dump({'donor_source': DONOR_SRC,
           'donor_recon_mismatch_n': len(mism),
           'donor_mismatch_ids': mism,
           'old': R_OLD, 'new': R_NEW,
           'flip_checks_failed': bad_flip},
          open(OUT / 'rerun_consistency.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2, default=float)

print()
print("=" * 100)
print("overall verdict: %d flipped check(s) %s" % (len(bad_flip), bad_flip if bad_flip else ""))
print("saved: results/rerun_consistency.json")
print("=" * 100)
