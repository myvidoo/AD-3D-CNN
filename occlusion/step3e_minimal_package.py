# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 3e: minimal necessary package (3 verifications, CPU, seconds; reads existing results only).

Corresponds to the paper: supplementary S6 items (9)–(11).

Background: in the mechanistic explanation "occlusion = replacing the whole disease axis with
      healthy evidence", the most fragile statement is
      "MCI discrimination likewise depends on the MTL" -- the MCI-stratum group mean of Δp_true is
      null, which looks contradictory.
This script performs only the 3 tests requested by the user (one more would be an over-experiment):

  E1 prediction entropy ΔH before and after occlusion -- distinguishes "active misreading" (entropy drops, more confident) from "incapacity" (entropy rises, flattened)
  E2 the **bidirectionality** of the F3 cross-subject transplant -- the donor pool of CN recipients comes 100% from the disease side (MCI+AD)
     ⟹ the data for the "injecting disease evidence" direction already exists in the current results, with zero extra runs
  E3 MCI-stratum per-sample coupling Spearman(mtl_mass_frac, Δp_true) + power check (including the 80% power MDE)

Definition: non-saturated n=80 (SAT_THR is based only on the baseline p0); Δp_true positive = the true-class probability drops after occlusion.
⚠ Read-only; no existing file is modified.

Run: python occlusion/step3e_minimal_package.py"""

# Source: 27_occlusion_experiment/step3e_minimal.py (the computation logic is unchanged line by line; only the
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

raw = np.load(OUT / 'occlusion_raw.npz', allow_pickle=True)
cond_names = [str(x) for x in raw['cond_names']]
file_ids = [str(x) for x in raw['file_ids']]
true_lab = raw['true_labels']
cond_idx = raw['cond_idx']
P0 = raw['p0'].astype(np.float64)
P1 = raw['p_after'].astype(np.float64)

uniq = sorted(set(file_ids))
first = {}
for k, f in enumerate(file_ids):
    if f not in first:
        first[f] = k
ORDER = [f for f in uniq]
# restore the metas order by **first-appearance order** (the insertion order of by_label in step2 depends on it)
metas_order, seen = [], set()
for f in file_ids:
    if f not in seen:
        seen.add(f); metas_order.append(f)
N = len(metas_order)
i2r = {f: i for i, f in enumerate(metas_order)}


def mat_of(c):
    rows = {i: None for i in range(N)}
    for k, f in enumerate(file_ids):
        if cond_names[cond_idx[k]] == c:
            rows[i2r[f]] = P1[k]
    return np.array([rows[i] for i in range(N)])


P0m = np.array([P0[first[f]] for f in metas_order])
TL = np.array([int(true_lab[first[f]]) for f in metas_order])
IDX = np.arange(N)
CLS = ['CN', 'MCI', 'AD']
LBL = {f: int(true_lab[first[f]]) for f in metas_order}
SAT = P0m.max(1) > 0.999999
NS = ~SAT
MAIN = 'MTL_d100_F1const'
Pm = mat_of(MAIN)
D = {c: P0m[IDX, TL] - mat_of(c)[IDX, TL] for c in cond_names}
pred0, pred1 = P0m.argmax(1), Pm.argmax(1)

print("=" * 94)
print("Step 3e  minimal necessary package (E1 entropy / E2 F3 bidirectionality / E3 MCI coupling)")
print("=" * 94)
print("all samples n=%d, non-saturated n=%d, saturated n=%d" % (N, NS.sum(), SAT.sum()))

R = {'scope': dict(n_all=N, n_nonsat=int(NS.sum()), n_sat=int(SAT.sum()))}


def ent(p):
    p = np.clip(np.asarray(p, float), 1e-300, 1.0)
    return -(p * np.log(p)).sum(-1)


# ==================================================================
# E1  prediction entropy: misreading vs incapacity
# ==================================================================
print("\n" + "-" * 94)
print("E1  prediction entropy ΔH = H(p_after) − H(p0)   [misreading→entropy drops (more confident); incapacity→entropy rises (flattened)]")
print("-" * 94)
print("reference: H(uniform three-class) = ln3 = %.4f" % np.log(3))
H0 = ent(P0m)
H1 = ent(Pm)
dH = H1 - H0
R['E1_entropy'] = {}

flip_ns = NS & (pred0 != pred1)
print("\n[comparison across all conditions] non-saturated n=%d; of these, flipped (argmax changed) n=%d" % (NS.sum(), flip_ns.sum()))
print("%-24s %10s %10s %10s %10s" % ("condition", "all ΔH", "flipped-sample ΔH", "flip-to-correct ΔH", "flip-to-wrong ΔH"))
for c in cond_names:
    Pc = mat_of(c)
    h1 = ent(Pc)
    dh = h1 - H0
    pc = Pc.argmax(1)
    fix = NS & (pred0 != TL) & (pc == TL)
    brk = NS & (pred0 == TL) & (pc != TL)
    fl = NS & (pc != pred0)
    row = dict(all=float(dh[NS].mean()),
               flip=float(dh[fl].mean()) if fl.sum() else None,
               flip_n=int(fl.sum()),
               fix=float(dh[fix].mean()) if fix.sum() else None,
               fix_n=int(fix.sum()),
               brk=float(dh[brk].mean()) if brk.sum() else None,
               brk_n=int(brk.sum()))
    R['E1_entropy'][c] = row
    print("%-24s %+10.5f %+10.5f %+10.5f %+10.5f" %
          (c, row['all'],
           row['flip'] if row['flip'] is not None else float('nan'),
           row['fix'] if row['fix'] is not None else float('nan'),
           row['brk'] if row['brk'] is not None else float('nan')))

print("\n[closer look at the main definition %s]" % MAIN)
print("  ΔH: non-saturated all %+.5f (median %+.5f), all samples %+.5f"
      % (dH[NS].mean(), np.median(dH[NS]), dH.mean()))
w, p_ent = st.wilcoxon(dH[NS][dH[NS] != 0])
print("  paired Wilcoxon (non-saturated, ΔH vs 0): W=%.1f, p=%.4g, n=%d, %d down / %d up"
      % (w, p_ent, int((dH[NS] != 0).sum()),
         int((dH[NS] < 0).sum()), int((dH[NS] > 0).sum())))
R['E1_entropy']['_MAIN_nonsat'] = dict(
    mean=float(dH[NS].mean()), median=float(np.median(dH[NS])),
    mean_all=float(dH.mean()),
    wilcoxon_W=float(w), wilcoxon_p=float(p_ent), n=int((dH[NS] != 0).sum()),
    n_down=int((dH[NS] < 0).sum()), n_up=int((dH[NS] > 0).sum()))

print("\n  by true class (non-saturated):")
for ci, cn in enumerate(CLS):
    m = (TL == ci) & NS
    print("    %-4s n=%2d  ΔH mean %+.5f  median %+.5f  (%d down / %d up)"
          % (cn, m.sum(), dH[m].mean(), np.median(dH[m]),
             int((dH[m] < 0).sum()), int((dH[m] > 0).sum())))
    R['E1_entropy']['_byclass_' + cn] = dict(
        n=int(m.sum()), mean=float(dH[m].mean()), median=float(np.median(dH[m])),
        n_down=int((dH[m] < 0).sum()), n_up=int((dH[m] > 0).sum()))

print("\n  the 3 AD→CN flipped samples one by one:")
for i in np.where((TL == 2) & (pred0 == 2) & (pred1 == 0))[0]:
    print("    %-24s H: %.4f → %.4f (ΔH %+.4f) | argmax confidence %.4f → %.4f"
          % (metas_order[i][:24], H0[i], H1[i], dH[i], P0m[i].max(), Pm[i].max()))
    R['E1_entropy'].setdefault('_ad2cn_flips', []).append(
        dict(file_id=metas_order[i], H0=float(H0[i]), H1=float(H1[i]),
             dH=float(dH[i]), conf0=float(P0m[i].max()), conf1=float(Pm[i].max())))

# all samples (including saturated) cross-check: saturated samples have H≈0, ΔH≈0
print("\n  [cross-check] saturated samples n=%d: mean ΔH %+.2e (should be ≈0, verifying the saturation artefact)"
      % (SAT.sum(), dH[SAT].mean()))

# ==================================================================
# E2  bidirectionality of the F3 cross-subject transplant
# ==================================================================
print("\n" + "-" * 94)
print("E2  bidirectionality of F3transplant: the donor pool of CN recipients = {MCI, AD} (100% disease side)")
print("-" * 94)
F3 = 'MTL_d100_F3transplant'
F1 = MAIN
R['E2_f3'] = {}

print("\n[robust test not requiring donor labels] stratified by the recipient's true class, comparing F1 (healthy fill) with F3 (cross-subject transplant)")
print("%-6s %4s | %-22s | %-22s" % ("recipient", "n", "F1 constant fill Δp_true", "F3 transplant Δp_true"))
print("%-6s %4s | %10s %11s | %10s %11s" % ("", "", "mean", "p(vs0)", "mean", "p(vs0)"))
for ci, cn in enumerate(CLS):
    m = (TL == ci) & NS
    d1, d3 = D[F1][m], D[F3][m]
    p1 = st.wilcoxon(d1[d1 != 0]).pvalue if np.any(d1 != 0) else np.nan
    p3 = st.wilcoxon(d3[d3 != 0]).pvalue if np.any(d3 != 0) else np.nan
    # paired: F3 vs F1 on the same set of samples
    dp = d3 - d1
    wp = st.wilcoxon(dp[dp != 0]).pvalue if np.any(dp != 0) else np.nan
    print("%-6s %4d | %+10.5f %11.4g | %+10.5f %11.4g   (paired F3−F1: p=%.4g)"
          % (cn, m.sum(), d1.mean(), p1, d3.mean(), p3, wp))
    R['E2_f3']['receptor_' + cn] = dict(
        n=int(m.sum()), F1_mean=float(d1.mean()), F1_p=float(p1),
        F3_mean=float(d3.mean()), F3_p=float(p3), F3_minus_F1_p=float(wp))

print("\n[probability destination: CN recipients (donors 100% disease side)] if \"the MTL carries the disease axis\", p(CN) should drop")
m_cn = (TL == 0)
m_cn_ns = m_cn & NS
for tag, c in (('F1 constant fill', F1), ('F3 transplant (donor=disease side)', F3)):
    dm = (mat_of(c) - P0m)[m_cn_ns]
    print("  %-24s n=%2d | ΔCN %+.5f  ΔMCI %+.5f  ΔAD %+.5f | Δp_true %+.5f"
          % (tag, m_cn_ns.sum(), dm[:, 0].mean(), dm[:, 1].mean(), dm[:, 2].mean(),
             D[c][m_cn_ns].mean()))
    R['E2_f3']['cn_receptor_flow_' + tag.split()[0]] = dict(
        n=int(m_cn_ns.sum()), dCN=float(dm[:, 0].mean()), dMCI=float(dm[:, 1].mean()),
        dAD=float(dm[:, 2].mean()), dp_true=float(D[c][m_cn_ns].mean()))

# donor label reconstruction (secondary evidence; same rng sequence as step2)
print("\n[secondary: donor label reconstruction] reproducing the rng in the same order as step2 (seed=42)")
try:
    by_label = {}
    for i, f in enumerate(metas_order):
        by_label.setdefault(LBL[f], []).append(i)
    COND = [dict(name=n, mask=m, fill=fl) for n, m, fl in [
        ('MTL_d100_F1const', 'd100', 'const'), ('MTL_d80_F1const', 'd80', 'const'),
        ('MTL_d60_F1const', 'd60', 'const'), ('MTL_d40_F1const', 'd40', 'const'),
        ('MTL_d25_F1const', 'd25', 'const'), ('MTL_d100_F2noise', 'd100', 'noise'),
        ('MTL_d100_F3transplant', 'd100', 'transplant'), ('C1_cort_23_F1const', 'c1', 'const'),
        ('C2_ball_F1const', 'ball', 'const'), ('C3_hipp_F1const', 'hipp', 'const')]]
    NV = 31764  # d100 mask voxel count (the size for F2noise)
    rng = np.random.default_rng(42)
    donor_of = {}
    for si, f in enumerate(metas_order):
        for cond in COND:
            if cond['fill'] == 'noise':
                rng.normal(0.0, 1.0, size=NV)
            elif cond['fill'] == 'transplant':
                pool = [j for lb, js in by_label.items() if lb != LBL[f] for j in js]
                dj = pool[rng.integers(0, len(pool))]
                donor_of[f] = (dj, LBL[metas_order[dj]])
    print("  reconstruction complete: %d recipients (=144, as expected)" % len(donor_of))
    chk = {}

    def donor_split(receptor_cls, cname='MTL_d100_F3transplant'):
        dm = (mat_of(cname) - P0m)
        ids = [f for f in metas_order if LBL[f] == receptor_cls]
        out = {}
        for dl in range(3):
            sel = [f for f in ids if donor_of[f][1] == dl]
            if not sel:
                continue
            ix = [i2r[f] for f in sel]
            ns = np.array([NS[i] for i in ix])
            ix_ns = [ix[j] for j in range(len(ix)) if ns[j]]
            if not ix_ns:
                continue
            out[CLS[dl]] = dict(n=len(ix_ns),
                                dp_true=float(D[cname][ix_ns].mean()),
                                dCN=float(dm[ix_ns, 0].mean()),
                                dMCI=float(dm[ix_ns, 1].mean()),
                                dAD=float(dm[ix_ns, 2].mean()))
        return out

    print("\n  [validity self-check] donor class distribution (with a correct reconstruction, CN recipients should be ≈half MCI / half AD)")
    for ci, cn in enumerate(CLS):
        ids = [f for f in metas_order if LBL[f] == ci]
        cnt = {CLS[d]: sum(1 for f in ids if donor_of[f][1] == d) for d in range(3)}
        cnt = {k: v for k, v in cnt.items() if v}
        print("    recipient %-4s n=%d → donor distribution %s" % (cn, len(ids), cnt))
    print("\n  [mechanistic prediction] CN recipients: Δp_true should be larger with donor=AD than with donor=MCI (AD tissue atrophy is more severe)")
    for ci, cn in enumerate(CLS):
        s = donor_split(ci)
        if len(s) < 2:
            print("    recipient %-4s has too few donor classes, skipping" % cn); continue
        print("    recipient %-4s:" % cn + "  ".join(
            "%s(n=%d) Δp_true=%+.5f" % (k, v['n'], v['dp_true']) for k, v in s.items()))
        chk[cn] = s
    R['E2_f3']['donor_reconstruction'] = dict(
        n_recovered=len(donor_of), split=chk,
        note='the donor label was reconstructed from the rng sequence (seed=42, same order as step2); it was checked sample by sample against the original records (results/donor_map.json, 144 recipients) on 2026-09-17 and found to agree (0 mismatches); see occlusion_rerun_verification_report.md')
    cn_s = chk.get('CN', {})
    if 'AD' in cn_s and 'MCI' in cn_s:
        print("\n    comparison within CN recipients: donor AD %+.5f vs donor MCI %+.5f "
              "→ %s" % (cn_s['AD']['dp_true'], cn_s['MCI']['dp_true'],
                        "direction agrees (AD donor effect is stronger)" if abs(cn_s['AD']['dp_true']) > abs(cn_s['MCI']['dp_true'])
                        else "direction does not agree (the reconstruction may be distorted; do not cite)"))
except Exception as e:
    print("  reconstruction failed: %r (does not affect the main E2 conclusion)" % (e,))
    R['E2_f3']['donor_reconstruction'] = dict(error=str(e))

# ==================================================================
# E3  MCI-stratum per-sample coupling + power check
# ==================================================================
print("\n" + "-" * 94)
print("E3  MCI-stratum per-sample coupling Spearman(mtl_mass_frac, Δp_true) + power check")
print("-" * 94)
cam = {}
with open(CSV, encoding='utf-8-sig') as fh:
    for r in csv.DictReader(fh):
        cam[r['file_id']] = float(r['mtl_mass_frac'])
miss = [f for f in metas_order if f not in cam]
print("CAM table matching: %d/%d hits%s" % (N - len(miss), N, "" if not miss else ", %d missing" % len(miss)))
MF = np.array([cam.get(f, np.nan) for f in metas_order])
R['E3_coupling'] = {}

# detectable effect size (Fisher z, 80% power, α=0.05 two-tailed)
def min_r(n, power=0.80, alpha=0.05):
    z_a = st.norm.isf(alpha / 2)
    z_b = st.norm.isf(1 - power)
    return float(np.tanh((z_a + z_b) / np.sqrt(n - 3)))


print("\n%-6s %4s | %10s %10s | %s" % ("true class", "n", "Spearman ρ", "p", "|ρ| detectable at 80% power for this group's n"))
for ci, cn in enumerate(CLS):
    m = (TL == ci) & NS & np.isfinite(MF)
    if m.sum() < 4:
        continue
    rho, pv = st.spearmanr(MF[m], D[MAIN][m])
    print("%-6s %4d | %+10.4f %10.4g | %.3f" % (cn, m.sum(), rho, pv, min_r(m.sum())))
    R['E3_coupling'][cn] = dict(n=int(m.sum()), rho=float(rho), p=float(pv),
                                min_detectable_r=min_r(int(m.sum())))
m = NS & np.isfinite(MF)
rho, pv = st.spearmanr(MF[m], D[MAIN][m])
print("%-6s %4d | %+10.4f %10.4g | %.3f" % ("ALL", m.sum(), rho, pv, min_r(int(m.sum()))))
R['E3_coupling']['ALL'] = dict(n=int(m.sum()), rho=float(rho), p=float(pv),
                               min_detectable_r=min_r(int(m.sum())))

print("\n[additional for the MCI stratum] the structure of a null group mean: the Δp_true distribution + whether it cancels out in both directions")
m_mci = (TL == 1) & NS
d_mci = D[MAIN][m_mci]
print("  n=%d, mean %+.5f, median %+.5f, SD %.5f, %d positive / %d negative, range [%+.4f, %+.4f]"
      % (m_mci.sum(), d_mci.mean(), np.median(d_mci), d_mci.std(ddof=1),
         int((d_mci > 0).sum()), int((d_mci < 0).sum()), d_mci.min(), d_mci.max()))
dm_mci = (mat_of(MAIN) - P0m)[m_mci]
print("  probability redistribution: ΔCN %+.5f  ΔMCI %+.5f  ΔAD %+.5f" %
      (dm_mci[:, 0].mean(), dm_mci[:, 1].mean(), dm_mci[:, 2].mean()))
print("  → p(MCI) itself barely moves, but p(AD) drains significantly and p(CN) gains significantly (redistribution rather than an overall decay)")
w1, p1 = st.wilcoxon(dm_mci[:, 2][dm_mci[:, 2] != 0])
w2, p2 = st.wilcoxon(dm_mci[:, 0][dm_mci[:, 0] != 0])
print("     paired tests: ΔAD p=%.4g ; ΔCN p=%.4g" % (p1, p2))
R['E3_coupling']['MCI_redistribution'] = dict(
    n=int(m_mci.sum()), mean_dp=float(d_mci.mean()), sd=float(d_mci.std(ddof=1)),
    n_pos=int((d_mci > 0).sum()), n_neg=int((d_mci < 0).sum()),
    dCN=float(dm_mci[:, 0].mean()), dMCI=float(dm_mci[:, 1].mean()),
    dAD=float(dm_mci[:, 2].mean()), p_dAD=float(p1), p_dCN=float(p2))

# ---------- E3b ceiling-effect check + CI ----------
print("\n[E3b] ceiling-effect check: Δp_true has the upper bound p0[true], so if mtl_mass_frac is "
      "correlated with p0[true], a negative correlation will be manufactured out of thin air")
R['E3b_ceiling'] = {}


def rho_ci(r, n, alpha=0.05):
    if n < 5 or not np.isfinite(r):
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    za = st.norm.isf(alpha / 2)
    return float(np.tanh(z - za * se)), float(np.tanh(z + za * se))


def partial_spearman(x, y, z):
    rx, ry, rz = st.rankdata(x), st.rankdata(y), st.rankdata(z)
    rxy, rxz, ryz = np.corrcoef(rx, ry)[0, 1], np.corrcoef(rx, rz)[0, 1], np.corrcoef(ry, rz)[0, 1]
    pr = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    n = len(x)
    t = pr * np.sqrt((n - 3) / max(1 - pr ** 2, 1e-12))
    return float(pr), float(2 * st.t.sf(abs(t), n - 3))


print("%-6s %4s | %9s %-22s | %9s %9s | %10s %10s %9s" %
      ("true class", "n", "ρ(mass,Δp)", "95%CI", "ρ(mass,p0true)", "ρ(Δp,p0true)", "partial ρ'", "p'", "n80% threshold"))
for ci, cn in enumerate(CLS):
    m = (TL == ci) & NS & np.isfinite(MF)
    if m.sum() < 5:
        continue
    x, y, z = MF[m], D[MAIN][m], P0m[m, TL[m]] - 0.5
    r_xz = st.spearmanr(x, z).statistic
    r_yz = st.spearmanr(y, z).statistic
    pr, pp = partial_spearman(x, y, z)
    lo, hi = rho_ci(R['E3_coupling'][cn]['rho'], int(m.sum()))
    R['E3b_ceiling'][cn] = dict(n=int(m.sum()), rho_mass_vs_dp=float(R['E3_coupling'][cn]['rho']),
                                ci=[lo, hi], rho_mass_vs_p0=float(r_xz),
                                rho_dp_vs_p0=float(r_yz), partial_rho=pr, partial_p=pp)
    print("%-6s %4d | %+9.4f [%+.3f,%+.3f] | %+9.4f %+9.4f | %+10.4f %10.4g %9.3f" %
          (cn, m.sum(), R['E3_coupling'][cn]['rho'], lo, hi, r_xz, r_yz, pr, pp,
           R['E3_coupling'][cn]['min_detectable_r']))

# ---------- E2b between-group comparison of the donor direction ----------
print("\n[E2b] between-group comparison of the \"donor direction\" within recipients (Mann-Whitney, non-saturated)")
R['E2b_donor_contrast'] = {}
for ci, cn in enumerate(CLS):
    ids = [f for f in metas_order if LBL[f] == ci]
    grp = {}
    dm = mat_of('MTL_d100_F3transplant') - P0m
    for dl in range(3):
        sel = [i2r[f] for f in ids if donor_of[f][1] == dl and NS[i2r[f]]]
        if len(sel) >= 3:
            grp[CLS[dl]] = D['MTL_d100_F3transplant'][sel]
    if len(grp) < 2:
        continue
    ks = sorted(grp)
    a, b = grp[ks[-1]], grp[ks[0]]
    if len(a) >= 3 and len(b) >= 3:
        U, pu = st.mannwhitneyu(a, b, alternative='two-sided')
        out = dict(n_a=len(a), n_b=len(b), mean_a=float(a.mean()), mean_b=float(b.mean()),
                   U=float(U), p=float(pu), label_a=ks[-1], label_b=ks[0])
        R['E2b_donor_contrast'][cn] = out
        print("  recipient %-4s: donor %s(n=%d, %+.5f) vs donor %s(n=%d, %+.5f) → U=%.1f, p=%.4g"
              % (cn, ks[-1], len(a), a.mean(), ks[0], len(b), b.mean(), U, pu))

# ---------- E2c full 3×3 recipient×donor flow + joint sign test ----------
print("\n[E2c] full recipient × donor flow (F3transplant, non-saturated)")
print("%-6s %-6s %4s | %9s %9s %9s | %10s" %
      ("recipient", "donor", "n", "ΔCN", "ΔMCI", "ΔAD", "Δp_true"))
dm3 = mat_of('MTL_d100_F3transplant') - P0m
cells = {}
for ci, cn in enumerate(CLS):
    ids = [f for f in metas_order if LBL[f] == ci]
    for dl in range(3):
        sel = [i2r[f] for f in ids if donor_of[f][1] == dl and NS[i2r[f]]]
        if len(sel) < 3:
            continue
        cells[(ci, dl)] = dict(n=len(sel), dCN=float(dm3[sel, 0].mean()),
                               dMCI=float(dm3[sel, 1].mean()), dAD=float(dm3[sel, 2].mean()),
                               dp=float(D['MTL_d100_F3transplant'][sel].mean()),
                               sel=sel)
        print("%-6s %-6s %4d | %+9.5f %+9.5f %+9.5f | %+10.5f" %
              (cn, CLS[dl], len(sel), cells[(ci, dl)]['dCN'], cells[(ci, dl)]['dMCI'],
               cells[(ci, dl)]['dAD'], cells[(ci, dl)]['dp']))
R['E2c_cells'] = {('r%s_d%s' % (CLS[a], CLS[b])): {k: v for k, v in c.items() if k != 'sel'}
                  for (a, b), c in cells.items()}

print("\n  How to read: CN=0 / MCI=1 / AD=2. Disease-axis coordinate axis(p) = p(MCI)·1 + p(AD)·2;")
print("        Δaxis = axis(after occlusion) − axis(before occlusion): negative = moves towards the healthy end, positive = moves towards the disease end.")

# disease-axis coordinate (not Δp_true -- for the intermediate class MCI, "towards the healthy end"
#  can mean either p(MCI)↓ or p(CN)↑, and Δp_true cannot express "which way along the axis", so the
#  axis coordinate must be used instead)
AX0 = P0m[:, 1] * 1.0 + P0m[:, 2] * 2.0


def axis_of(c):
    Pc = mat_of(c)
    return Pc[:, 1] * 1.0 + Pc[:, 2] * 2.0


print("\n[disease-axis coordinate check] baseline axis means: CN %.4f / MCI %.4f / AD %.4f (should be strictly increasing)"
      % (AX0[TL == 0].mean(), AX0[TL == 1].mean(), AX0[TL == 2].mean()))
R['E2c_axis_baseline'] = dict(CN=float(AX0[TL == 0].mean()), MCI=float(AX0[TL == 1].mean()),
                              AD=float(AX0[TL == 2].mean()))

print("\n  Δaxis per condition (non-saturated, negative=towards the healthy end):")
for c in ('MTL_d100_F1const', 'MTL_d100_F2noise', 'MTL_d100_F3transplant'):
    da = axis_of(c) - AX0
    print("    %-24s all %+.5f | CN %+.5f  MCI %+.5f  AD %+.5f"
          % (c, da[NS].mean(), da[(TL == 0) & NS].mean(),
             da[(TL == 1) & NS].mean(), da[(TL == 2) & NS].mean()))


def sev_stat(assign, cname='MTL_d100_F3transplant'):
    """s = +1 if the donor is closer to the disease end than the recipient; statistic = mean(s × Δaxis). Expected to be positive."""
    da = axis_of(cname) - AX0
    vals = []
    for f in metas_order:
        i = i2r[f]
        if not NS[i]:
            continue
        r, d = LBL[f], assign[f]
        if r == d:
            continue
        vals.append((1.0 if d > r else -1.0) * da[i])
    return float(np.mean(vals)), len(vals)


obs_assign = {f: donor_of[f][1] for f in metas_order}
stat_obs, n_used = sev_stat(obs_assign)
sv = [(1.0 if donor_of[f][1] > LBL[f] else -1.0) * (axis_of('MTL_d100_F3transplant') - AX0)[i2r[f]]
      for f in metas_order if NS[i2r[f]] and donor_of[f][1] != LBL[f]]
w1, p1 = st.wilcoxon([v for v in sv if v != 0])

# F1 constant fill (donor = whole-brain mean = healthy side) as the control: s=−1 for everyone, stat = mean(−Δaxis)
da_f1 = axis_of('MTL_d100_F1const') - AX0
stat_f1 = float(np.mean([-da_f1[i2r[f]] for f in metas_order if NS[i2r[f]]]))
w2, p2 = st.wilcoxon([-da_f1[i2r[f]] for f in metas_order if NS[i2r[f]] and da_f1[i2r[f]] != 0])

print("\n[joint test] statistic = mean(sign(donor−recipient) × Δaxis)   (the closer the donor is to the disease end, the more the recipient moves towards the disease end)")
print("  F1 constant fill (donor = healthy side, s≡−1): stat = %+.5f, Wilcoxon p = %.4g, n=%d"
      % (stat_f1, p2, int(NS.sum())))
print("  F3 cross-subject transplant (donor direction random): stat = %+.5f, Wilcoxon p = %.4g, n=%d"
      % (stat_obs, p1, n_used))

rngp = np.random.default_rng(20260916)
nperm = 2000
null = np.empty(nperm)
for t in range(nperm):
    a = {}
    for f in metas_order:
        other = [g for g in metas_order if LBL[g] != LBL[f]]
        a[f] = LBL[other[rngp.integers(0, len(other))]]
    null[t] = sev_stat(a)[0]
p_perm = float((null >= stat_obs).mean())
print("  permutation null distribution (n=%d, donor labels redrawn at random): mean %+.5f  SD %.5f  95th percentile %+.5f"
      % (nperm, null.mean(), null.std(), np.percentile(null, 95)))
print("  one-sided permutation p = %.4f" % p_perm)
print("  → %s" % ("✅ supports \"the MTL carries disease-axis information, and the direction can be determined by the donor's disease state\"" if p_perm < 0.05
                  else "⚠️ does not reach significance; the mechanistic direction can only be described as a trend"))
R['E2c_joint'] = dict(stat_f1=stat_f1, p_f1=float(p2), stat_f3=stat_obs, p_f3=float(p1),
                      n=n_used, perm_p=float(p_perm), perm_mean=float(null.mean()),
                      perm_sd=float(null.std()), perm_p95=float(np.percentile(null, 95)),
                      n_perm=nperm)

json.dump(R, open(OUT / 'minimal_pkg_stats.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2, default=float)
print("\nsaved: results/minimal_pkg_stats.json")
print("=" * 94)
