# -*- coding: utf-8 -*-
"""Occlusion causal analysis Step 3d: pre-registration compliance add-on tests (CPU, seconds; read-only over occlusion_raw.npz).

Corresponds to the paper: supplementary material S6 (multiple-comparison correction, probability-destination decomposition, exact permutation test).

For the gaps found by the "pre-registration pipeline audit", this performs a zero-cost (pure post-hoc) fill-in:
  P1 effect-size definition  uniformly adopts matched-pairs rank-biserial ∈ [−1,1], and additionally reports Cohen's d_z
                             (r = z/√N exceeds its bounds at small sample sizes; here the nominal value for the AD stratum already reaches 1.087 > 1)
  P2 Holm multiple-comparison correction  Holm for the {each condition vs 0} family; Holm for the {MTL vs C1/C2/C3} family
  P3 C3 vs C2 direct paired comparison  the only genuinely missing paired comparison (core evidence for specificity)
  P4 AD-group probability-destination decomposition  after Δp_AD decreases, does the flow go to MCI or CN (the three Δp terms always sum to 0, so the destination can be decomposed directly)
  P5 Δp cross-stratified by correct/incorrect × true class
  P6 dose-response exact permutation p, 5 levels, 5! = 120 full-permutation exact distribution (replacing the asymptotic t approximation)

Definition fully consistent with step3 / step3c: non-saturated subset n=80 (SAT_THR is based on baseline p0 only).
⚠ This script only performs description and testing; it makes no causal claim.

Run: python occlusion/step3d_patch_flow.py"""

# Source: 27_occlusion_experiment/step3d_patch.py (computation logic unchanged line by line; only the
# "environment bootstrap layer" was changed to be provided uniformly by occlusion_config;
# see occlusion/README.md for the adaptation notes).

import json
import sys
import itertools
import numpy as np
from scipy import stats as st

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from occlusion_config import (  # noqa: E402
    OUT, SAT_THR, setup_stdout,
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
N = len(uniq)
i2r = {f: i for i, f in enumerate(uniq)}
CLS = ['CN', 'MCI', 'AD']
MAIN = 'MTL_d100_F1const'


def mat_of(c):
    rows = {i: None for i in range(N)}
    for k, f in enumerate(file_ids):
        if cond_names[cond_idx[k]] == c:
            rows[i2r[f]] = P1[k]
    return np.array([rows[i] for i in range(N)])


first = {}
for k, f in enumerate(file_ids):
    if f not in first:
        first[f] = k
P0m = np.array([P0[first[f]] for f in uniq])
TL = np.array([int(true_lab[first[f]]) for f in uniq])
IDX = np.arange(N)

SAT = P0m.max(1) > 0.999999
NS = ~SAT
n_sat, n_ns = int(SAT.sum()), int(NS.sum())

D = {c: P0m[IDX, TL] - mat_of(c)[IDX, TL] for c in cond_names}   # Δp_true, positive = true-class probability drops after occlusion
R = {'scope': dict(n_all=N, n_nonsat=n_ns, n_sat=n_sat, sat_thr=0.999999)}


def eff(d, name=''):
    """Complete effect-size panel for the paired difference d (positive = true-class probability drops)"""
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return dict(name=name, n=0)
    w, p = st.wilcoxon(d, alternative='two-sided')
    z = float(st.norm.isf(p / 2))
    rk = st.rankdata(np.abs(d))
    Tp = float(rk[d > 0].sum())
    Tn = float(rk[d < 0].sum())
    r_rb = (Tp - Tn) / (Tp + Tn)                    # matched-pairs rank-biserial ∈ [-1,1]
    d_z = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float('nan')
    r_ros = z / np.sqrt(n)                          # ← old definition (Rosenthal r), can be > 1
    return dict(name=name, n=int(n), median_delta=float(np.median(d)),
                mean_delta=float(d.mean()), W=float(w), p_value=float(p),
                n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()),
                r_rank_biserial=float(r_rb), cohen_dz=d_z, r_rosenthal=float(r_ros))


def holm(pvals, labels, alpha=0.05):
    """Holm-Bonferroni step-down; returns each item's corrected p and whether it remains significant"""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return [dict(label=labels[i], p_raw=float(pvals[i]), p_holm=float(adj[i]),
                 sig_holm=bool(adj[i] < alpha)) for i in range(m)]


print("=" * 92)
print("Step 3d  pre-registration compliance add-on tests")
print("=" * 92)
print("all samples n=%d, nonsat n=%d, sat n=%d (SAT_THR=0.999999, based on baseline p0 only)" % (N, n_ns, n_sat))

# ==================== P1 effect-size panel (main mask + three strata) ====================
print("\n" + "-" * 92)
print("P1  effect-size definition: rank-biserial (bounded) / Cohen's d_z / Rosenthal r")
print("-" * 92)
print("%-34s %5s | %9s %9s %10s %10s %11s" %
      ("definition", "n", "medianΔ", "meanΔ", "rank-bis", "Cohen d_z", "Rosenthal r"))
panel = {}
rows = [('all MTL_d100', D[MAIN])]
for ci, cn in enumerate(CLS):
    rows.append(('nonsat %s stratum MTL_d100' % cn, D[MAIN][(TL == ci) & NS]))
rows.append(('nonsat all MTL_d100', D[MAIN][NS]))
for nm, dd in rows:
    e = eff(dd, nm)
    panel[nm] = e
    print("%-34s %5d | %+9.5f %+9.5f %+10.3f %+10.3f %+11.3f" %
          (nm, e['n'], e['median_delta'], e['mean_delta'],
           e['r_rank_biserial'], e['cohen_dz'], e['r_rosenthal']))
R['P1_effect_panel'] = panel

# ==================== P2 Holm multiple comparisons ====================
print("\n" + "-" * 92)
print("P2  Holm correction (nonsat n=80, each condition vs 0)")
print("-" * 92)
ps, labs, detail = [], [], {}
for c in cond_names:
    e = eff(D[c][NS], c)
    detail[c] = e
    ps.append(e['p_value'])
    labs.append(c)
fam_all = holm(ps, labs)
print("family A: all 10 conditions")
for r_ in sorted(fam_all, key=lambda x: x['p_raw']):
    print("   %-24s p_raw=%.5g  p_holm=%.5g  %s" %
          (r_['label'], r_['p_raw'], r_['p_holm'], "still significant" if r_['sig_holm'] else "→ no longer significant"))
MAIN6 = [MAIN, 'MTL_d100_F2noise', 'MTL_d100_F3transplant',
         'C1_cort_23_F1const', 'C2_ball_F1const', 'C3_hipp_F1const']
fam6 = holm([detail[c]['p_value'] for c in MAIN6], MAIN6)
print("family B: 6 main conditions (MTL×3 fill strategies + C1/C2/C3)")
for r_ in sorted(fam6, key=lambda x: x['p_raw']):
    print("   %-24s p_raw=%.5g  p_holm=%.5g  %s" %
          (r_['label'], r_['p_raw'], r_['p_holm'], "still significant" if r_['sig_holm'] else "→ no longer significant"))
R['P2_holm_vs0'] = dict(family_all=fam_all, family_main6=fam6, per_condition=detail)

# ==================== P3 direct paired-comparison family (incl. C3 vs C2) ====================
print("\n" + "-" * 92)
print("P3  direct paired comparisons (core evidence for specificity): MTL vs C1 / MTL vs C2 / MTL vs C3 / C3 vs C2")
print("-" * 92)
PAIRS = [('MTL_d100_F1const', 'C1_cort_23_F1const', 'MTL vs C1 non-MTL cortex'),
         ('MTL_d100_F1const', 'C2_ball_F1const', 'MTL vs C2 random sphere'),
         ('MTL_d100_F1const', 'C3_hipp_F1const', 'MTL vs C3 hippocampus proper'),
         ('C3_hipp_F1const', 'C2_ball_F1const', 'C3 vs C2 hippocampus vs random sphere')]
pair_out = []
print("%-26s %5s | %10s %9s %9s %9s %8s" %
      ("pair", "n", "Wilcoxon p", "rank-bis", "d_z", "mean diff", "n+/n-"))
for a, b, lab in PAIRS:
    dd = (D[a] - D[b])[NS]
    e = eff(dd, lab)
    e['pair'] = (a, b)
    pair_out.append(e)
    print("%-26s %5d | %10.5g %+9.3f %+9.3f %+9.5f %4d/%-3d" %
          (lab, e['n'], e['p_value'], e['r_rank_biserial'], e['cohen_dz'],
           e['mean_delta'], e['n_pos'], e['n_neg']))
print("\nFamily Holm (4 paired comparisons):")
f4 = holm([e['p_value'] for e in pair_out], [e['name'] for e in pair_out])
for r_ in sorted(f4, key=lambda x: x['p_raw']):
    print("   %-26s p_raw=%.5g  p_holm=%.5g  %s" %
          (r_['label'], r_['p_raw'], r_['p_holm'], "still significant" if r_['sig_holm'] else "→ no longer significant"))
R['P3_direct_pairs'] = dict(per_pair=pair_out, holm=f4)

# ==================== P4 AD-group probability-destination decomposition ====================
print("\n" + "-" * 92)
print("P4  probability-destination decomposition (main mask %s; the three Δp terms always sum to 0, so the destination can be decomposed)" % MAIN)
print("-" * 92)
Pm = mat_of(MAIN)
dM = Pm - P0m     # (N,3) "net change" of each class probability, positive = that recipient-class probability rises after occlusion
dest = {}
for ci, cn in enumerate(CLS):
    m = TL == ci
    for tag, mask in (('all', m), ('nonsat', m & NS), ('sat', m & SAT)):
        if mask.sum() == 0:
            continue
        dm = dM[mask]
        gains = dm[:, [0, 1]].mean(0)
        tot = float(gains.sum())
        dest['%s|%s' % (cn, tag)] = dict(
            n=int(mask.sum()),
            mean_change_CN=float(dm[:, 0].mean()), mean_change_MCI=float(dm[:, 1].mean()),
            mean_change_AD=float(dm[:, 2].mean()),
            mean_drop_true=float(D[MAIN][mask].mean()),
            share_to_CN_vs_MCI=float(gains[0] / tot) if tot > 1e-12 else float('nan'))
print("%-16s %4s | %11s %11s %11s | %13s" %
      ("true class|definition", "n", "ΔCN(+rise)", "ΔMCI(+rise)", "ΔAD(+rise)", "true-class drop"))
for k, v in dest.items():
    if '|' not in k:
        continue
    print("%-16s %4d | %+11.5f %+11.5f %+11.5f | %+13.5f" %
          (k, v['n'], v['mean_change_CN'], v['mean_change_MCI'],
           v['mean_change_AD'], v['mean_drop_true']))

# AD stratum: pred destination counts (true class = AD)
m_ad = TL == 2
pred0, pred1 = P0m.argmax(1), Pm.argmax(1)
tab = np.zeros((3, 3), int)
for i in np.where(m_ad)[0]:
    tab[pred0[i], pred1[i]] += 1
print("\nAD-stratum prediction destination (rows = baseline prediction, cols = prediction after occlusion; n=%d all samples)" % int(m_ad.sum()))
print("         " + "".join("%8s" % c for c in CLS))
for i, cn in enumerate(CLS):
    print("  base %-6s" % cn + "".join("%8d" % tab[i, j] for j in range(3)))
dest['AD_flip_table'] = tab.tolist()
dest['AD_flip_table_labels'] = CLS

# AD stratum, fine-grained: counts of the probability "most probable recipient" + median flow + individual values of flipped samples
m_ad_ns = m_ad & NS
dm_ad = dM[m_ad_ns]          # positive = that class probability rises
gain = dm_ad.argmax(1)
dest['AD_nonsat_gain_argmax_count'] = {CLS[j]: int((gain == j).sum()) for j in range(3)}
dest['AD_nonsat_median_in'] = {CLS[j]: float(np.median(dm_ad[:, j])) for j in range(3)}
dest['AD_nonsat_mean_in'] = {CLS[j]: float(dm_ad[:, j].mean()) for j in range(3)}
share = dm_ad[:, [0, 1]].mean(0)
dest['AD_nonsat_share_CN_vs_MCI'] = float(share[0] / (share[0] + share[1]))
print("\nAD stratum (nonsat n=%d) most probable recipient counts: CN %d / MCI %d / AD %d"
      % (int(m_ad_ns.sum()), dest['AD_nonsat_gain_argmax_count']['CN'],
         dest['AD_nonsat_gain_argmax_count']['MCI'],
         dest['AD_nonsat_gain_argmax_count']['AD']))
print("AD stratum (nonsat) median net change: CN %+.5f / MCI %+.5f / AD %+.5f"
      % (dest['AD_nonsat_median_in']['CN'], dest['AD_nonsat_median_in']['MCI'],
         dest['AD_nonsat_median_in']['AD']))
print("→ share of the lost probability flowing to CN = %.1f%% (flowing to MCI = %.1f%%)"
      % (100 * dest['AD_nonsat_share_CN_vs_MCI'],
         100 * (1 - dest['AD_nonsat_share_CN_vs_MCI'])))
flip_ix = [i for i in np.where(m_ad)[0] if pred0[i] == 2 and pred1[i] == 0]
print("AD→CN flipped samples, one by one (n=%d):" % len(flip_ix))
for i in flip_ix:
    print("   %-22s p0=[%.4f,%.4f,%.4f] → p=[%.4f,%.4f,%.4f]  Δp_true=%+.4f"
          % (uniq[i][:22], P0m[i, 0], P0m[i, 1], P0m[i, 2],
             Pm[i, 0], Pm[i, 1], Pm[i, 2], D[MAIN][i]))
    print("      confidence %.4f → %.4f  |  nonsat=%s" % (P0m[i].max(), Pm[i].max(), bool(NS[i])))
dest['AD_to_CN_flip_ids'] = [uniq[i] for i in flip_ix]

# Cross-condition comparison: AD-stratum flow (to judge whether "AD→CN" is specific to F1const, i.e. an OOD artefact)
print("\nAD-stratum flow comparison across conditions (nonsat AD n=%d):" % int(m_ad_ns.sum()))
print("%-24s %10s %10s %10s %10s %9s" %
      ("condition", "true-class drop", "ΔCN(+rise)", "ΔMCI(+rise)", "CN share %", "AD recall Δ"))
cross_flow = {}
for c in cond_names:
    dmc = (mat_of(c) - P0m)[m_ad_ns]
    s = dmc[:, [0, 1]].mean(0)
    totshare = s[0] / (s[0] + s[1]) if (s[0] + s[1]) > 1e-12 else float('nan')
    pc_ = mat_of(c).argmax(1)
    adr = float(np.mean(pc_[m_ad] == 2)) - float(np.mean(pred0[m_ad] == 2))
    cross_flow[c] = dict(mean_drop_true=float(D[c][m_ad_ns].mean()),
                         change_CN=float(s[0]), change_MCI=float(s[1]),
                         share_to_CN=float(totshare), ad_recall_delta=adr)
    print("%-24s %+10.5f %+10.5f %+10.5f %10s %+9.4f" %
          (c, cross_flow[c]['mean_drop_true'], s[0], s[1],
           ("%.1f" % (100 * totshare)) if np.isfinite(totshare) else "n/a", adr))
dest['flow_by_condition'] = cross_flow
R['P4_prob_destination'] = dest

# ==================== P5 Δp by correct/incorrect × true class ====================
print("\n" + "-" * 92)
print("P5  Δp_true by prediction correctness × true class (nonsat n=80)")
print("-" * 92)
cor0 = (pred0 == TL)
cross = {}
print("%-6s %-8s %5s | %11s %11s %9s %9s %8s" %
      ("true class", "prediction", "n", "medianΔ", "meanΔ", "n+/n-", "p", "rank-bis"))
for ci, cn in enumerate(CLS):
    for lab, sel in (('correct', cor0), ('incorrect', ~cor0)):
        mask = (TL == ci) & sel & NS
        if mask.sum() == 0:
            continue
        e = eff(D[MAIN][mask], '%s/%s' % (cn, lab))
        cross['%s|%s' % (cn, lab)] = e
        print("%-6s %-8s %5d | %+11.5f %+11.5f %5d/%-3d %9.4g %+8.3f" %
              (cn, lab, e['n'], e['median_delta'], e['mean_delta'],
               e['n_pos'], e['n_neg'], e['p_value'], e['r_rank_biserial']))
for lab, sel in (('correct', cor0), ('incorrect', ~cor0)):
    mask = sel & NS
    e = eff(D[MAIN][mask], 'all/%s' % lab)
    cross['all|%s' % lab] = e
    print("%-6s %-8s %5d | %+11.5f %+11.5f %5d/%-3d %9.4g %+8.3f" %
          ('all', lab, e['n'], e['median_delta'], e['mean_delta'],
           e['n_pos'], e['n_neg'], e['p_value'], e['r_rank_biserial']))
R['P5_correctness_x_class'] = cross

# ==================== P6 dose-response exact permutation p ====================
print("\n" + "-" * 92)
print("P6  dose-response: 5-level exact permutation test (5! = 120 full permutations)")
print("-" * 92)
DOSE = ['MTL_d25_F1const', 'MTL_d40_F1const', 'MTL_d60_F1const',
        'MTL_d80_F1const', 'MTL_d100_F1const']
VOX = [7941, 12706, 19058, 25411, 31764]
med = [float(np.median(D[c][NS])) for c in DOSE]
mn = [float(D[c][NS].mean()) for c in DOSE]
print("dose    :  " + "  ".join("%9s" % v for v in VOX))
print("medianΔ :  " + "  ".join("%+9.2e" % v for v in med))
print("meanΔ   :  " + "  ".join("%+9.5f" % v for v in mn))


def rho(x, y):
    return st.spearmanr(x, y).statistic


def exact_perm_p(x, y):
    """Exact permutation p: x fixed, y fully permuted; two-sided = #{|ρ_perm| >= |ρ_obs|} / n!"""
    obs = abs(rho(x, y))
    vals = [abs(rho(x, list(p))) for p in itertools.permutations(y)]
    vals = np.array(vals)
    return float(obs), float((vals >= obs - 1e-12).mean()), int(len(vals)), float(vals.mean())


for tag, seq in (('medianΔ', med), ('meanΔ', mn)):
    obs, pex, nperm, mnull = exact_perm_p(VOX, seq)
    asym = st.spearmanr(VOX, seq)
    print("%s : ρ_obs = %.4f | exact two-sided permutation p = %.4f (=%d/%d) | asymptotic t approximation p = %.4f"
          % (tag, obs, pex, int(round(pex * nperm)), nperm, asym.pvalue))
    R.setdefault('P6_dose_exact', {})[tag] = dict(
        rho=float(obs), p_exact_two_sided=pex, n_perm=nperm,
        p_asymptotic_t=float(asym.pvalue),
        perm_null_mean_abs_rho=float(mnull))

# Adjacent dose-level pairing (finer evidence of monotonicity)
print("\nAdjacent dose levels, paired Wilcoxon (nonsat n=80):")
adj = []
for i in range(len(DOSE) - 1):
    lo, hi = DOSE[i], DOSE[i + 1]
    e = eff((D[hi] - D[lo])[NS], '%s_+%d - %s_%d' % (hi.split('_')[1], VOX[i + 1],
                                                     lo.split('_')[1], VOX[i]))
    adj.append(e)
    print("   Δ(%s - %s): p = %.4g  rank-bis = %+.3f  %d/%d (positive/negative)"
          % (e['name'].split(' - ')[0], e['name'].split(' - ')[1],
             e['p_value'], e['r_rank_biserial'], e['n_pos'], e['n_neg']))
R['P6_dose_exact']['adjacent_pairs'] = adj

# ==================== write to disk ====================
json.dump(R, open(OUT / 'patch_stats.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2, default=float)
print("\nsaved: results/patch_stats.json")
print("=" * 92)
