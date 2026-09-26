# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 5: report generation (CPU, seconds).

Reads <work>/results/occlusion_stats.json (+ optionally perclass_stats.json),
and writes <work>/occlusion_report.md (data summary and interpretation).

Key points
----------
1. main analysis convention = non-saturated subset n=80 (the Δp_true of saturated samples is
   swallowed by the float64 softmax clamp);
2. the M1 table gives the all-samples n=144 and the non-saturated n=80 columns side by side;
3. newly added: an explicit disclosure of the two-way flips and of the zero net accuracy change;
4. the interpretation must disclose that "MTL has no significant advantage over the hippocampus alone".

Run: python occlusion/step5_report.py"""

# Source: 27_occlusion_experiment/step5_report.py (computation logic unchanged line by line; only the
# "environment bootstrap layer" was changed to be provided uniformly by occlusion_config;
# see occlusion/README.md for the adaptation notes).

import json
import sys
import numpy as np

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from occlusion_config import (  # noqa: E402
    OUT, ensure_dirs, setup_stdout,
)

setup_stdout()
ensure_dirs()

R = json.load(open(OUT / 'occlusion_stats.json', encoding='utf-8'))
meta = json.load(open(OUT / 'occlusion_meta.json', encoding='utf-8'))
NULL = None
_p = OUT / 'random_null_stats.json'
if _p.exists():
    NULL = json.load(open(_p, encoding='utf-8'))
PC = None
_p2 = OUT / 'perclass_stats.json'
if _p2.exists():
    PC = json.load(open(_p2, encoding='utf-8'))

SAT = R['saturation']
N_NS = SAT['n_non_saturated']
N_ALL = 144

L = []
A = L.append
A("# Occlusion experiment report (MTL causality evidence)\n")
A("**Date**: 2026-09-16  **Model**: Run 128 (3D DenseNet-169 + axial spatial gating, 5-fold equal-weight ensemble)")
A("**Samples**: ADNI fixed test set n = 144  **Main mask**: MTL (same convention as the paper's T3, 31,764 voxels = 1.830%)")
A("**Inference convention**: deterministic (`cudnn.deterministic=True` + `benchmark=False` + "
  "`use_deterministic_algorithms(warn_only=True)`), bitwise identical to the Step 0 baseline (max|Δp| = 0.00e+00)\n")
A("")
A("**Correspondence with supplementary material S6** (S6 = the methodological supplement on interpretability analysis and robustness):\n")
A("| Supplementary S6 item | Corresponding subsection of this report (if not covered, it points to the artifact) |")
A("|---|---|")
A("| (1) Methods and reproducibility controls | not contained in this report; see `rerun_diag.json`, `rerun_consistency.json`, `probe_vs_data.json`, `repeat_check.json` |")
A("| (2) Masks and controls | the \"Main mask\" convention line in the header; for the authoritative volume values see `masks_v2_meta.json` |")
A("| (3) Fill strategy | §7  M6  fill-strategy robustness |")
A("| (4) Main results and multiple comparisons | §0  conventions and limitations, §2  M1  Δp_true paired test, §3  M2  prediction flip rate |")
A("| (5) Direct paired comparison | §6  M5  main mask vs controls |")
A("| (6) Dose-response | §4  M3  dose-response |")
A("| (7) Disease-axis coordinates and two-way transplant | not contained in this report; see `minimal_pkg_stats.json` (E2 donor reconstruction) |")
A("| (8) Prediction entropy | not contained in this report; see `minimal_pkg_stats.json` (E1 entropy) |")
A("| (9) Stratification and per-sample coupling (exploratory) | §5  M4  sample-level coupling, §7·supplement ii  stratified by true class |")
A("| (10) On the random null-distribution control | §7·supplement  random-mask null distribution (this control is retired, see `occlusion/README.md` §8) |")
A("| (11) Inferences that should not be drawn (statement of limitations) | §0  conventions and limitations, §1  interpretation |")
A("")
A("---\n")

# ---------- 0 convention warnings ----------
A("## 0. Read this section first: three convention limitations you must know up front (S6 (4) defining the main convention · (11) statement of limitations)\n")
A("> This section is a **disclosure of limitations**, not a conclusion. Ignoring these three items will overstate the strength of the conclusions.\n")
A("| # | Limitation | Fact | Consequence |")
A("|---|---|---|---|")
A("| 1 | **softmax saturation** | the top-1 probability of %d/%d cases is clamped to exactly 1.0 under float64"
  % (SAT['n_saturated'], N_ALL))
A("| | | for these samples Δp_true is swallowed at the bit level, **and all 29 strictly zero differences come from saturated samples** | the main analysis is restricted to the non-saturated subset n=%d |" % N_NS)
A("| 2 | **two-way flips** | after occlusion with the main mask, wrong→correct %d cases, correct→wrong %d cases" % (R['M2_flip']['MTL_d100_F1const']['flip_to_correct'],
                                                       R['M2_flip']['MTL_d100_F1const']['flip_to_wrong']))
A("| | | **net accuracy change = %+.4f (i.e. 0)**" % R['M2_flip']['MTL_d100_F1const']['acc_delta'])
A("| | | occlusion is **not** a one-way removal of the disease signal, but a two-way perturbation | one cannot argue from 'a drop in accuracy' |")
A("| 3 | **the effect size is very small** | for the main mask, non-saturated median Δ = %+.2e, mean = %+.5f (mean/median ≈ %.0f×)"
  % (R['M1_delta_p_true']['MTL_d100_F1const']['median_delta'],
     R['M1_delta_p_true']['MTL_d100_F1const']['mean_delta'],
     R['M1_delta_p_true']['MTL_d100_F1const']['mean_delta'] /
     R['M1_delta_p_true']['MTL_d100_F1const']['median_delta']))
A("| | | the effect is driven by a minority of samples | the median is almost uninformative, and the statistical significance rests on the sign counts of the Wilcoxon test |")
A("")
A("---\n")

# ---------- 1 interpretation ----------
v = R['verdict']
A("## 1. Interpretation (S6 (4) · (11))\n")
A("```")
A(v['text'])
A("```\n")
A("Basis for the interpretation (convention: %s):" % v.get('scope', '—'))
A("- significantly lower (main mask): `%s` (p = %.4g)"
  % ('yes' if v['sig_down'] else 'no', R['M1_delta_p_true']['MTL_d100_F1const']['p_value']))
A("- significantly stronger than the C1 volume-matched control: `%s` (p = %.4g)"
  % ('yes' if v.get('mtl_vs_c1', {}).get('p', 1) < 0.05 else 'no',
     v.get('mtl_vs_c1', {}).get('p', float('nan'))))
A("- significantly stronger than the C2 random-sphere control: `%s` (p = %.4g)"
  % ('yes' if v.get('mtl_vs_c2', {}).get('p', 1) < 0.05 else 'no',
     v.get('mtl_vs_c2', {}).get('p', float('nan'))))
A("- **significantly stronger than the C3 hippocampus-alone control: `%s` (p = %.4g) ← this is the single most important row**"
  % ('yes' if v.get('hipp_better') else 'no', v.get('mtl_vs_c3', {}).get('p', float('nan'))))
A("")
if NULL:
    A("- **random scattered-mask control**: this control **does not hold** (it mixes the two variables of volume and spatial coherence),")
    A("  see the spatial-coherence diagnosis in the '7·supplement' section. **It is not grounds for downgrading the interpretation.**")
    A("")


# ---------- 2 M1 ----------
A("---\n")
A("## 2. M1  Δp_true paired test (main conclusion metric) (S6 (4) main results and multiple comparisons)\n")
A("`Δp_true = p0[true] − p_occ[true]`, **a positive value means that occlusion lowers the true-class probability**.")
A("The table gives two columns side by side: **non-saturated n=%d (main convention)** and **all samples n=%d (for reference only)**.\n" % (N_NS, N_ALL))
A("| Condition | Mask voxels | Non-saturated median Δ | Non-saturated mean Δ | n⁺/n⁻ | p | r | All-samples p |")
A("|---|---|---|---|---|---|---|---|")
mv = meta['mask_vox']
for c, r in R['M1_delta_p_true'].items():
    if 'p_value' not in r:
        A("| `%s` | — | %.5f | — | — | %s | — | — |"
          % (c, r.get('median_delta', float('nan')), r.get('note', '')))
        continue
    vox = next((x['mask_vox'] for x in meta['conditions'] if x['name'] == c), 0)
    rf = R['M1_delta_p_true_full'].get(c, {})
    A("| `%s` | %s | %+.2e | %+.5f | %d/%d | %.3g | %+.3f | %.3g |"
      % (c, '{:,}'.format(vox), r['median_delta'], r['mean_delta'],
         r['n_pos'], r['n_neg'], r['p_value'], r['effect_r'], rf.get('p_value', float('nan'))))
A("")
A("> `n⁺/n⁻` = the number of samples with a positive/negative Δp_true in the non-saturated subset. `r` = paired Wilcoxon effect size (Z/√N).\n")
A("> ⚠️ **the median is of order +%.2e and the mean is of order %+.4f**, roughly a factor of %.0f apart:"
  % (R['M1_delta_p_true']['MTL_d100_F1const']['median_delta'],
     R['M1_delta_p_true']['MTL_d100_F1const']['mean_delta'],
     R['M1_delta_p_true']['MTL_d100_F1const']['mean_delta'] /
     R['M1_delta_p_true']['MTL_d100_F1const']['median_delta']))
A("> the effect is highly right-skewed and the mean is propped up by a few large-effect samples; **any citation should give the median first, or state the shape of the distribution explicitly**.\n")

# ---------- 3 M2 ----------
A("---\n")
A("## 3. M2  prediction flip rate (all-samples convention; flips are unaffected by saturation) (S6 (4))\n")
A("| Condition | Flip rate | Flip to correct (wrong→correct) | Flip to wrong (correct→wrong) | **Net flips** | Accuracy after occlusion | Relative to baseline |")
A("|---|---|---|---|---|---|---|")
A("| **Baseline** | — | — | — | — | %.4f | — |" % R['baseline_acc'])
for c, r in R['M2_flip'].items():
    A("| `%s` | %.4f | %d | %d | **%+d** | %.4f | %+.4f |"
      % (c, r['flip_rate'], r['flip_to_correct'], r['flip_to_wrong'],
         r['flip_to_correct'] - r['flip_to_wrong'],
         r['acc_after'], r['acc_delta']))
A("")
A("> ★ the net flips of the main mask MTL_d100 = **%+d**, and the net accuracy change = **%+.4f**."
  % (R['M2_flip']['MTL_d100_F1const']['flip_to_correct'] - R['M2_flip']['MTL_d100_F1const']['flip_to_wrong'],
     R['M2_flip']['MTL_d100_F1const']['acc_delta']))
A("> this shows that occlusion is **not** the one-way process of 'taking the disease signal away → more errors'; instead it pushes the")
A("> samples near the decision boundary to both sides; arguing from accuracy alone cannot support a causal claim.\n")


# ---------- 4 M3 ----------
A("---\n")
A("## 4. M3  dose-response (S6 (6))\n")
m3 = R['M3_dose_response']
A("| Level | Voxels | Share of the full MTL | Median Δp_true | Mean Δp_true |")
A("|---|---|---|---|---|")
for d, vx, md in zip(m3['doses'], m3['vox'], m3['median_delta']):
    A("| `%s` | %s | %.1f%% | %+.5f | %+.5f |"
      % (d, '{:,}'.format(vx), 100 * vx / mv['mtl'], md,
         R['M1_delta_p_true'][d]['mean_delta']))
A("")
A("**Spearman(voxels, median Δp_true)** = ρ %.4f, p = %.4g; "
  "**Spearman(voxels, mean Δp_true)** = ρ %.4f, p = %.4g\n"
  % (m3['spearman_rho_median'], m3['spearman_p_median'],
     m3['spearman_rho_mean'], m3['spearman_p_mean']))
A("> Note: `d100 vs d25` paired Wilcoxon W = %.0f, p = %.4g; "
  "in this round the 5 levels are the basic sampling sites and **no strict volume-gradient design was used** (the EDT distance is quantised to integers, "
  "and after switching to rank selection the volumes of the levels are not evenly spaced), so the dose-response is only suitable for a trend description.\n"
  % (m3['d100_vs_d25']['W'], m3['d100_vs_d25']['p']))

# ---------- 5 M4 ----------
A("---\n")
A("## 5. M4  sample-level coupling (Δp_true vs baseline metrics) (S6 (9))\n")
A("| Baseline metric | Spearman ρ | p |")
A("|---|---|---|")
for k, r in R['M4_coupling'].items():
    if k == 'correctness_group':
        continue
    A("| %s | %+.4f | %.4g |" % (r['label'], r['rho'], r['p']))
cg = R['M4_coupling']['correctness_group']
A("")
A("Correct vs incorrect grouping: correct %.5f vs incorrect %.5f, U = %.0f, p = %.4g, Cliff's δ = %+.3f\n"
  % (cg['correct_mean'], cg['incorrect_mean'], cg['u'], cg['p'], cg['cliffs_delta']))

# ---------- 6 M5 ----------
A("---\n")
A("## 6. M5  main mask vs controls (paired Wilcoxon) (S6 (5) direct paired comparison)\n")
A("| Control | Main-mask median Δ | Control median Δ | W | p | Cliff's δ |")
A("|---|---|---|---|---|---|")
for c, r in R['M5_vs_controls'].items():
    if 'W' not in r:
        A("| `%s` | — | — | — | %s | — |" % (c, r.get('note', '')))
        continue
    A("| `%s` | %+.5f | %+.5f | %.0f | %.4g | %+.3f |"
      % (c, r['median_main'], r['median_ctrl'], r['W'], r['p'], r['cliffs_delta']))
A("")

# ---------- 7 M6 ----------
A("---\n")
A("## 7. M6  fill-strategy robustness (S6 (3) fill strategy)\n")
m6 = R['M6_fill_strategy']
A("| Fill strategy | Median Δ | Mean Δ | p | Effect r | Flip rate | Accuracy after occlusion |")
A("|---|---|---|---|---|---|---|")
for c, r in m6['per_strategy'].items():
    if 'p_value' not in r:
        A("| `%s` | %+.5f | — | %s | — | — | — |" % (c, r.get('median_delta', float('nan')), r.get('note', '')))
        continue
    A("| `%s` | %+.5f | %+.5f | %.3g | %+.3f | %.4f | %.4f |"
      % (c, r['median_delta'], r['mean_delta'], r['p_value'], r['effect_r'],
         R['M2_flip'][c]['flip_rate'], R['M2_flip'][c]['acc_after']))
A("")
A("Pairwise Spearman across the three strategies:")
for k, r in m6['pairwise_rho'].items():
    A("- `%s`  ρ = %.4f (p = %.3g)" % (k, r['rho'], r['p']))
A("")
A("**Direction consistency = %s**\n" % ('yes (all three strategies in the same direction)' if m6['direction_consistent'] else 'no'))

# ---------- 7b random-mask null distribution ----------
A("---\n")
A("## 7·supplement  random-mask null distribution (same-volume floor effect) (S6 (10))\n")
if NULL is None:
    A("> ⏳ this archive does not include the random null-distribution control (that control has been judged retired, see occlusion/README.md §8), so this section is not output.\n")
else:
    rn = NULL['random_null_nonsat']; ra = NULL['random_null_all']
    A("**Motivation**: the C1/C2 controls still carry an anatomical prior (they avoid the MTL, and the sphere centres stay away from the MTL centroid),")
    A("which is not enough to answer 'would any same-volume perturbation produce the same effect'. This section generates N = %d **fully random** masks"
      % NULL['n_random_target'])
    A("(voxels sampled uniformly inside the brain, volume strictly = 31,764, with no anatomical constraint whatsoever), and runs them through the same fill strategy F1 and the same deterministic inference.\n")
    if NULL.get('interrupted'):
        A("> ⚠️ **Run completeness**: target N = %d, actually completed **N = %d**. This batch of computation was aborted twice by the runtime environment at the end of a turn"
          % (NULL['n_random_target'], NULL['n_random']))
        A("> (not a data or model problem; per-mask checkpointing to disk was enabled, so the completed part is preserved intact).")
        A("> The statistics below use N = %d as the effective sample size. **The direction of the conclusion is already fully determined by 44 masks, but the tail quantiles should not be over-interpreted.**\n"
          % NULL['n_random'])
    A("| Distribution | Non-saturated mean Δ (mean ± SD) | Interval |")
    A("|---|---|---|")
    A("| Random-mask null distribution (n=%d) | %+.5f ± %.5f | [%+.5f, %+.5f] |"
      % (NULL['n_random'], rn['mean'], rn['sd'], rn['min'], rn['max']))
    A("| **MTL main mask** | **%+.5f** | — |" % NULL['mtl']['nonsat'])
    A("")
    A("One-sided empirical p-value (MTL > random): **p = %.4f**, z = **%+.2f** (non-saturated convention);"
      % (NULL['empirical_p']['nonsat'], NULL['z']['nonsat']))
    A("all-samples convention p = %.4f, z = %+.2f.\n" % (NULL['empirical_p']['all'], NULL['z']['all']))
    A("**the effect of all %d / %d random masks is larger than MTL, and none is smaller**; the random mean is **%.2f times** that of MTL.\n"
      % (NULL['n_above_ns'], NULL['n_random'], NULL['ratio_ns']))
    A("Flip behaviour of the random masks: %.2f flips to correct, %.2f flips to wrong, **net flips %+.2f** (the main mask nets 0)"
      % (NULL['flip']['random_fix_mean'], NULL['flip']['random_break_mean'],
         NULL['flip']['net_rand_mean']))
    A("→ random perturbation likewise produces two-way flips; **the flips themselves do not indicate 'a directional exclusion of the disease signal'**.\n")

    # ★★★ spatial-coherence diagnosis: why this control does not hold
    sp = NULL.get('spatial')
    A("### ★ Key correction: this random control **does not constitute a volume control of the same nature**\n")
    if sp:
        sm, sr = sp['mtl'], sp['random']
        A("| Mask | Voxels | **Number of connected components** | Bounding box (voxels) | Fill ratio |")
        A("|---|---|---|---|---|")
        A("| MTL | %s | **%d** | %s | %.4f |"
          % ('{:,}'.format(sm['vox']), sm['n_components'], str(tuple(sm['bbox'])), sm['fill_ratio']))
        A("| Random mask | %s | **%s** | %s | %.4f |"
          % ('{:,}'.format(sr['vox']), '{:,}'.format(sr['n_components']),
             str(tuple(sr['bbox'])), sr['fill_ratio']))
        A("")
        A("The random mask consists of **%s isolated single points** (fill ratio only %.4f), scattered across the **whole brain** (bounding box %s);"
          % ('{:,}'.format(sr['n_components']), sr['fill_ratio'], str(tuple(sr['bbox']))))
        A("whereas the MTL has only **%d compact connected blocks** (bounding box %s, fill ratio %.4f, geometric mean edge length %.1f mm)."
          % (sm['n_components'], str(tuple(sm['bbox'])), sm['fill_ratio'], sp['mtl_bbox_geomean']))
        A("")
    A("The two masks apply perturbations of a **different nature**:")
    A("a scattered mask fills the in-brain mean into about %s isolated points, which amounts to **whole-brain salt-and-pepper noise**, and destroys the intensity of local texture"
      % ('{:,}'.format(sp['random']['n_components']) if sp else '30 thousand'))
    A("to a far greater extent than removing one contiguous region does. Therefore 'the effect of the random mask is larger' **only reflects a difference in spatial coherence,")
    A("and cannot disprove that the MTL region contains no diagnostic information**.")
    A("")
    A("★ Correct interpretation:")
    A("- **this control is retired**—it mixes the two variables of 'volume' and 'spatial coherence', so the volume effect cannot be isolated.")
    A("  To establish a rigorous volume floor, one would need **equal-volume and equally compact** random blocks (the C2 random sphere already comes close to this design).")
    A("- among **comparable-in-nature** compact controls, MTL is still significantly stronger than the C1 non-MTL cortex (p = %.3g) and the C2 random sphere (p = %.3g)."
      % (v.get('mtl_vs_c1', {}).get('p', float('nan')), v.get('mtl_vs_c2', {}).get('p', float('nan'))))
    A("- but MTL still has no significant increment over the **hippocampus alone** (p = %.4g); this limitation is unchanged."
      % v.get('mtl_vs_c3', {}).get('p', float('nan')))
    A("")


# ---------- 7c stratified by true class (important) ----------
A("---\n")
A("## 7·supplement ii  stratified by true class: the strongest signal in this experiment (S6 (9))\n")
if PC is None:
    A("> ⏳ `step3c_per_class_stats.py` has not been run yet, so this section is left empty.\n")
else:
    pd_ = PC['per_class_delta']
    pf = PC['per_class_flip']
    A("**Motivation**: the overall net accuracy change of 0 (section 3) masks an asymmetry between classes. In the confusion matrix the AD row")
    A("goes from `[0, 3, 45]` at baseline to `[3, 3, 42]` after occlusion—AD→CN errors appear that were completely absent at baseline.\n")
    A("### Q1  Δp_true stratified by true class (non-saturated subset)\n")
    A("| True class | n (non-saturated/all) | Median Δ | Mean Δ | n⁺/n⁻ | Wilcoxon p | Effect r |")
    A("|---|---|---|---|---|---|---|")
    for cn in ('CN', 'MCI', 'AD'):
        r = pd_[cn]
        A("| **%s** | %d / %d | %+.5f | %+.5f | %d/%d | **%.3g** | %+.3f |"
          % (cn, r['n_nonsat'], r['n_all'], r['median_nonsat'], r['mean_nonsat'],
             r['n_pos'], r['n_neg'], r['p'], r['effect_r']))
    A("")
    kw = pd_['_KW']
    A("Between-class difference (Kruskal-Wallis): **H = %.3f, p = %.3g**\n" % (kw['H'], kw['p']))
    A("Pairwise Mann-Whitney:")
    for cn in ('CN', 'MCI', 'AD'):
        for cn2 in ('CN', 'MCI', 'AD'):
            k = '_pair_%s_%s' % (cn, cn2)
            if k in pd_:
                A("- %s vs %s: U = %.0f, p = %.4g" % (cn, cn2, pd_[k]['U'], pd_[k]['p']))
    A("")
    A("### Q2  change in class-level recall (all-samples convention; flips are unaffected by saturation)\n")
    A("| True class | n | Baseline recall | Occlusion recall | Δ recall | Flip to correct | Flip to wrong | Net |")
    A("|---|---|---|---|---|---|---|---|")
    for cn in ('CN', 'MCI', 'AD'):
        r = pf[cn]
        A("| **%s** | %d | %.4f | %.4f | **%+.4f** | %d | %d | **%+d** |"
          % (cn, r['n'], r['recall_base'], r['recall_occ'], r['recall_delta'],
             r['n_fix'], r['n_break'], r['net']))
    A("")
    ad = PC['ad_specificity']
    A("**AD-class McNemar exact test**: correct→wrong %d cases vs wrong→correct %d cases, p = %.4g → **%s**\n"
      % (ad['ad_break'], ad['ad_fix'], ad['mcnemar_p'],
         'significant' if ad['significant'] else 'not significant'))
    A("All-classes McNemar (correct→wrong %d vs wrong→correct %d): p = %.4g\n" % (ad['all_break'], ad['all_fix'], ad['all_mcnemar_p']))
    A("### Q3  is the drop in AD recall specific to MTL\n")
    A("| Condition | AD recall Δ | CN recall Δ | MCI recall Δ | AD McNemar p |")
    A("|---|---|---|---|---|")
    for cond, r in PC['controls_ad_asymmetry'].items():
        A("| `%s` | %+.4f | %+.4f | %+.4f | %.4g |"
          % (cond, r['ad_delta'], r['cn_delta'], r['mci_delta'], r['mcnemar_p']))
    A("")
    A("★ Interpretation (this section is the most noteworthy part of the whole experiment):\n")
    A("- after stratifying by true class, **the AD group shows the only significant and directionally consistent effect**:")
    A("  non-saturated AD subset n=%d, Δp_true positive in %d cases and negative in **0** cases, Wilcoxon p = %.3g, r = %+.3f."
      % (pd_['AD']['n_nonsat'], pd_['AD']['n_pos'], pd_['AD']['p'], pd_['AD']['effect_r']))
    A("- whereas **neither the CN nor the MCI group is significant** (p = %.3g / %.3g), and the CN mean is even negative."
      % (pd_['CN']['p'], pd_['MCI']['p']))
    A("- the between-class difference is **highly significant** (KW p = %.3g)."
      % kw['p'])
    A("- ⚠️ **but three points must be stated at the same time**:")
    A("  1. the drop in AD recall is only %d cases (%.4f → %.4f), McNemar p = %.4g, **not significant**—"
      % (ad['ad_break'], pf['AD']['recall_base'], pf['AD']['recall_occ'], ad['mcnemar_p']))
    A("     this is a counter-argument of 'insufficient sample size' rather than of 'the effect does not exist', and cannot be argued in reverse.")
    A("  2. this asymmetry **also appears in the controls** (see the table above: the AD recall of F2noise/F3transplant/C2/C3 also drops),")
    A("     so one cannot directly infer that 'MTL is specific to AD discrimination'.")
    A("  3. the stratification is a **post-hoc** analysis and was not pre-registered; it must be labelled explicitly as exploratory.")
    A("")
    A("**Conclusion**: the only positive evidence that stands in this experiment should point to")
    A("the **stratified result** that 'within the AD group the effect is significant and one-directional, while CN/MCI are not',")
    A("and not to the overall Δp_true (the latter is driven by a minority of samples and does not differ significantly from the random control).")
    A("")

# ---------- 8 removed: author-side writing material ----------
# This section originally hard-coded two passages of main-text-oriented prose (overall convention /
# stratified-by-true-class convention). They are author-side writing material and should not be
# emitted with the code. Removed in full by the decision of 2026-09-17.
# The relevant values all remain in the data subsections of this report and can be rechecked item by item from reference_results/.
A("---\n")
A("## 8. File inventory\n")
A("| File | Content |")
A("|---|---|")
A("| `results/occlusion_raw.npz` | per-sample per-condition p_after (raw output of Step 2, 1440 records = 144×10) |")
A("| `results/occlusion_meta.json` | condition table, mask voxel counts, dose levels |")
A("| `results/occlusion_stats.json` | all statistics (M1–M6 + interpretation), both the non-saturated and the all-samples conventions |")
A("| `results/perclass_stats.json` | statistics stratified by true class (step3c) |")
A("| `results/patch_stats.json` | probability flow and the detailed dose-response table (step3d) |")
A("| `results/minimal_pkg_stats.json` | minimal necessary package E1 entropy / E2 transplant / E3 coupling (step3e) |")
A("| `masks/masks_v2.npz`, `masks/masks_v2_meta.json` | v2 masks and volume-convention metadata (step1) |")
A("| `figures/fig_occlusion_*.png` | 6 figures (step4) |")
A("| `step0_baseline_p0.py` | baseline inference (deterministic convention, acc = 0.8681) |")
A("| `step1_build_masks.py` | v2 mask construction (with the triple assertion on voxel count / ratio / centroid) |")
A("| `step2_occlusion_sweep.py` | occlusion inference (10 conditions × 144 cases) |")
A("| `step3_occlusion_stats.py` | statistical analysis (with saturation stratification and residual robustness) |")
A("| `step3c_per_class_stats.py` | stratification by true class + class-level directionality |")
A("| `step3d_patch_flow.py` | probability flow and the detailed dose-response table |")
A("| `step3e_minimal_package.py` | minimal necessary package (E1 entropy / E2 transplant / E3 coupling) |")
A("| `step4_figures.py` | plotting |")
A("| `step5_report.py` | this report |")
A("| `verify_occlusion.py` | archive self-verification (read-only recomputation + 63 paper numeric anchors) |")
A("| `reference_results/` | frozen authoritative artifacts + `MANIFEST.md5` (the reference basis for every value in this report) |")
A("| `manuscript_checks/` | numeric traceability table (87 items) and its row-by-row checking script |")


txt = "\n".join(L)
open(OUT.parent / 'occlusion_report.md', 'w', encoding='utf-8').write(txt)
print(txt)
print("\nsaved:", OUT.parent / 'occlusion_report.md')
