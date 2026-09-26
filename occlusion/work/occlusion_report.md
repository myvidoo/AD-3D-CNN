# Occlusion experiment report (MTL causality evidence)

**Date**: 2026-09-16  **Model**: Run 128 (3D DenseNet-169 + axial spatial gating, 5-fold equal-weight ensemble)
**Samples**: ADNI fixed test set n = 144  **Main mask**: MTL (same convention as the paper's T3, 31,764 voxels = 1.830%)
**Inference convention**: deterministic (`cudnn.deterministic=True` + `benchmark=False` + `use_deterministic_algorithms(warn_only=True)`), bitwise identical to the Step 0 baseline (max|Δp| = 0.00e+00)


**Correspondence with supplementary material S6** (S6 = the methodological supplement on interpretability analysis and robustness):

| Supplementary S6 item | Corresponding subsection of this report (if not covered, it points to the artifact) |
|---|---|
| (1) Methods and reproducibility controls | not contained in this report; see `rerun_diag.json`, `rerun_consistency.json`, `probe_vs_data.json`, `repeat_check.json` |
| (2) Masks and controls | the "Main mask" convention line in the header; for the authoritative volume values see `masks_v2_meta.json` |
| (3) Fill strategy | §7  M6  fill-strategy robustness |
| (4) Main results and multiple comparisons | §0  conventions and limitations, §2  M1  Δp_true paired test, §3  M2  prediction flip rate |
| (5) Direct paired comparison | §6  M5  main mask vs controls |
| (6) Dose-response | §4  M3  dose-response |
| (7) Disease-axis coordinates and two-way transplant | not contained in this report; see `minimal_pkg_stats.json` (E2 donor reconstruction) |
| (8) Prediction entropy | not contained in this report; see `minimal_pkg_stats.json` (E1 entropy) |
| (9) Stratification and per-sample coupling (exploratory) | §5  M4  sample-level coupling, §7·supplement ii  stratified by true class |
| (10) On the random null-distribution control | §7·supplement  random-mask null distribution (this control is retired, see `occlusion/README.md` §8) |
| (11) Inferences that should not be drawn (statement of limitations) | §0  conventions and limitations, §1  interpretation |

---

## 0. Read this section first: three convention limitations you must know up front (S6 (4) defining the main convention · (11) statement of limitations)

> This section is a **disclosure of limitations**, not a conclusion. Ignoring these three items will overstate the strength of the conclusions.

| # | Limitation | Fact | Consequence |
|---|---|---|---|
| 1 | **softmax saturation** | the top-1 probability of 64/144 cases is clamped to exactly 1.0 under float64
| | | for these samples Δp_true is swallowed at the bit level, **and all 29 strictly zero differences come from saturated samples** | the main analysis is restricted to the non-saturated subset n=80 |
| 2 | **two-way flips** | after occlusion with the main mask, wrong→correct 5 cases, correct→wrong 5 cases
| | | **net accuracy change = +0.0000 (i.e. 0)**
| | | occlusion is **not** a one-way removal of the disease signal, but a two-way perturbation | one cannot argue from 'a drop in accuracy' |
| 3 | **the effect size is very small** | for the main mask, non-saturated median Δ = +9.08e-05, mean = +0.01791 (mean/median ≈ 197×)
| | | the effect is driven by a minority of samples | the median is almost uninformative, and the statistical significance rests on the sign counts of the Wilcoxon test |

---

## 1. Interpretation (S6 (4) · (11))

```
A- —— MTL occlusion significantly lowers the true-class probability and is significantly stronger than the volume-matched / random-sphere controls,
      but **MTL has no significant advantage over the hippocampus alone** (p=0.19)
      → supports “the medial temporal lobe carries diagnostic information”, but the evidence points to **the hippocampus itself** rather than the whole MTL
      → if the paper needs to claim MTL (rather than hippocampal) specificity, it must be cautious
```

Basis for the interpretation (convention: non-saturated subset n=80):
- significantly lower (main mask): `yes` (p = 0.008108)
- significantly stronger than the C1 volume-matched control: `yes` (p = 0.03566)
- significantly stronger than the C2 random-sphere control: `yes` (p = 0.0128)
- **significantly stronger than the C3 hippocampus-alone control: `no` (p = 0.1904) ← this is the single most important row**

---

## 2. M1  Δp_true paired test (main conclusion metric) (S6 (4) main results and multiple comparisons)

`Δp_true = p0[true] − p_occ[true]`, **a positive value means that occlusion lowers the true-class probability**.
The table gives two columns side by side: **non-saturated n=80 (main convention)** and **all samples n=144 (for reference only)**.

| Condition | Mask voxels | Non-saturated median Δ | Non-saturated mean Δ | n⁺/n⁻ | p | r | All-samples p |
|---|---|---|---|---|---|---|---|
| `MTL_d100_F1const` | 31,764 | +9.08e-05 | +0.01791 | 53/27 | 0.00811 | +0.296 | 0.00013 |
| `MTL_d80_F1const` | 25,411 | +1.35e-04 | +0.01404 | 54/26 | 0.00413 | +0.321 | 0.00025 |
| `MTL_d60_F1const` | 19,058 | +2.69e-05 | +0.01342 | 50/30 | 0.0415 | +0.228 | 0.00636 |
| `MTL_d40_F1const` | 12,706 | +8.05e-06 | +0.01492 | 52/28 | 0.0604 | +0.210 | 0.00922 |
| `MTL_d25_F1const` | 7,941 | +1.55e-06 | +0.00008 | 49/30 | 0.283 | +0.121 | 0.21 |
| `MTL_d100_F2noise` | 31,764 | +5.46e-05 | +0.00713 | 51/29 | 0.0538 | +0.216 | 0.00173 |
| `MTL_d100_F3transplant` | 31,764 | +1.61e-04 | +0.02855 | 59/21 | 0.00232 | +0.341 | 1.07e-05 |
| `C1_cort_23_F1const` | 32,849 | -8.34e-07 | -0.01104 | 33/46 | 0.458 | +0.084 | 0.417 |
| `C2_ball_F1const` | 31,910 | +1.12e-05 | +0.02357 | 46/34 | 0.0254 | +0.250 | 0.0262 |
| `C3_hipp_F1const` | 11,263 | +2.47e-05 | +0.02385 | 47/33 | 0.0059 | +0.308 | 0.00219 |

> `n⁺/n⁻` = the number of samples with a positive/negative Δp_true in the non-saturated subset. `r` = paired Wilcoxon effect size (Z/√N).

> ⚠️ **the median is of order +9.08e-05 and the mean is of order +0.0179**, roughly a factor of 197 apart:
> the effect is highly right-skewed and the mean is propped up by a few large-effect samples; **any citation should give the median first, or state the shape of the distribution explicitly**.

---

## 3. M2  prediction flip rate (all-samples convention; flips are unaffected by saturation) (S6 (4))

| Condition | Flip rate | Flip to correct (wrong→correct) | Flip to wrong (correct→wrong) | **Net flips** | Accuracy after occlusion | Relative to baseline |
|---|---|---|---|---|---|---|
| **Baseline** | — | — | — | — | 0.8681 | — |
| `MTL_d100_F1const` | 0.0694 | 5 | 5 | **+0** | 0.8681 | +0.0000 |
| `MTL_d80_F1const` | 0.0625 | 4 | 5 | **-1** | 0.8611 | -0.0069 |
| `MTL_d60_F1const` | 0.0556 | 3 | 5 | **-2** | 0.8542 | -0.0139 |
| `MTL_d40_F1const` | 0.0208 | 0 | 3 | **-3** | 0.8472 | -0.0208 |
| `MTL_d25_F1const` | 0.0069 | 0 | 1 | **-1** | 0.8611 | -0.0069 |
| `MTL_d100_F2noise` | 0.0486 | 3 | 4 | **-1** | 0.8611 | -0.0069 |
| `MTL_d100_F3transplant` | 0.0833 | 4 | 8 | **-4** | 0.8403 | -0.0278 |
| `C1_cort_23_F1const` | 0.0069 | 1 | 0 | **+1** | 0.8750 | +0.0069 |
| `C2_ball_F1const` | 0.0347 | 1 | 4 | **-3** | 0.8472 | -0.0208 |
| `C3_hipp_F1const` | 0.0486 | 3 | 4 | **-1** | 0.8611 | -0.0069 |

> ★ the net flips of the main mask MTL_d100 = **+0**, and the net accuracy change = **+0.0000**.
> this shows that occlusion is **not** the one-way process of 'taking the disease signal away → more errors'; instead it pushes the
> samples near the decision boundary to both sides; arguing from accuracy alone cannot support a causal claim.

---

## 4. M3  dose-response (S6 (6))

| Level | Voxels | Share of the full MTL | Median Δp_true | Mean Δp_true |
|---|---|---|---|---|
| `MTL_d25_F1const` | 7,941 | 25.0% | +0.00000 | +0.00008 |
| `MTL_d40_F1const` | 12,706 | 40.0% | +0.00001 | +0.01492 |
| `MTL_d60_F1const` | 19,058 | 60.0% | +0.00003 | +0.01342 |
| `MTL_d80_F1const` | 25,411 | 80.0% | +0.00013 | +0.01404 |
| `MTL_d100_F1const` | 31,764 | 100.0% | +0.00009 | +0.01791 |

**Spearman(voxels, median Δp_true)** = ρ 0.9000, p = 0.03739; **Spearman(voxels, mean Δp_true)** = ρ 0.7000, p = 0.1881

> Note: `d100 vs d25` paired Wilcoxon W = 1081, p = 0.009732; in this round the 5 levels are the basic sampling sites and **no strict volume-gradient design was used** (the EDT distance is quantised to integers, and after switching to rank selection the volumes of the levels are not evenly spaced), so the dose-response is only suitable for a trend description.

---

## 5. M4  sample-level coupling (Δp_true vs baseline metrics) (S6 (9))

| Baseline metric | Spearman ρ | p |
|---|---|---|
| baseline confidence | -0.0709 | 0.5323 |
| cam_mean | -0.1327 | 0.2406 |
| MTL mass fraction | +0.0118 | 0.917 |
| hippocampus mass fraction | +0.0142 | 0.9008 |

Correct vs incorrect grouping: correct 0.07064 vs incorrect -0.16371, U = 872, p = 0.0003038, Cliff's δ = +0.563

---

## 6. M5  main mask vs controls (paired Wilcoxon) (S6 (5) direct paired comparison)

| Control | Main-mask median Δ | Control median Δ | W | p | Cliff's δ |
|---|---|---|---|---|---|
| `MTL_d80_F1const` | +0.00009 | +0.00013 | 1322 | 0.1529 | +0.008 |
| `MTL_d60_F1const` | +0.00009 | +0.00003 | 1109 | 0.02135 | +0.079 |
| `MTL_d40_F1const` | +0.00009 | +0.00001 | 1070 | 0.0084 | +0.124 |
| `MTL_d25_F1const` | +0.00009 | +0.00000 | 1081 | 0.009732 | +0.196 |
| `MTL_d100_F2noise` | +0.00009 | +0.00005 | 1200 | 0.04422 | +0.080 |
| `MTL_d100_F3transplant` | +0.00009 | +0.00016 | 1619 | 0.9962 | -0.023 |
| `C1_cort_23_F1const` | +0.00009 | -0.00000 | 1182 | 0.03566 | +0.257 |
| `C2_ball_F1const` | +0.00009 | +0.00001 | 1101 | 0.0128 | +0.085 |
| `C3_hipp_F1const` | +0.00009 | +0.00002 | 1347 | 0.1904 | +0.023 |

---

## 7. M6  fill-strategy robustness (S6 (3) fill strategy)

| Fill strategy | Median Δ | Mean Δ | p | Effect r | Flip rate | Accuracy after occlusion |
|---|---|---|---|---|---|---|
| `MTL_d100_F1const` | +0.00009 | +0.01791 | 0.00811 | +0.296 | 0.0694 | 0.8681 |
| `MTL_d100_F2noise` | +0.00005 | +0.00713 | 0.0538 | +0.216 | 0.0486 | 0.8611 |
| `MTL_d100_F3transplant` | +0.00016 | +0.02855 | 0.00232 | +0.341 | 0.0833 | 0.8403 |

Pairwise Spearman across the three strategies:
- `MTL_d100_F1const|MTL_d100_F2noise`  ρ = 0.2844 (p = 0.0106)
- `MTL_d100_F1const|MTL_d100_F3transplant`  ρ = 0.4789 (p = 7.01e-06)
- `MTL_d100_F2noise|MTL_d100_F3transplant`  ρ = 0.3224 (p = 0.00354)

**Direction consistency = yes (all three strategies in the same direction)**

---

## 7·supplement  random-mask null distribution (same-volume floor effect) (S6 (10))

> ⏳ this archive does not include the random null-distribution control (that control has been judged retired, see occlusion/README.md §8), so this section is not output.

---

## 7·supplement ii  stratified by true class: the strongest signal in this experiment (S6 (9))

**Motivation**: the overall net accuracy change of 0 (section 3) masks an asymmetry between classes. In the confusion matrix the AD row
goes from `[0, 3, 45]` at baseline to `[3, 3, 42]` after occlusion—AD→CN errors appear that were completely absent at baseline.

### Q1  Δp_true stratified by true class (non-saturated subset)

| True class | n (non-saturated/all) | Median Δ | Mean Δ | n⁺/n⁻ | Wilcoxon p | Effect r |
|---|---|---|---|---|---|---|
| **CN** | 24 / 48 | -0.00001 | -0.05824 | 10/14 | **0.101** | +0.335 |
| **MCI** | 30 / 48 | +0.00001 | -0.02143 | 17/13 | **0.612** | +0.093 |
| **AD** | 26 / 48 | +0.00739 | +0.13359 | 26/0 | **2.98e-08** | +1.087 |

Between-class difference (Kruskal-Wallis): **H = 24.439, p = 4.93e-06**

Pairwise Mann-Whitney:
- CN vs MCI: U = 265, p = 0.09996
- CN vs AD: U = 40, p = 1.349e-07
- MCI vs AD: U = 223, p = 0.006231

### Q2  change in class-level recall (all-samples convention; flips are unaffected by saturation)

| True class | n | Baseline recall | Occlusion recall | Δ recall | Flip to correct | Flip to wrong | Net |
|---|---|---|---|---|---|---|---|
| **CN** | 48 | 0.8542 | 0.8958 | **+0.0417** | 2 | 0 | **+2** |
| **MCI** | 48 | 0.8125 | 0.8333 | **+0.0208** | 3 | 2 | **+1** |
| **AD** | 48 | 0.9375 | 0.8750 | **-0.0625** | 0 | 3 | **-3** |

**AD-class McNemar exact test**: correct→wrong 3 cases vs wrong→correct 0 cases, p = 0.25 → **not significant**

All-classes McNemar (correct→wrong 5 vs wrong→correct 5): p = 1

### Q3  is the drop in AD recall specific to MTL

| Condition | AD recall Δ | CN recall Δ | MCI recall Δ | AD McNemar p |
|---|---|---|---|---|
| `MTL_d100_F1const` | -0.0625 | +0.0417 | +0.0208 | 0.25 |
| `MTL_d80_F1const` | -0.0625 | +0.0208 | +0.0208 | 0.25 |
| `MTL_d60_F1const` | -0.0625 | +0.0000 | +0.0208 | 0.25 |
| `MTL_d40_F1const` | -0.0208 | +0.0000 | -0.0417 | 1 |
| `MTL_d25_F1const` | -0.0208 | +0.0000 | +0.0000 | 1 |
| `MTL_d100_F2noise` | -0.0625 | +0.0000 | +0.0417 | 0.25 |
| `MTL_d100_F3transplant` | -0.1042 | +0.0000 | +0.0208 | 0.0625 |
| `C1_cort_23_F1const` | +0.0208 | +0.0000 | +0.0000 | 1 |
| `C2_ball_F1const` | -0.0417 | +0.0208 | -0.0417 | 0.5 |
| `C3_hipp_F1const` | -0.0417 | +0.0208 | +0.0000 | 0.5 |

★ Interpretation (this section is the most noteworthy part of the whole experiment):

- after stratifying by true class, **the AD group shows the only significant and directionally consistent effect**:
  non-saturated AD subset n=26, Δp_true positive in 26 cases and negative in **0** cases, Wilcoxon p = 2.98e-08, r = +1.087.
- whereas **neither the CN nor the MCI group is significant** (p = 0.101 / 0.612), and the CN mean is even negative.
- the between-class difference is **highly significant** (KW p = 4.93e-06).
- ⚠️ **but three points must be stated at the same time**:
  1. the drop in AD recall is only 3 cases (0.9375 → 0.8750), McNemar p = 0.25, **not significant**—
     this is a counter-argument of 'insufficient sample size' rather than of 'the effect does not exist', and cannot be argued in reverse.
  2. this asymmetry **also appears in the controls** (see the table above: the AD recall of F2noise/F3transplant/C2/C3 also drops),
     so one cannot directly infer that 'MTL is specific to AD discrimination'.
  3. the stratification is a **post-hoc** analysis and was not pre-registered; it must be labelled explicitly as exploratory.

**Conclusion**: the only positive evidence that stands in this experiment should point to
the **stratified result** that 'within the AD group the effect is significant and one-directional, while CN/MCI are not',
and not to the overall Δp_true (the latter is driven by a minority of samples and does not differ significantly from the random control).

---

## 8. File inventory

| File | Content |
|---|---|
| `results/occlusion_raw.npz` | per-sample per-condition p_after (raw output of Step 2, 1440 records = 144×10) |
| `results/occlusion_meta.json` | condition table, mask voxel counts, dose levels |
| `results/occlusion_stats.json` | all statistics (M1–M6 + interpretation), both the non-saturated and the all-samples conventions |
| `results/perclass_stats.json` | statistics stratified by true class (step3c) |
| `results/patch_stats.json` | probability flow and the detailed dose-response table (step3d) |
| `results/minimal_pkg_stats.json` | minimal necessary package E1 entropy / E2 transplant / E3 coupling (step3e) |
| `masks/masks_v2.npz`, `masks/masks_v2_meta.json` | v2 masks and volume-convention metadata (step1) |
| `figures/fig_occlusion_*.png` | 6 figures (step4) |
| `step0_baseline_p0.py` | baseline inference (deterministic convention, acc = 0.8681) |
| `step1_build_masks.py` | v2 mask construction (with the triple assertion on voxel count / ratio / centroid) |
| `step2_occlusion_sweep.py` | occlusion inference (10 conditions × 144 cases) |
| `step3_occlusion_stats.py` | statistical analysis (with saturation stratification and residual robustness) |
| `step3c_per_class_stats.py` | stratification by true class + class-level directionality |
| `step3d_patch_flow.py` | probability flow and the detailed dose-response table |
| `step3e_minimal_package.py` | minimal necessary package (E1 entropy / E2 transplant / E3 coupling) |
| `step4_figures.py` | plotting |
| `step5_report.py` | this report |
| `verify_occlusion.py` | archive self-verification (read-only recomputation + 63 paper numeric anchors) |
| `reference_results/` | frozen authoritative artifacts + `MANIFEST.md5` (the reference basis for every value in this report) |
| `manuscript_checks/` | numeric traceability table (87 items) and its row-by-row checking script |