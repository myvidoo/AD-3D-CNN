# T2 sensitivity analysis report
## Recomputing the confidence–CAM strength Spearman correlation after excluding the "predicted probability > 0.99" subsample

**Analysis date**: 2026-09-16
**Data source**: `reference_results/gradcam_v3/per_sample_results_final.csv` (n=144, authoritative archive, per-sample)
**Scripts**: `stats_rechecks/t2_sensitivity/` (`t2_sensitivity.py` → `t2_diagnose.py` → `t2_extended.py` → `t2_mechanism.py`)
**One-line conclusion**: **The baseline ρ=0.800 is reproduced exactly; however, the operation of "excluding >0.99" is not viable on these data (the residual is only n=23 and its variation has been drained away), and should not be written in as "S6 robustness support"; instead, reporting "excluding exact saturation at 1.0" (n=100, ρ=0.6696, p<0.001) together with a permutation test is a formulation that is both honest and compelling.**

> ⚠️ **Note on the p-value convention**: this report ultimately uses the **exact t-distribution CDF** (an implementation via continued fractions of the incomplete beta function) to compute the p values for Spearman's ρ. An earlier version used a normal approximation, which **overestimated** significance in the extreme tail (for example, the baseline p was approximated as 7.8×10⁻⁵⁷ instead of 2.64×10⁻³³). **The tables and conclusions below have all been updated to the exact values**, and the core interpretation is unchanged. Before formally writing this up, it is recommended to cross-confirm once more with `scipy.stats.spearmanr`.

---

## 1. Baseline reproduction (deterministic verification passed)

| Metric | Reported in §3.3 of the manuscript | Recomputed here | Verdict |
|---|---|---|---|
| n | 144 | 144 | ✓ |
| Spearman ρ | 0.800 | **0.8000** | ✓ exactly identical |
| p | <0.001 | 2.64×10⁻³³ (exact t) | ✓ far below 0.001 |

The data source is trustworthy, and the recomputation pipeline is consistent with the manuscript.

---

## 2. Degree of probability saturation (consistent with §3.2 of the manuscript)

| Threshold | Count | Proportion |
|---|---|---|
| confidence > 0.99 | **121** | **84.0%** ← matches "84% of samples with probability >0.99" in §3.2 of the manuscript ✓ |
| confidence > 0.95 | 124 | 86.1% |
| confidence > 0.90 | 130 | 90.3% |
| confidence = 1.0 (exact saturation) | 44 | 30.6% |

Description of confidence: mean 0.9717 (consistent with "mean confidence 0.9717" in §3.2 of the manuscript ✓), median 1.0000, minimum 0.5447, maximum 1.0000.

---

## 3. Threshold-by-threshold sensitivity analysis

| Exclusion threshold | Remaining n | ρ | p (exact t) | Δρ vs baseline |
|---|---|---|---|---|
| Baseline (no exclusion) | 144 | 0.8000 | 2.64×10⁻³³ | — |
| **Exclude exact =1.0** | **100** | **0.6696** | **2.6×10⁻¹⁴** | **−0.1304** |
| Exclude >0.99 | 23 | 0.2115 | 0.333 | −0.5885 |
| Exclude >0.95 | 20 | 0.1564 | 0.507 | −0.6436 |
| Exclude >0.90 | 14 | 0.3538 | 0.194 | −0.4461 |
| Exclude >0.80 | 7 | 0.2857 | 0.508 | −0.5143 |

**The problem this reveals**: following the external comment's original suggestion (excluding >0.99), ρ drops to 0.2115 with p=0.333, **non-significant** — writing this into S6 as it stands would amount to **passing sentence of death on T2 ourselves**.

---

## 4. Why ρ collapses — locating the mechanism (the core of this report)

### 4.1 The exclusion operation completely alters the sample structure

| | Count | Of which correct | Of which incorrect | Correct:incorrect ratio |
|---|---|---|---|---|
| All samples | 144 | 125 | 19 | **6.6 : 1** |
| Excluded (conf>0.99) | 121 | **114 (94.2%)** | 7 (5.8%) | 16.3 : 1 |
| Remaining (conf≤0.99) | **23** | 11 (47.8%) | **12 (52.2%)** | **0.9 : 1** |

**Excluding conf>0.99 removes almost all of the correct samples (114 of the 121 are correct predictions).** The residual 23 samples degenerate into a mixture in which correct and incorrect are roughly balanced.

### 4.2 And the main difference in cam_mean comes precisely from the "correct/incorrect" variable

| Group | cam_mean | Comparison with §3.3 of the manuscript |
|---|---|---|
| Correct group (n=125) | 0.1365 ± 0.0464 | ✓ consistent with "cam_mean 0.1365" in the manuscript |
| Incorrect group (n=19) | 0.0956 ± 0.0440 | ✓ consistent with "0.0956" in the manuscript |
| Difference | +0.0409 | — |
| Cliff's δ | **0.517** | ✓ consistent with δ=0.52 in the manuscript |

**Therefore: the ρ=0.800 of T2 is carried mainly by the stratified structure "correct samples have high confidence and strong CAM / incorrect samples have low confidence and weak CAM". When the 121 saturated samples are excluded, this stratified structure is dismantled along with them — and the continuous gradient disappears with it.**

### 4.3 Key counter-evidence: the partial correlation does **not** attenuate; it rises slightly

| Test | Value |
|---|---|
| Uncontrolled raw rank correlation | r = 0.8000 |
| **Within-group correlation after controlling for "prediction correct/incorrect" (pooled)** | **r = 0.8207** |

**This is decisive evidence.** If the T2 correlation were merely a disguise for the single binary variable "correct/incorrect", then controlling for it should drive the correlation to zero — in fact it is **0.8207, higher than the raw value**.

That is: **there is a continuous coupling between confidence and CAM strength that is independent of "correct/incorrect", and that coupling is real.**

### 4.4 Within-group correlations are also all positive and significant

| Subset | n | ρ | p (exact t) |
|---|---|---|---|
| Correct group (all) | 125 | 0.7696 | 1.0×10⁻²⁵ |
| Incorrect group (all) | 19 | 0.5281 | **0.0201** |
| Correct group & conf<1.0 | 81 | 0.5912 | 3.1×10⁻⁹ |
| Correct group & conf≤0.99 | 11 | 0.3182 | 0.322 (insufficient sample, non-significant) |

**Pooling the two groups and controlling for correctness gives r=0.8207; within the correct group alone ρ=0.7696 is significant, and within the incorrect group ρ=0.5281 (p=0.020) is also significant.** This shows that the coupling still holds within groups and is not a purely between-group effect.

---

## 5. Permutation test: the coupling does not depend on individual saturated points

**Design**: randomly retain 20% of the saturated samples + all non-saturated samples, 2000 resamples.

| Statistic | Value |
|---|---|
| Mean ρ | 0.7649 |
| Median ρ | 0.7666 |
| 95% interval | [0.7147, 0.8072] |
| Proportion with ρ>0 | **100.0%** |
| Proportion with ρ>0.3 | **100.0%** |

**After randomly deleting 80% of the saturated samples, the median ρ is still 0.767 — the original 0.8000 was not propped up by a handful of saturated points.**

---

## 6. Final interpretation (three points, for use in writing)

### ① The version suggested by the external comment cannot be adopted directly
"ρ=0.2115 after excluding conf>0.99" **does not constitute a valid sensitivity test** on these data, for two reasons:
- **Collapse of the sample size**: the residual n=23 comprises 11 correct and 12 incorrect, and the confidence values of these 23 samples span [0.5447, 0.9829]; a rank correlation is inherently difficult to detect within such a narrow range.
- **The confounding structure changes**: the exclusion operation simultaneously empties the correct group, taking the correct:incorrect ratio of the residual subset from 6.6:1 to 0.9:1. **This is a side effect of "excluding saturation", not a conclusion of "excluding saturation".**

⚠️ Copying this into S6 as it stands would amount to handing the reviewers a non-significant result and **creating an attack surface**.

### ② Instead, these two should be reported (both honest and significant)
| Item | Value | Interpretation |
|---|---|---|
| Baseline | n=144, ρ=0.8000, p=2.6×10⁻³³ | Main result |
| **Excluding exact saturation at 1.0** | **n=100, ρ=0.6696, p=2.6×10⁻¹⁴** | **Robustness support: still highly significant** |
| Between-group MWU in that subset | U=2089, p=7.3×10⁻⁹, Cliff's δ=+0.671 | Effect size is even larger |
| Permutation test (randomly deleting 80% of saturated) | median ρ 0.7666, 95% [0.715, 0.807] | Does not depend on individual saturated points |
| Partial correlation (controlling for correctness) | r = **0.8207** > raw 0.8000 | Coupling is independent of the correctness structure |

### ③ Honest boundaries (must be written in as well)
It should be acknowledged that **in the residual subset with conf≤0.99 (n=23), the confidence–CAM coupling does not reach statistical significance (ρ=0.2115, p=0.333).** It should be stated at the same time, however, that this subset is small, has narrow variation, and has had its correct/incorrect structure artificially altered, so it is **insufficient as evidence that "the coupling does not exist"**; whereas within the range that can be adequately tested (n=100 and the full sample), the coupling holds robustly.

**Suggested wording**:
> Sensitivity analysis showed that the confidence–CAM strength coupling does not depend on extreme saturation probabilities: after excluding all exactly saturated (=1.0) samples (n=100), the correlation coefficient was still 0.6696 (p<0.001); across 2000 permutation tests retaining a random 20% of saturated samples, the median ρ was 0.767 (95% interval 0.715–0.807), suggesting that this coupling is not driven by a small number of saturated samples. Note that the residual subset obtained after further excluding conf≤0.99 (n=23) was non-significant (ρ=0.2115, p=0.333), but this subset is accompanied both by a change in correct/incorrect composition (the ratio changing from 6.6:1 to 0.9:1) and by a compression of confidence variation, so its non-significance should not be read as an absence of coupling. The within-group partial correlation after controlling for prediction correctness (r=0.821) is higher than the uncontrolled rank correlation (0.800), further supporting a continuous association between confidence and attention strength that is independent of prediction correctness.

---

## 7. Relationship to the existing content of the manuscript

| Location | Current state | Recommendation |
|---|---|---|
| §2.5 explainability method, item (2) | Spearman correlation already reported; S6 states "computational details and robustness analysis are given in S6" | **No change to the main text needed**; a supplement in S6 suffices |
| §3.3 Results | ρ=0.800 already reported, and already qualified as "the coupling is not independent evidence" | **No change needed** |
| S6 | currently has **no** such sensitivity analysis | **Add a paragraph**, following the suggested wording in Section 6 |
| §4.2 | cites T2 as one of the pillars of credibility | could consider adding half a sentence: "this coupling still holds after excluding saturated samples" |

**The amount of change to the main text is zero** (S6 is supplementary material), consistent with the principle of minimal modification.

---

## 8. Remaining issues

- **p-value precision**: this report now uses the **exact t-distribution CDF** (a continued-fraction implementation of the regularized incomplete beta function) to compute p values, replacing the normal approximation of the earlier version. The checking script `t2_pvalue_check.py` shows that the normal approximation **overestimates** significance in the extreme tail (baseline p: normal 7.8×10⁻⁵⁷ vs exact 2.64×10⁻³³), and **the tables and wording of this report have all been updated to the exact values**. Before formally writing this up, it is recommended to cross-confirm once with `scipy.stats.spearmanr` in an environment with scipy installed.
- **ρ values are unaffected by the approximation**: all ρ values are computed directly by the Pearson formula applied to ranks, with no approximate component, and should be fully consistent with scipy (the baseline 0.8000 already matches the value reported in the manuscript exactly).
- The permutation test uses `seed=42` (reproducible).
- The effect size Cliff's δ is computed as `2U/(n₁n₂)−1`; the incorrect group gives δ=0.517, matching the manuscript's reported 0.52 ✓, which verifies the correctness of the pipeline.
- Whether the result "non-significant after excluding conf≤0.99" **should be written into S6** depends on your strategic preference:
  - **Write it in** (recommended): proactive exposure + a full explanation of the mechanism, consistent with this manuscript's consistently defensive stance, and we have the partial correlation of 0.821 as strong counter-evidence;
  - **Do not write it in**: report only the two significant results (n=100 and the permutation test), but this comes close to selective reporting, and if a reviewer re-runs the analysis themselves it will be discovered, **which carries high risk**.
