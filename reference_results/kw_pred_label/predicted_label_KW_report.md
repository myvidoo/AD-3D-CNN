# Kruskal-Wallis re-run using predicted labels — analysis report

**Analysis date**: 2026-09-16
**Data source**: `reference_results/gradcam_v3/per_sample_results_final.csv` (n=144, per-sample)
**Scripts**: `stats_rechecks/kw_pred_label/` (`pred_label_kw.py` → `two_way_decomp.py` → `collinearity.py`)
**One-line conclusion**: **The signal on the predicted side is real and slightly stronger than on the true side (H 13.55 vs 11.14), and the post-hoc conclusions are entirely consistent (again only CN vs MCI is significant, and more strongly so); however, the two factors are highly collinear (Cramér's V=0.81), so their contributions cannot be separated additively. The current wording of the manuscript is robust and requires no change; if this is written into S6, the two KW results should be reported side by side together with a note on the collinearity.**

---

## 1. Baseline reproduction (pipeline correctness check)

| Metric | Reported in §3.3 of the manuscript | Recomputed here | Verdict |
|---|---|---|---|
| Kruskal-Wallis H | 11.14 | **11.1375** | ✓ |
| p | 0.0038 | **0.003815** | ✓ exactly identical |
| Dunn CN vs MCI corrected p | 0.0026 | **0.002628** | ✓ exactly identical |
| Group means | CN 2.24% / MCI 2.50% / AD 2.37% | 2.2385 / 2.5031 / 2.3745 | ✓ |

The pipeline reproduces correctly, and when **only the "absolute values" are tested** the value 0.002628 matches the manuscript's 0.0026 exactly (indicating that the manuscript used a two-tailed Dunn test with Bonferroni×3).

---

## 2. Core result: KW re-run using predicted labels

### 2.1 Contingency table (true × predicted)

| True \ Predicted | pred CN | pred MCI | pred AD | Total |
|---|---|---|---|---|
| true CN | 41 | 5 | 2 | 48 |
| true MCI | 5 | 39 | 4 | 48 |
| true AD | 0 | 3 | 45 | 48 |
| **Total** | **46** | **47** | **51** | **144** |

### 2.2 The two KW tests side by side

| Grouping | H | p | ε² | η² | Group means |
|---|---|---|---|---|---|
| By **true** label (manuscript T4) | 11.14 | 0.0038 | 0.0779 | 0.0648 | CN 2.2385 / MCI 2.5031 / AD 2.3745 |
| By **predicted** label (this analysis) | **13.55** | **0.0011** | **0.0947** | 0.0819 | CN 2.2428 / MCI 2.5309 / AD 2.3424 |

**The signal is stronger on the predicted side than on the true side** (H 13.55 > 11.14, p 0.0011 < 0.0038).

### 2.3 Post-hoc tests (Dunn + Bonferroni×3)

| Contrast | By true label | By predicted label |
|---|---|---|
| CN vs MCI | z=3.328, p_adj=**0.0026** ✅ significant | z=3.650, p_adj=**0.00079** ✅ significant |
| CN vs AD | z=1.884, p_adj=0.179 n.s. | z=1.475, p_adj=0.420 n.s. |
| MCI vs AD | z=1.444, p_adj=0.447 n.s. | z=2.260, p_adj=0.0714 n.s. (marginal) |

**Key point**: under both grouping schemes **only CN vs MCI survives**, and it is more significant on the predicted side. **The pattern of conclusions is exactly the same; only the strength increases slightly.**

---

## 3. Two-way decomposition: why an additive split cannot be performed

### 3.1 The two factors are highly collinear

| Metric | Value |
|---|---|
| Agreement rate (= accuracy) | 86.81% (125/144) |
| **Cramér's V** | **0.8054** |
| χ² (df=4) | 186.81, p=2.6×10⁻³⁹ |

**V=0.81 is a strong association.** This directly explains why the standard additive decomposition fails — I measured **SS_inter = −5.26% (a negative interaction term)**, which is the classic signal of collinearity.

### 3.2 Rank-space variance contributions (directional reference only; not to be read additively)

| Source | Share of total |
|---|---|
| True label | 7.79% |
| Predicted label | 9.47% |
| Interaction | −5.26% (negative value = collinear) |
| Within group | 88.00% |

⚠️ **This must not be written as "7.8% for true + 9.5% for predicted"** — the negative interaction term shows that the two factors share one and the same pool of variance, so an additive split is invalid here.

### 3.3 Comparison against "correctness"

| Grouping variable | k | Share of total |
|---|---|---|
| True label | 3 | 7.79% |
| Predicted label | 3 | 9.47% |
| **Correctness (correct/incorrect)** | 2 | **0.06%** |

Correctness explains **essentially none** of the variance in MTL quality scores — this is **entirely self-consistent** with the null result in the manuscript's T5 (no difference in MTL quality scores between correct and incorrect cases, p=0.768), and constitutes a cross-check ✓.

---

## 4. The cleanest design: using misclassified samples only

There are n=19 misclassified samples, composed as follows:

| Cell | Count |
|---|---|
| true MCI → pred AD | 4 |
| true MCI → pred CN | 5 |
| true CN → pred MCI | 5 |
| true AD → pred MCI | 3 |
| true CN → pred AD | 2 |

### 4.1 Within the misclassified samples, neither grouping variable is significant

| Grouping | H | p |
|---|---|---|
| By true label | 1.653 | 0.438 |
| By predicted label | 2.067 | 0.356 |

**Neither is significant** → this indicates that the between-group effect in T4 is **not driven by the misclassified samples**, but instead arises from the overall distributional difference among the 125 correctly classified samples. This once again confirms that the entanglement of the two factors occurs in the correctly classified region.

### 4.2 Directional test: which side does the MTL of misclassified samples follow?

| Cell | n | MTL | Distance to true side | Distance to predicted side | Closer to |
|---|---|---|---|---|---|
| true CN → pred MCI | 5 | 2.4686% | 0.2301 | 0.0622 | **predicted side** |
| true MCI → pred CN | 5 | 2.2994% | 0.2037 | 0.0566 | **predicted side** |
| true AD → pred MCI | 3 | 2.5513% | 0.1768 | 0.0205 | **predicted side** |
| true MCI → pred AD | 4 | 2.4251% | 0.0780 | 0.0827 | true side (weakly) |
| true CN → pred AD | 2 | 1.7179% | 0.5206 | 0.6245 | true side (weakly) |

**3 of the 5 cells are "closer to the predicted side", and the closeness of these 3 is large in each case (|Δ_pred| only 0.02–0.06);**
**the 2 cells that are "closer to the true side" are both small in magnitude (differences of only 0.0047 and 0.104), and have n of only 2–4.**

⚠️ This **tends to support the view that "MTL attention is dominated by the decision side"**, but **the sample size is too small (n=2–5 per cell) for this to be a statistical conclusion**; it can only serve as a directional indication.

### 4.3 A continuous-variable perspective

| Variable | Spearman ρ with MTL | p |
|---|---|---|
| True label | +0.1575 | 0.0573 |
| Predicted label | +0.1107 | 0.1843 |
| **Prediction confidence** | **+0.4073** | **1.1×10⁻⁷** |

**Note**: from the continuous perspective, the true label (marginally significant) is in fact stronger than the predicted label. This **appears to contradict** the KW result, for the following reasons:
- The KW test compares **differences in group means** (a categorical structure), whereas the rank correlation looks at a **monotonic trend** (an ordinal relationship).
- The labels are nominal variables (CN/MCI/AD have no natural order), so the rank correlation is itself only a crude reference and no conclusion should be drawn from it.

**The only robust continuous signal is prediction confidence, ρ=+0.407** — consistent in direction with the finding in T2 (confidence–CAM strength coupling ρ=0.800).

---

## 5. Interpretation and writing recommendations

### 5.1 Answer to the original concern

The external comment raised the objection: "the between-group difference in T4 = true-label effect + predicted-class effect + sampling fluctuation, and the three are so entangled that they cannot be separated."

**The present analysis confirms the "entanglement" judgement (V=0.81), but adds two facts that the original objection did not anticipate:**

1. **The entanglement is not a case of "cannot be analysed" but of "the two variables carry almost identical information"** — because the model agrees on the two for 86.81% of samples. This means that **whichever variable is used for grouping, the conclusions are much the same** (in both cases only CN vs MCI is significant). This in fact **reduces the severity of the T4 problem** rather than aggravating it.
2. **The signal on the predicted side is slightly stronger** (H 13.55 vs 11.14), with the post-hoc p value shrinking by roughly a factor of 3 (0.00079 vs 0.0026). That is, the original objection's conjecture that "the predicted side is the true signal" **holds in direction**, but the **magnitude is limited** and it does not constitute decisive evidence.

### 5.2 Does the manuscript need to be changed? — **No**

| Location | Current state | Assessment |
|---|---|---|
| §3.3 [064] T4 report | Grouped by true label, H=11.14, p=0.0038, only MCI>CN significant, mechanism unknown and not inferred | ✅ Robust and faithful |
| §3.3 T5 null result | No difference in MTL quality scores between correct/incorrect (p=0.768) | ✅ Cross-validated here (correctness explains only 0.06% of the variance) |
| §4.2 qualifiers | "attentional distribution and decision dependence are not equivalent" | ✅ This is precisely the defence for this issue, and it remains valid |

**Conclusion: even when re-run using predicted labels, the pattern of conclusions is exactly the same, and the evidential status of T4 is unchanged. The qualifiers already present in the manuscript are sufficient.**

### 5.3 Suggested wording if this is to be written into S6

> Given that the predicted labels in the test set are highly consistent with the true labels (accuracy 86.81%, Cramér's V=0.81), this study additionally tested the between-group difference when grouping by predicted label: MTL quality scores also differed significantly across the three groups (predicted CN n=46 / MCI n=47 / AD n=51) (Kruskal-Wallis H=13.55, p=0.0011, ε²=0.095), and the post-hoc Dunn test (Bonferroni correction) likewise left only the difference between predicted CN and predicted MCI significant (corrected p=0.00079). The two grouping schemes yield the same pattern of conclusions, suggesting that this between-group pattern is insensitive to the choice of grouping variable. It should be noted that, because the two factors are highly collinear (Cramér's V=0.81), their independent contributions cannot be separated additively; within the misclassified samples (n=19), the between-group difference was non-significant in both directions (by true label H=1.65, p=0.438; by predicted label H=2.07, p=0.356), suggesting that the between-group effect is carried mainly by the overall distributional difference among the correctly classified samples.

**Advantages of this wording**: ① it proactively reports the collinearity (defending against "why did you not separate them?"); ② it reports the **consistency** of the two KW results (converting the entanglement into "the conclusion is robust"); ③ it states that the difference is non-significant within the misclassified samples (defending against "is the effect driven by misclassification?").

---

## 6. Remaining issues and boundaries

- **It must not be claimed that "the predicted side is the true signal"** — the directional indication holds, but the sample size is insufficient (misclassified n=19, cell n=2–5), and no formal comparison test was performed on the KW difference (13.55 vs 11.14). If such a comparison is genuinely required, a **paired/resampling test of the difference in H** should be performed (not done at present; could be added).
- **ε² vs η²**: for the present KW, H/(n−1)=0.0779 (true side) and 0.0947 (predicted side), and should strictly be termed **ε²**; the true η² values are 0.0648 and 0.0819 respectively. This should be handled together with the terminology issue of the existing η²=0.078 in §3.3 (see the authors' internal verification note).
- All p values were computed with the **exact t-distribution CDF / upper incomplete gamma Q(a,x)** (hand-implemented in the absence of a scipy environment), and H includes a tie correction (there are no ties in these data, so the values are identical before and after correction).
- Before formally writing this up, it is recommended to cross-confirm once with `scipy.stats.kruskal` + `scikit-posthocs.posthoc_dunn`.
