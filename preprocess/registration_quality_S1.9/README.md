# S1.9 Registration-quality artifacts

Raw artifacts behind the registration-QC numbers reported in paper section 2.2 / supplementary S1.9.
All 954 subjects (318 CN/MCI/AD) were registered with ANTs SyN-fast and quantified with three
metrics: brain-mask Dice, global Pearson correlation (r), and mean absolute error (MAE).

## 1. Authoritative file

| File | Description |
|---|---|
| **`registration_quality_summary.csv`** | **★ Authoritative source.** Per-group N / Dice / r / MAE mean±SD and threshold shares. `step5b_registration_anova.py` takes this file as input. |
| `summary_stats_3groups.txt` | Narrative summary generated from the CSV above |
| `between_group_effect_sizes.txt` | Three-group ANOVA + effect size η² |

## 2. ⚠️ Definition warning: two files disagree on the per-group values

The per-group r / MAE values in `registration_quality_report.txt` are **inconsistent with the
authoritative CSV**:

| Group | Authoritative CSV (paper protocol) | `registration_quality_report.txt` |
|---|---|---|
| CN | r 0.7841 / MAE 621.0 | r 0.7840 / MAE 618.5 |
| MCI | r **0.7741** / MAE **644.9** | r **0.7769** / MAE **651.1** |
| AD | r 0.7825 / MAE 640.1 | r 0.7796 / MAE 636.3 |

The report is an early intermediate document. **Paper section S1.9 uses the CSV values**
(the ANOVA script in this directory also reads the CSV). Use only
`registration_quality_summary.csv` when checking paper numbers.

> **Label convention.** The archived tables and reports in this directory label this metric
> `CC` (`CC_mean`, `CC_std`, `CC_ge_0.85`, …), whereas the manuscript and this README call it
> `r`; the two labels denote the **same** quantity (the global Pearson correlation used as the
> registration similarity metric). For example, the `CC_ge_0.75` column (874/954 = 91.6%) is
> the manuscript's "91.6% of subjects with r ≥ 0.75".

## 3. ⚠️ Data-compliance note (resolved in this release)

The tail of `registration_quality_report.txt` previously listed **subject-level ADNI imaging
identifiers**. In this release those identifiers have been **replaced with deterministic
pseudonyms** (e.g. `ADNI-BB2F2B_AD_M_Sag_IR-SPGR`); the mapping table is retained offline by
the authors and is not distributed. The authoritative CSV contains only group-level summary
statistics and was never affected.

## 4. Reproduction

```bash
python preprocess/step5b_registration_anova.py
```

Prints the ANOVA F / p / η² of the three metrics and compares them item-by-item with the
values reported in paper S1.9 (exit code 0 = all consistent).
