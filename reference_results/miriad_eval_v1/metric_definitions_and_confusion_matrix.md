# MIRIAD result files — definition of the metrics and the confusion matrix (v45 addendum)

> Purpose: the repository contains both `0.7971` and the `78.3%` quoted in the paper text.
> These are **two different definitions, and both are correct**. The definitions are set out
> here so that neither a reproducer nor a reviewer mistakes them for an inconsistency.

**The paper protocol is the v2 evaluation** (`reference_results/miriad_eval_v2/`); the numbers
cited in the paper (AUC 0.8790, sensitivity 76.1%, specificity 87.0%, exact-match accuracy
78.3%) come from v2. The v1 files in this directory are archived alongside it and use the same
definitions.

## 1. The two accuracy definitions

| File / location | Value | Definition |
|---|---|---|
| `Accuracy` in `miriad_ensemble_metrics.csv` / `miriad_binary_metrics.csv` | **0.7971** (55/69) | **Binary definition**: `y_pred_binary = (argmax ≠ 0)`, i.e. a prediction of CN counts as 0 (normal) while a prediction of **either MCI or AD counts as 1 (abnormal/AD)**. Correct = HC→CN 20 + AD→AD 34 + AD→MCI 1 = 55. |
| Paper §3.4 / supplementary S7 | **78.3%** (54/69) | **Three-class exact-match (argmax) definition**: the predicted class must equal the true class exactly. Correct = HC→CN 20 + AD→AD 34 = 54 (the one AD→MCI case counts as an error). |

The two must not be mixed; the value quoted in the paper text is the three-class exact-match
definition.

## 2. Other metrics consistent with the above

- `Sensitivity = 0.7609 = 35/46`: within the AD group, predictions of AD (34) or MCI (1) both
  count as positive.
- `Specificity = 0.8696 = 20/23`: within the HC group, predictions of CN.
- `AUC (AD vs HC) = 0.8790`: computed over all 69 HC/AD subjects using `prob_AD` as the
  decision score (`roc_auc_score`).
- `F1 Macro = 0.7870`: macro-averaged F1 under the binary definition.

## 3. Three-class confusion matrix (rows = true, columns = predicted)

|  | Pred CN | Pred MCI | Pred AD |
|---|---|---|---|
| **True HC (n=23)** | 20 | 0 | 3 |
| **True AD (n=46)** | 11 | 1 | 34 |

- AD→HC missed diagnoses 11/46 = **23.9%**
- AD→MCI 1/46 = **2.2%**
- HC→AD false positives 3/23 = **13.0%**
- HC→MCI 0

## 4. Ensemble versus single folds (`miriad_single_fold_auc.csv`)

| Fold | AD vs HC AUC |
|---|---|
| 1 | 0.8459 |
| 2 | 0.8790 |
| 3 | 0.6928 |
| 4 | 0.8658 |
| 5 | 0.8554 |
| **Mean** | **0.8278** |
| **Ensemble** | **0.8790** |

> The ensemble AUC coinciding numerically with the fold-2 AUC is a coincidence (the ensemble is
> the output after averaging the logits of the five models and is not the same evaluation object
> as any single-fold model). The paper reports them **side by side neutrally** and performs no
> significance test on the difference, so it does not claim an ensembling gain.

## 5. Reproduction

The evaluation script ships with the package as `evaluate/miriad_eval.py`:

```bash
python evaluate/miriad_eval.py
```

The corresponding archived original (authors' workspace, not distributed) was
`run_miriad_ensemble_evaluation_v2.py`.

See `reference_results/miriad_eval_v2/` for the authoritative per-sample predictions and
confusion matrix.
