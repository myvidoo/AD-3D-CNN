# 03_training/legacy — archive of the original training scripts

`cnn_model_v3_7_d169_group_comparison.py` is the **original training script** actually used in
this study (v3.7 optimized edition), archived for code provenance.

> **Note on this release.** The script's code, constants, identifiers, file paths and training
> semantics are exactly those of the authors' original. Only its comments and docstrings have
> been translated to English (and, in the translated edition, its diagnostic `print()` messages).
> Nothing that affects behaviour was changed, so the provenance argument below still holds: a
> reviewer diffing this file against the refactored `train/train_grid_search.py` will see
> comment/text differences only, never a structural or numerical one. The one exception is a
> path literal that intentionally preserves the authors' original workspace layout.

## Why this file is in the package

1. **Single authoritative source of the model architecture.**
   `evaluate/run_inference_benchmark.py`, `stats_rechecks/inference_pipeline_check.py`, and
   `stats_rechecks/miriad_single_fold_recheck.py` rebuild the network via
   `from cnn_model_v3_7_d169_group_comparison import ...` and load the checkpoints in
   `weights/`, guaranteeing that the model definition matches the paper's weights verbatim.
2. **Diff-based provenance.**
   Reviewers can diff this file against the refactored `train/train_grid_search.py` to verify
   that the parameterization refactor (paths injected from config.yaml, interactive modes
   turned into CLI arguments) changed neither the model structure nor the training semantics.

## RUN_MODEs and their relation to the paper

The file switches between six working modes via a `RUN_MODE` constant — all real features from
the research iteration:

| RUN_MODE | Purpose | Relation to the paper |
|---|---|---|
| `train` (SEARCH_METHOD=grid) | 256-run grid-search training | **Paper main path** (2.3–3.1; source of Fig. S1–S3 data) |
| `choose` | Grid best-run selection (validation AUC/F1/ACC stability) | Selection process of Run128 (2.3/3.1); not actually used |
| `test` | Unified test-set evaluation of all finished runs | Development-stage predecessor of 3.2 (official version in `evaluate/`) |
| `inference` / `batch_inference` | Single-sample Grad-CAM inference and batch inference | Development-stage predecessor of 2.5–2.7 (official version in `evaluate/`) |
| `group_comparison` | Grad-CAM group comparison | Development-stage predecessor of 3.3/4.2 |
| `train` (SEARCH_METHOD=bayesian) | Bayesian-optimization branch | Explored but abandoned; the paper uses grid search |

## Usage notes

- **Do not run this file directly**: it contains the author's local hardcoded paths, and the
  interactive configuration has been superseded by CLI arguments in the refactored scripts.
- **If you do run it anyway**, be aware that its split handling differs from the refactored script.
  Inside `run_single_training()` — reached both from `run_grid_search()` and from the Bayesian
  `objective()` — a bundled split that does not match the local dataset is regenerated and written
  to `03_training/legacy/fixed_data_split.json` (the module constant `FIXED_DATA_SPLIT_JSON`,
  resolved next to this script). `train/train_grid_search.py` performs the same regeneration but
  writes the result to `output_dir/` instead, so that the de-identified
  `data/fixed_data_split.json` is never overwritten. This file is preserved verbatim for
  provenance and is not the reproduction path — see the note under §4.3 of the main README.
- The **official reproduction path** for all paper results is:
  `train/train_grid_search.py` (grid search) → `train/train_5fold.py` (5-fold CV)
  → `evaluate/` (evaluation and inference benchmark).
- The remaining RUN_MODEs (test / choose / group_comparison / bayesian) are development-stage
  iterations; **the paper and the refactored scripts are the authoritative protocol**.
