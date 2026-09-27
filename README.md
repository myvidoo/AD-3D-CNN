# AD_CNN_code — Reproduction Code Package

Corresponding paper: **"CN/MCI/AD Classification of Alzheimer's Disease from Single-Modality Structural MRI Alone: 3D Convolutional Neural Network, Controlled Baselines, and the Cross-Cohort Generalization Boundary"**.

This package is organised to SCI reproducibility standards and covers **all training, preprocessing, inference, statistics, and plotting** stages:

- All values in Tables 1–4 and S1–S6
- All figures 2–5 and S1–S4 (Figure 1 is the flowchart; source in `assets/`)
- 5-fold best model weights (`weights/`) and 144-subject per-sample probabilities (`data/ensemble_test_predictions.csv`)

> **De-identification notice.** Every ADNI subject-level identifier has been replaced with a deterministic pseudonym: image IDs (`Ixxxxxxx`), subject IDs (`xxx_S_xxxx`) and their combined forms were removed, and each subject is now referred to by a sequential pseudonym `ADNI-0001`…`ADNI-2101`. Where a filename previously carried the numeric subject component of a real ADNI subject ID, that component has been replaced by a `P`-prefixed six-character hash (e.g. `ADNI-0029_S_P7C3A9`), so no original subject number survives. The mapping tables are retained locally by the authors and are **not** distributed. Local file-system paths are replaced by placeholders (`<WORKSPACE>`, `<AUTHOR_DATA_ROOT>`, `<LOCAL_HOME>`, `<TOOL_DIR>`, `<PY_ENV>`, `<SCRATCH_DIR>`). Code logic and every reported numeric value are unchanged. MIRIAD and OASIS-2 are public datasets and are not subject to the same restriction.

> **Note on this English edition.** The package was prepared in Chinese and has been fully translated. What was translated: all documentation, all code comments and docstrings, all console output, and the explanatory notes/field labels embedded in the reference artifacts. What was **not** changed: any executable statement, any identifier, any hyperparameter, any file path used by the code, and **any number** — every statistic in this package is value-identical to the authors' original. Two deliberate exceptions are documented where they occur: (i) the archived `occlusion/reference_results/*.log` and `reference_results/s9_baselines/_*.log` files are transcriptions of the original runs' stdout — every value, table and layout is verbatim, but the authors' internal Chinese workspace folder names are rendered in English (e.g. the occlusion experiment workspace becomes `<WORKSPACE>/27_occlusion_experiment/results/…`), so the logs are transcriptions rather than byte-identical raw stdout; (ii) some spreadsheets still contain Chinese **format boilerplate** — Office theme and built-in style names embedded in `xl/theme/theme1.xml`, `xl/styles.xml` and `docProps/app.xml`. These are part of the xlsx template that any Simplified-Chinese-locale Excel writes, not content: no cell value, column header or sheet name in any shipped spreadsheet is Chinese. Aside from the xlsx templates, every text file in this package is Chinese-free.

---

## 1. Directory layout → paper mapping

```
AD_CNN_code/
├── config.yaml                    Single path configuration
├── config.py                      Configuration loader
├── requirements.txt               Dependencies
├── preprocess/                    Paper 2.1–2.2 / S1 (7-step pipeline) — see preprocess/README.md
├── model/densenet169_attention.py Model (DenseNet-169 + SE/CBAM/ECA/axial gating, 2.3/S8)
├── common/utils.py                Data pipeline / training loop / checkpoint
├── train/                         Paper 2.3–2.4 / S2–S3 (grid search + 5-fold CV)
├── 03_training/legacy/            Original training scripts archive (see its README.md)
├── inference/predict_ensemble.py  5-fold ensemble inference
├── evaluate/                      Paper 2.5–2.7 / 3.2–3.4 / 4.2–4.3 / S5–S8
├── baselines/                     Paper 2.7 / 3.5 / S9 — see baselines/README.md
├── occlusion/                     Paper 2.5 / 3.3 / 4.3 / S6 — see occlusion/README.md
├── figures/                       Figure 4 redraw (redraw_figures_ensemble.py)
├── assets/                        Figure 1 source (HTML); delivered as PDF (vector) + PNG 4134×4572
├── stats_rechecks/                Independent statistical recheck scripts (§5)
├── data/                          Frozen data (splits / probabilities / lists)
├── reference_results/             Authoritative artifacts per stage (comparison anchors)
├── weights/                       fold_1..5_best_geo.pth (Run128 checkpoints)
├── results/                       Run output directory (author's copy included)
└── smoke_tests/                   ★ Smoke tests (run this first)
```

---

## 2. Environment

### 2.1 Main environment (training / inference / evaluation / plotting)

```bash
conda create -n ad3dcnn python=3.10 && conda activate ad3dcnn
pip install -r requirements.txt
```

### 2.2 Preprocessing and baselines (ants_env)

Structural-panel extraction (S9) and SyN registration need ANTsPy:

```bash
conda create -n ants_env python=3.10
conda activate ants_env
pip install ants nibabel SimpleITK scikit-learn pandas openpyxl scipy pyyaml
```

### 2.3 External command-line tools

| Tool | Purpose | Used in |
|---|---|---|
| [dcm2niix](https://github.com/rordenlab/dcm2niix) | DICOM → NIfTI | preprocess/step1 |
| [ANTs](http://stnava.github.io/ANTs/) (antspy) | N4 correction, SyN registration, Jacobian | preprocess/step3, baselines |
| [FSL](https://fsl.fmrib.ox.ac.uk/) | Harvard-Oxford atlas | gradcam_roi, structural panel |
| MONAI Label + 3D Slicer | Interactive skull stripping | preprocess/step2 (server config in `monailabel_configs/`, operating notes shipped) |

---

## 3. Quick reproduction (10 min, offline, no imaging/GPU)

Every statistic in Table 1, Tables 2 and S5, Figures 2–4 and S4, and the whole occlusion analysis
is reproducible from the bundled data alone:

```bash
# (1) Smoke test: recompute all core Table 2 / Table S5 metrics and assert them against the
#     paper's numeric anchors (~1 s, standard library only)
python smoke_tests/smoke_offline_recheck.py

# (2) Table 2 full metrics + confusion matrix + Brier/ECE + Youden + confidence (writes results/)
python evaluate/report_metrics.py

# (3) Bootstrap 95% CIs for Table 2
python evaluate/bootstrap_ci.py --n_bootstrap 2000

# (4) Figure 2 calibration / Figure 3 confusion matrices / Figure S4 confidence
python evaluate/plot_figures.py --all

# (4b) Figure 4 ROC + PR curves — produced by a **separate** script; plot_figures.py does not write Figure 4
python figures/redraw_figures_ensemble.py

# (5) Figure S3 (RF importance) / Figure S2 (learning-rate violin), from the 256-run grid search
python evaluate/generate_grid_figures.py

# (6) Recompute every occlusion / tissue-transplant statistic (63 numeric anchors + 87 traceability assertions)
python occlusion/verify_occlusion.py

# (7) S9 controlled-baseline main definition (ants_env; full 2000-resample bootstrap ≈5–8 min)
python baselines/run_baselines_tuned.py              # paper definition
python baselines/run_baselines_tuned.py --n-boot 200 # quick

# (8) Table 1 demographics — offline: reads only the bundled subject summary, ≈2 s
#     (needs pdf2image + system poppler for the PNG preview only; PDF and XLSX are written either way)
python preprocess/step6b_generate_table1.py
```

**Expected results** (matching the manuscript): accuracy 0.8681 (125/144), macro-F1 0.8675,
macro-AUC 0.9630, confusion matrix CN `[41,5,2]` / MCI `[5,39,4]` / AD `[0,3,45]`,
macro Brier 0.0748, macro ECE 0.0774, argmax ECE 0.1163;
B1 0.5284 / B2 0.6941 / B3 0.6492 / B4 0.7079 / B5 0.7075 (B4 and B5 are essentially tied
after tuning, differing by 0.0004); MIRIAD AUC 0.8790, OASIS-2 CN-vs-AD AUC 0.7567.
Table 1: 318 subjects per group, 166 M / 152 F (52.2% / 47.8%), age 73.11±8.40 (CN) /
73.66±8.24 (MCI) / 74.55±8.30 (AD).

> The assertion anchors of step (1) are exactly the values above; exit code 0 = all passed.
>
> **Note on step (6)**: `verify_occlusion.py` **passes in full, exit code 0** — 26 manifest items
> byte-identical, **63/63 numeric anchors**, **8/8 recomputed JSON artifacts field-by-field
> identical (1631 fields)**, **87/87 traceability rows** and **14/14 cross-artifact invariants**.
> Two conventions in that check are worth knowing, and both are documented in
> `occlusion/README.md` §10:
>
> 1. Figure comparison uses a **robust criterion** rather than a strict max-pixel rule, because two
>    renderings of byte-identical data can differ by a few 1/255 steps where a dashed line crosses a
>    marker cluster. Every figure's `max|Δpixel|` and its fraction of pixels differing by more than
>    8/255 are printed, and the check fails only if that fraction exceeds 0.1% (or a channel exceeds
>    32/255) — a genuine change alters far more than that. Five of the six occlusion figures are
>    pixel-identical to the frozen copies; `fig_occlusion_paired.png` has 0.0000% of pixels above the
>    threshold (`max|Δpixel| = 0.0157`, i.e. 4/255). The frozen PNGs were not modified.
> 2. `patch_stats.json` → `P4_prob_destination.AD_to_CN_flip_ids` was **re-ordered** to the
>    deterministic order the script produces: the frozen file came from an earlier script revision
>    whose ordering followed the original, pre-de-identification subject numbers, which
>    desensitisation necessarily removed. The same three samples are listed either way, no other field
>    changed, and `MANIFEST.md5` was rebuilt. The order of that three-element list carries no meaning.

---

## 4. Full reproduction (data acquisition → preprocessing → training → evaluation)

### 4.1 Data acquisition (governed by data-use agreements; imaging is not distributed)

1. **ADNI**: apply at https://adni.loni.usc.edu/ and download the 954 3T T1 scans (the subject
   Image Data ID list ships as
   `data/data_lists/ADNI_subject_search_list_selected_AD_MCI_CN_MR_3T_T1_ALL_2025-09-30.xlsx`;
   identifiers in it are pseudonyms).
2. **MIRIAD**: apply at https://miriad.drc.ion.ucl.ac.uk/, 69 subjects (HC 23 + AD 46);
   `preprocess/step0_download_miriad.py` provides a resumable bulk download.
3. **OASIS-2**: apply at https://www.oasis-brains.org/, 147 single-session scans;
   the de-identified list ships as `data/oasis2_basic_info.csv`.

### 4.2 Preprocessing (details in `preprocess/README.md`)

```
step1   DICOM retrieval / series selection + dcm2niix → NIfTI (S1.1–S1.2)
step1b  sex hard-constraint + age soft-optimisation matching → 318 per group (2.1/S1.4)
step2   MONAI Label + 3D Slicer interactive skull stripping (S1.5, Dice 0.96–0.98;
        server config monailabel_configs/deepedit_20260521.py ships with the package,
        deployment command in preprocess/README.md §3.1)
step3   batch ANTs SyN-fast registration to MNI152 (S1.6) ★ resumable; outputs registered
        images + deformation fields
step4   WhiteStripe intensity normalisation (S1.7) → final training data (directory suffix *_fast_ws)
step5   registration QC (Dice/CC/MAE summary + three-group ANOVA, S1.9)
step6   Table 1 demographics statistics (2.1)
```

Fill in the per-step paths in the `preprocess:` section of `config.yaml` (or override them with
the step3/step4 command-line arguments).

### 4.3 Training

```bash
# (1) 256-run grid search (methodological reference; the full 256 runs take weeks of GPU time)
#     Smoke test (verifies the pipeline runs, ~10 min): --max_runs 1 --epochs 1
python train/train_grid_search.py --max_runs 256
python train/train_grid_search.py --max_runs 1 --epochs 1    # smoke

# (2) 5-fold cross-validation + ensemble using the Run128 configuration (run this step directly
#     to reproduce the final model, together with the bundled weights/; if re-running the grid
#     search selects a different best run, update best_run_id in config.yaml)
python train/train_5fold.py

# (3) Generate the 144 per-sample prediction probabilities
python inference/predict_ensemble.py
```

> **About Run 128 (important)**: Run 128 (axial spatial gating + lr=1e-4, validation AUC
> 0.9665) is the best combination observed in this run, and sits in the same performance tier as
> ranks 2–8 (0.963–0.966). GPU floating-point non-determinism means a re-run of the grid search
> may place any of the top few first (three independent retrainings of the same configuration
> gave AUC 0.9419/0.9478/0.9512, see S6). This changes none of the paper's conclusions.
> To reproduce the paper's final model, use the Run128 configuration and the bundled weights.

> **About the bundled test split (`data/fixed_data_split.json`) — read before running training.**
> The shipped split is the **de-identified** edition: its entries are pseudonymised relative paths
> (e.g. `1/ADNI-2025_S_P83F6BB_...`). The images you obtain through the DUA carry their original
> ADNI filenames, and the pseudonym mapping table is not distributed — so the two sets cannot be
> joined, and the intersection is empty by construction rather than through a packaging error.
>
> `train/train_grid_search.py` detects this, prints a `[WARN]`, and **regenerates** a split with
> `random_state=42`, writing it to `output_dir/fixed_data_split_regenerated.json`; it never
> overwrites the bundled file. `train/train_5fold.py` only reads the split and does **not**
> regenerate it — so if the bundled split matches none of your files, all five folds train to
> completion and the final ensemble step then fails while computing the AUC
> (`ValueError: Found array with 0 sample(s)`), i.e. hours of GPU time are wasted before the
> failure surfaces. Check the `[INFO] total data: ..., test set: N, ...` line printed at startup:
> if `N` is 0, stop and supply a split that matches your data first.
>
> Two consequences for reproduction:
> 1. The regenerated split reproduces the authors' exact 144 test subjects **only if your file list
>    is in the same order** as the authors' list Excel — the split is drawn from that row order
>    under a fixed seed. Otherwise you obtain a different but equally valid stratified split.
> 2. The paper's reported numbers do not depend on re-deriving the split: they come from the frozen
>    `data/ensemble_test_predictions.csv` and `reference_results/`, which §3 reproduces offline.
>    Use those for any comparison against the manuscript.

### 4.4 Evaluation, interpretability and external validation

```bash
# Offline evaluation (same as steps (2)–(5) of §3)
# External validation (needs preprocessed MIRIAD / OASIS-2 images + weights; zero fine-tuning)
python evaluate/miriad_eval.py
python evaluate/oasis2_eval.py

# Grad-CAM population statistics + ROI overlap (needs the 144 ADNI images + weights + Harvard-Oxford atlas)
python evaluate/gradcam_roi.py
python evaluate/gradcam_quadgrid.py --recompute   # Figure 5 single-sample 3x3 grid (paper rule)
python evaluate/gradcam_quadgrid.py --recompute --match-published   # ... pinned to the published rendering (see 7.10)
python evaluate/random_init_control.py            # random-initialisation control (S6)

# Inference latency benchmark (4.3: GPU 113±2 ms / CPU ≈3.4 s)
python evaluate/run_inference_benchmark.py
```

> **On the latency numbers.** `run_inference_benchmark.py` measures wall-clock time, which depends
> on the GPU/CPU model, driver, thermal state and background load. The paper's §4.3 figure
> (GPU 113±2 ms, CPU ≈3.4 s) **is** the measurement archived in
> `reference_results/inference_benchmark/` — `112.9 ± 1.8 ms` / `3.419 s`, taken on an RTX 5090 D
> under Python 3.10.20 + torch 2.13.0.dev. Two further measurements of the same model exist
> (109.8 ± 6.2 ms under Python 3.12.7 + torch 2.7.0; and 93.2 ± 1.4 ms in a later session on the
> same stack as the first), while the **CPU** figure stayed within 7 % of 3.4 s across all three
> (3.419 / 3.427 / 3.199 s). A re-measurement will therefore land in the same range but not match
> digit for digit; what is claimed is the order of magnitude (ensemble ≈5× a single fold, GPU
> ≈30× faster than CPU). Because the number is a hardware snapshot rather than a reproducible
> quantity, the `results/` copy is **not shipped** — the authoritative value is the one under
> `reference_results/inference_benchmark/`.
>
> **On the field names in `benchmark_results.json`.** The `model_size` block labels its values
> `single_model_fp32_mb` and `ensemble_5fold_total_mb`, but the values are **MiB**, not MB:
> `72.7` is (18,898,926 parameters + 158,674 buffers) × 4 B = 76,230,400 B = 72.70 MiB
> (76.23 MB decimal), and `363.49` is five times that. The manuscript's §4.3 text states the unit
> as MiB for this reason. The field names are **left unchanged on purpose** — renaming them would
> break the byte-level reproducibility of the archived JSON against `MANIFEST`-style comparisons —
> so read `*_mb` as MiB here; the sibling key `single_checkpoint_file_mb_mean: 73.28` is likewise
> MiB (the actual `.pth` file size, which is larger than the parameter payload because the
> checkpoint stores more than the FP32 weights).

### 4.5 Controlled baselines (ants_env)

```bash
# (1) Structural measurement extraction (6 features, 1170 subjects, ≈13 min; needs the atlas entries in config.yaml)
python baselines/extract_structural_panel_s9.py

# (2) Paper main definition (tuned) and the untuned control
python baselines/run_baselines_tuned.py
python baselines/verify_tuned_baselines.py

# (2b) 5-fold reproduction with hyperparameters fixed (paper S9 parenthetical values 0.5044–0.7110; offline, ≈1 min)
python baselines/run_baselines_5fold_fixedhp.py

# (3) Measurement correctness verification (atlas blind test + threshold calibration + directionality + threshold sensitivity)
python baselines/verify_atlas_identity.py
python baselines/verify_structural_panel.py
```

> To reproduce the S9 numbers without re-extracting the measurements, just run (2), (2b) and (3):
> the `structural_panel.csv` they need (1170 subjects, 671225 B) already ships in place, in three
> bit-identical copies — `data/` is read by (2) and (2b), `results/` by (3) (and is also where
> (1) writes), and `reference_results/s9_baselines/` is the comparison archive. Nothing needs to
> be placed or edited by hand.
>
> **Note on (1).** `extract_structural_panel_s9.py` needs the registered images *and* their
> deformation fields (`<data_root>/<label>/warpfields/<subject>_warp.nii.gz`), both produced by
> `preprocess/step3`. If those are absent it now stops with a non-zero exit and a message naming
> the paths it looked for, instead of writing an all-but-empty panel; and it refuses to overwrite
> an existing non-empty `results/structural_panel.csv` unless `--force` is given. `--min-success-frac`
> (default 0.5) sets how large a fraction of the 1170 subjects must extract successfully. The
> failure list is always written to `results/structural_panel_failures.csv` for diagnosis.

### 4.6 Occlusion and tissue-transplant perturbation analysis (S6)

```bash
# Recheck all statistics only (no imaging/GPU/weights needed, ≈2 min)
python occlusion/verify_occlusion.py

# Regenerate from imaging (needs the 144 ADNI images + weights + atlas; ≈6 min GPU)
python occlusion/step0_baseline_p0.py       # baseline p0
python occlusion/step1_build_masks.py       # masks
python occlusion/step2_occlusion_sweep.py   # 10-condition sweep
# then re-run the CPU statistics chain as described in occlusion/README.md §7.1
```

---

## 5. Independent statistical recheck scripts (stats_rechecks/)

Independent recomputation of the statistics reported in each paper section (all inputs already
point at this package's `data/` and `reference_results/`):

| Script | Paper location | Input |
|---|---|---|
| `grid_stats_recheck.py` | 3.1/S2/S4 grid statistics, KW, permutation importance | data/all_training_results.csv |
| `testset_perf_recheck.py` | 3.2 test-set metric recheck | data/ensemble_test_predictions.csv |
| `gradcam_kw_eta2_recheck.py` | S6 KW η² (baseline = archived JSON; this group of numbers is not reported in the manuscript) | reference_results/gradcam_v3 |
| `gradcam_dunn_posthoc_recheck.py` | S6 Dunn pairwise post-hoc tests | reference_results/gradcam_v3 |
| `t2_sensitivity/` (5 scripts) | T2 sensitivity (robustness of the confidence–CAM correlation) | reference_results/gradcam_v3 |
| `kw_pred_label/` (3 scripts) | S6 Kruskal-Wallis re-run by predicted label / collinearity / two-way decomposition | reference_results/gradcam_v3 |
| `miriad_single_fold_recheck.py` | S7 MIRIAD single-fold AUC recheck | list file + weights/ (needs a GPU environment) |
| `inference_pipeline_check.py` | S8 inference pipeline correctness (144-case recheck) | data + weights/ (needs a GPU environment) |

> **Note on `grid_stats_recheck.py`.** This script is a *secondary* check; the manuscript's
> grid-search figures are produced by `evaluate/generate_grid_figures.py`. Two points are worth
> knowing before reading its output.
>
> 1. **Dispersion convention (why `ddof=1` is written out explicitly).** A NumPy array's bare
>    `.std()` is the *population* SD (ddof=0), whereas the manuscript reports the **sample** SD for
>    every AUC dispersion it quotes: `0.9471±0.0094` over all 256 runs, `0.9472±0.0080` (CBAM) /
>    `0.9466±0.0097` (ECA) / `0.9478±0.0087` (axial) per re-scaling module, and `0.9470±0.0047` for
>    the three Run128 retrainings. The script therefore passes `ddof=1` explicitly wherever it
>    reports an AUC SD, so its self-comparison lines match the manuscript. The two conventions are
>    numerically close — `0.00935096` (ddof=1) versus `0.00933268` (ddof=0) for the 256-run set —
>    so the choice only matters at the 4th decimal.
>    The **one exception is the R² line**, which deliberately uses the population SD (`ddof=0`):
>    the manuscript's `R² = 0.2391±0.1300` comes from `evaluate/generate_grid_figures.py`, which
>    also computes that figure with `ddof=0` (`0.1300`; the sample SD would be `0.1453`). The
>    script annotates this inline.
> 2. **Random-forest R² and importance share.** The script re-fits the random forest under a
>    *different feature encoding*, so it prints `R² = 0.1164 ± 0.2018` and a learning-rate share in
>    the 48–58% range, whereas the manuscript reports `R² = 0.2391 ± 0.1300` and a share of
>    `48.3%`. The manuscript's values come from `evaluate/generate_grid_figures.py`, which
>    reproduces them exactly (the two scripts also differ in forest hyperparameters:
>    `min_samples_split=5, min_samples_leaf=2` there versus scikit-learn defaults here); the script
>    states this in its own output. It is a robustness control under a different encoding, not a
>    contradiction.

## 6. reference_results/ comparison anchors

`reference_results/` archives the authoritative artifact of each stage (cv_ensemble,
miriad_eval_v1/v2, oasis2_eval_v2, s9_baselines, inference_benchmark, t2_sensitivity,
kw_pred_label, gradcam_v3, table1). After reproducing, compare your own outputs against these
file by file; the assertion anchors used by `smoke_tests/smoke_offline_recheck.py` come from
the manuscript and from these artifacts.

---

## 7. FAQ and caveats

1. **Random seeds**: methodological constants such as `GLOBAL_SEED=42` are hard-coded in
   `common/utils.py` — do not modify them (the paper's numbers cannot be reproduced afterwards).
2. **AUC definition**: the paper's macro-AUC is **0.9630** (exact value 0.9629629629629629; see
   `reference_results/cv_ensemble/ensemble_test_result.csv` and `results/report_metrics.json`;
   any standard recomputation from `data/ensemble_test_predictions.csv` gives 0.9630). Note the
   distinction: 0.9629445 in the `Macro_AUC` column of `stratified_bootstrap_ci_results.csv` is a
   **bootstrap resampling mean**, not the point estimate, and must not be reported as the value.
   The 0.8278 in S7 is the MIRIAD single-fold mean and is unrelated to the internal test-set metrics.
3. **ECE definition**: macro-ECE 0.0774 is a one-vs-rest macro average (10 equal-width bins,
   weighted by sample count; implementation in `common/calibration_metrics.py`). The argmax ECE
   0.1163 is a different definition; the two must not be mixed.
4. **Saturation threshold**: the non-saturated subset in the occlusion analysis is defined as
   `top-1 > 0.999999` (64 saturated / 80 non-saturated).
5. **Registration strategy**: the paper and this code both use **syn_fast** (iterations 50/30/20,
   CC, sampling 4, flow_sigma 3, total_sigma 0). An earlier manuscript version stated (100,70,50)
   (the syn_standard parameters); the v61 proof corrected this to 50/30/20, matching the step3
   defaults here, so no discrepancy remains.
6. **Training logs**: the authors' original training logs are not shipped (size); they are
   archived in the authors' paper workspace (not distributed).
7. **Number alignment**: the baseline main-results table in `baselines/README.md` has been
   aligned with the manuscript's tuned definition (B4 0.7079 / B5 0.7075).
8. **Implementation detail: three "inactive/identity" configurations** (listed so that a
   reproducer comparing code against the paper need not worry — none materially affects the
   training dynamics, and §2.3 of the paper correspondingly makes no claim about them):
   ① `flip_lr_prob` (`AUGMENTATION_PARAMS` in `common/utils.py`) — not enabled; the transform
   pipeline contains no `RandFlipd`, so the effective augmentation is four types: rotation,
   scaling, Gaussian noise and contrast. ② `label_smoothing=0.05` (the default argument of
   `get_loss_function`) — inactive, because the training configuration sets `loss_name='ce'` and
   takes the branch without smoothing (the parameter is only used by the `ce_ls` branch).
   ③ Inverse-proportional class weights — approximately the identity: the data follow a balanced
   318×3 design, the 5-fold path has an exactly equal training subset per fold (216 per class) so
   the weights are exactly [1,1,1]; the grid-search path has 667/143, not divisible by 3, giving
   [223,222,222] and weights [0.997, 1.001, 1.001] — a deviation from equal weighting below 0.3%,
   with negligible practical effect.
9. **Atlas paths must be filled in**: the three `atlas:` entries in `config.yaml`
   (Harvard-Oxford subcortical/cortical and the MNI152 template) default to placeholders and must
   be pointed at your local FSL installation. Until then the following scripts raise
   `FileNotFoundError` for the Harvard-Oxford atlas, which is expected behaviour:
   `evaluate/gradcam_roi.py`, `evaluate/random_init_control.py`, `evaluate/gradcam_quadgrid.py`,
   `baselines/extract_structural_panel_s9.py`. These scripts require the FSL atlas and
   preprocessed images anyway; the offline reproduction chain (§3) does not depend on them.
   Related: when `data.miriad_root` / `data.oasis2_root` holds no `*.nii.gz` images,
   `evaluate/miriad_eval.py` and `evaluate/oasis2_eval.py` print an explicit
   "No … images found … exiting 0" message and stop cleanly, rather than failing. The two
   external-validation results of §3.4 can therefore be verified from
   `reference_results/miriad_eval_v2/` and `reference_results/oasis2_eval_v2/` alone, with no
   imaging and no GPU.
10. **Figure 5 reproduction boundary**: Figure 5 (Grad-CAM single-sample 3×3 grid) is the only
    main-text figure that cannot be reproduced offline — it needs imaging plus a GPU and the
    sequential execution of `evaluate/gradcam_roi.py` and `evaluate/gradcam_quadgrid.py`. The
    finished figure ships with the package
    (`results/gradcam_aggregate_v3/single_sample_grid/Figure_5_gradcam_quadgrid_rebuild.png`);
    the intermediate slices are not distributed.

    **Representative-sample rule (paper §2.5 / S6).** By default `gradcam_quadgrid.py` applies the
    paper's rule, which has no free parameter: within each (true, predicted) cell it takes the
    sample whose predicted-class confidence is closest to that cell's median confidence. Two
    caveats, both worth knowing before comparing a re-rendered figure against the shipped one:

    - **Exact ties.** In saturated cells many confidences are equal to within float64 resolution,
      so several samples can sit at the same minimal distance and the rule alone does not single one
      out. The script then takes the first such sample in predictions-CSV order and prints a
      `[NOTE]` naming the affected cells. With the bundled predictions this happens for `CN->CN`,
      `CN->AD`, `MCI->AD` and `AD->AD`.
    - **Recomputation.** Re-running the ensemble shifts confidences by ~1e-7 (measured drift between
      two runs of the same five checkpoints: 1.5e-7). That is enough to change which sample is
      closest to the median, so a re-rendered figure can show a different — equally representative —
      subject than the published one. This affects the illustrative single-sample panels only: the
      population-level Grad-CAM statistics of §3.3 are computed over all 144 test samples
      (`evaluate/gradcam_roi.py`) and are unaffected.

      Concretely, with the bundled predictions the default rule agrees with the published figure in
      six of the eight cells; `MCI->AD` and `AD->AD` resolve to a different member of their tied
      set. Because the figure normalises every panel to one global colour scale, that substitution
      also re-scales the colouring of the other panels slightly (95th-percentile vmax 0.1517 →
      0.1473). The contour structure and the reading of the figure are unchanged.

    **Reproducing the exact published rendering.** Pass `--match-published` to pin the eight samples
    shown in the shipped figure. That mode deliberately reintroduces a recorded, non-rule-based
    choice (which is why it is not the default); the anchors can be overridden under
    `gradcam.published_selection` in `config.yaml`. If an anchor cannot be located the script says
    so and falls back to the rule for that cell.

    **Before running `--recompute`**, note that `data.pred_csv` must match your own file naming: its
    default, the bundled `data/ensemble_test_predictions.csv`, uses pseudonymised names that cannot
    be joined to the images you obtain under the DUA. Generate predictions for your own data first
    (`python inference/predict_ensemble.py`) and point `data.pred_csv` at the result. The script
    checks this up front and stops with an explicit, actionable message rather than letting MONAI
    raise a bare `FileNotFoundError`.

    With the bundled predictions CSV and images renamed to match it, `--match-published` reproduces
    the shipped figure **byte-identically**.
11. **The bundled test split is de-identified and cannot be joined to your own data**: see the note
    under §4.3. `data/fixed_data_split.json` carries pseudonymised relative paths, so the ADNI
    filenames you obtain through the DUA will not match it. `train/train_grid_search.py`
    regenerates a split under `output_dir/` (the bundled file is never overwritten).
    `train/train_5fold.py` reads the split as-is without regenerating, so a non-matching split
    leaves the test set empty: the five folds run to completion and the AUC computation then
    raises `ValueError` — check the `test set: N` value in the startup `[INFO]` line before
    committing to a full run. For the paper's numbers, compare against
    `data/ensemble_test_predictions.csv` and `reference_results/`, not against a re-derived split.

---

## 8. Citation and data statement

**Software.** If you use this code, please cite the archived release:

> Tan Z, Li H, Zhang Q, Wu Y. AD-3D-CNN: CN/MCI/AD Classification of Alzheimer's Disease from
> Single-Modality Structural MRI Alone. Zenodo; 2026. doi:10.5281/zenodo.22980035

Concept DOI (all versions): 10.5281/zenodo.22980034
A machine-readable record ships as `CITATION.cff` (GitHub renders a "Cite this repository" button
from it). The corresponding author is **Zeru Tan** (Xinyang Central Hospital).

**Paper.** Citation to be added on acceptance.

**Data sources.** Please also cite or acknowledge the data sources:
ADNI (https://adni.loni.usc.edu/), MIRIAD (Malone et al., NeuroImage 2013),
OASIS-2 (Marcus et al., J Cogn Neurosci 2010).

### 8.1 What this package does and does not contain

The imaging data themselves are **not** redistributed here. ADNI and MIRIAD are governed by
data-use agreements that require an independent application and prohibit passing the images to
third parties; obtain them yourself and run `preprocess/` as described in §4.1–4.2.

What the package does contain is the derived, non-image material — which is what makes the reported
numbers checkable without the images:

| Shipped | Where | Supports |
|---|---|---|
| Fixed 144-subject test split | `data/fixed_data_split.json` | the test set used by every experiment |
| Per-sample prediction probabilities | `data/ensemble_test_predictions.csv` | every metric in Tables 2 / S5 and Figures 2–4 / S4 (§3, offline) |
| Structural-measurement panel | `data/structural_panel.csv`, `results/`, `reference_results/s9_baselines/` | Tables 3–4 / S9 without re-extracting measurements |
| 5-fold weights (Git LFS) | `weights/` | inference, external validation, Grad-CAM, occlusion |
| Occlusion intermediates and statistics | `occlusion/reference_results/` | all of §3.3 / §4.3 / S6, offline |

### 8.2 What reproduces exactly, and what cannot

Two effects mean a re-run will not be bit-identical to the published one. Neither changes any
conclusion in the paper:

- **Interactive skull stripping.** The preprocessing pipeline (paper Supplementary S1.5) uses a
  MONAI Label + 3D Slicer DeepEdit workflow with manual refinement. Subject-level segmentations
  therefore carry a human component and are not bit-wise reproducible: a reproducer's masks will
  differ from ours, and that difference can propagate downstream. Segmentation quality is reported
  as Dice 0.96–0.98 and registration quality as Dice 0.9947 ± 0.0020 (S1.9).
- **GPU floating point.** Re-running training or inference shifts results at the ~1e-4 level. The
  paper quantifies this for inference (`max|Δp0| = 9.449e-4` between sessions, argmax unchanged at
  144/144) and for retraining (three independent retrainings of the Run128 configuration gave AUC
  0.9419 / 0.9478 / 0.9512).

What does *not* depend on either effect reproduces exactly: the offline chains of §3 and §5, the
occlusion CPU statistics chain, and the occlusion masks built by `occlusion/step1_build_masks.py`
(bit-identical).
