# Preprocessing pipeline — operating instructions (paper 2.1–2.2 / supplementary S1)

This directory implements the five preprocessing modules and the sample-preparation
step of Figure 1, "Stage 1":

```
DICOM retrieval / series selection → dcm2niix conversion → MONAI Label interactive skull stripping
→ N4 bias-field correction → quantile histogram matching → ANTs SyN registration to MNI152
→ WhiteStripe intensity normalisation → (training-ready data, directory suffix *_fast_ws)
```

**Registration QC over all 954 subjects (S1.9)**: brain-mask Dice 0.9947±0.0020,
CC 0.7802±0.0243, MAE 635.3±52.0; 99.1% of subjects have Dice ≥ 0.990.

---

## 0. Environment and paths

- Registration and baseline extraction use **ants_env**
  (`pip install ants nibabel SimpleITK scikit-learn pandas openpyxl scipy pyyaml`).
- The conversion scripts require **dcm2niix** on `PATH`.
- Paths are configured centrally in the `preprocess:` section of `../config.yaml`;
  `step3`/`step4` additionally accept command-line overrides.
- The scripts that were copied over unchanged (step1/1b/1c/2/2b/5/5b/6a/6b) hold local
  paths at the line numbers listed in their header comments — edit them before running.

## 1. step1 — DICOM retrieval, series selection and format conversion (S1.1–S1.2)

`step1_dicom_to_nifti.py`: locates each subject's DICOM folder from the Image Data ID;
filters modalities to MR only; prioritises the keywords MPRAGE/SPGR/FSPGR/BRAVO and
excludes T2/FLAIR/DTI/DWI/fMRI/scout images; selects the best T1 series by a weighted
score; converts with dcm2niix (gzip, BIDS-compatible JSON sidecar); automatically
quarantines any series with a dimension below 130 slices.

```bash
# Edit the paths in the script's header configuration block first (see the header comment)
python step1_dicom_to_nifti.py
```

`step1c_classify_by_sequence.py`: classifies scans by acquisition type (tissue before
skull stripping).

## 2. step1b — balanced sample matching (2.1 / S1.4)

`step1b_sample_matching.py`: uses the AD group as the anchor and greedily matches
318 subjects from each of the CN and MCI groups under a **hard sex constraint** and a
**soft age criterion** (score function combining the between-group mean difference, the
standard-deviation difference and the Kolmogorov–Smirnov p-value). After matching the
three groups have identical sex composition (166 M / 152 F, χ²<0.001) and comparable age
(Kruskal–Wallis H=5.336, p=0.069); pairwise Mann–Whitney U: AD vs MCI U=54234.5, p=0.113;
AD vs CN U=55719.5, p=0.026 (not significant after Bonferroni correction), Cohen's d 0.11–0.17.

`step6a_demographics_stats.py` / `step6b_generate_table1.py`: reproduce Table 1 (xlsx,
booktabs layout).

## 3. step2 — interactive skull stripping (S1.5)

Built on **MONAI Label** (server side: SegResNet, DeepEdit paradigm) plus **3D Slicer**
(client side, interactive labelling): an incremental annotate–train–infer–correct
active-learning loop; inputs resampled to 1.0 mm³ and 192³ voxels; DiceCELoss;
best validation Dice 0.98 (0.96–0.98 across the individual models). Service deployment
and the interactive workflow are documented in `monailabel_skullstrip_operations.md`.

### 3.1 MONAI Label server configuration (`monailabel_configs/deepedit_20260521.py`)

This file is a server-side model configuration for the MONAI Label application directory
`...\sample-apps\radiology\lib\configs\` (DeepEdit paradigm), modified by the authors from
the stock MONAI Label `deepedit`: **a SegResNet branch was added (absent upstream, including
`segresnet_large`) and made the default**, and the input space changed from 160³ to 192³
(to fit 32 GB VRAM). To deploy, copy this file into the `configs` directory above;
**the filename must not be changed** — MONAI Label registers models by filename, so the
launch argument `--conf models deepedit_20260521` must match the filename.

Server launch command (the `--app` and `--studies` paths below are examples from the
author's machine; substitute your own):

```bash
monailabel start_server --app "<TOOL_DIR>/monailabel/sample-apps/radiology" --studies "<AUTHOR_DATA_ROOT>/datasets/ADNI/NIfTI_organized_monailabel_test/0.5" --conf models deepedit_20260521 --conf network segresnet
```

The client **3D Slicer** needs the MONAI Label module installed (available through the
Slicer Extensions Manager); once connected to the server it supports interactive labelling.
The detailed interactive procedure is in `monailabel_skullstrip_operations.md`.

- `step2_export_skullstripped.py` (ADNI) / `step2b_export_miriad.py` (MIRIAD): apply the
  segmentation labels to the original images and export the brain parenchyma (preserving
  spatial information and data type; existing outputs are skipped automatically).

> The segmentation model uses only brain/non-brain anatomical annotation, the interaction
> samples are chosen independently of the diagnostic label, and the outputs carry no class
> information — so no diagnostic-label leakage occurs at the preprocessing stage
> (see paper §2.2).

## 4. step3 — batch ANTs SyN-fast registration (S1.6) ★ core script

`step3_register_syn_fast.py` (parameterised from the v2.0.4 / 20260525 script actually used
by the authors; the interactive original is archived in `legacy/` — for this release its
comments, docstrings and console messages were translated to English, with its logic,
constants and file-path literals unchanged).

**Pipeline**: N4 bias-field correction (shrink factor 2, threshold 1e-7, iterations
(50,50,50,50), constrained by the template brain mask) → quantile histogram matching
([0.5, 99.5]%, 1024 quantiles, preserving pathological information) → ANTs SyN
registration to MNI152 (1×1×1 mm³, cross-correlation CC metric).

**Default strategy `syn_fast` (the one used in the paper)**: `reg_iterations=(50, 30, 20)`,
`syn_metric='CC'`, `syn_sampling=4`, `flow_sigma=3`, `total_sigma=0`. The other strategies
(syn_standard/syn_precise etc.) remain available.

**Outputs** (subdirectories created automatically under `--output`):

| Directory | Content |
|---|---|
| `registered/` | `<file_id>.nii.gz` — registered image (MNI152 space) |
| `warpfields/` | `<file_id>_warp.nii.gz` — forward deformation field |
| `overlays/` | `<file_id>_overlay.png` — registration overlay QC image |
| `reports/` | `completed_files.txt` resume manifest; `registration_report_*.csv` per-case Dice/CC/MAE |

**Resume**: completed cases are detected by both `completed_files.txt` and the presence of
the output files, so after an interruption **re-running the same command resumes**.

```bash
# Full batch (954 subjects, ≈400 s per subject, RTX 5090)
python step3_register_syn_fast.py \
    --input  <skull-stripped NIfTI directory> \
    --template <MNI152 skull-stripped template mni.nii.gz> \
    --output <output directory>

# Smoke test (single case; checks that the environment works)
python step3_register_syn_fast.py --input ... --template ... --output ... \
    --single <substring of a filename>

# If Jacobian maps are needed (not produced by the paper pipeline by default; the S9 lateral
# ventricle Jacobian volume is computed on demand by the baseline scripts)
python step3_register_syn_fast.py ... --save-jacobian
```

> ⚠ Note on consistency with the paper text: paper S1.6 states "three-level multi-resolution
> optimisation (iterations 100/70/50)", which corresponds to the syn_standard strategy; the
> data the authors actually produced (directory suffix `..._ants_20260525_fast_ws`) used
> **syn_fast (50/30/20)**. To reproduce, follow syn_fast as implemented in this script.
> This discrepancy should be reconciled at the paper's proof stage.

**QC**: per-case Dice/CC/MAE are written to `reports/`; CC ≥0.85 excellent / ≥0.80 good /
≥0.75 acceptable. `step5_registration_qc.py` parses the logs and summarises the three-group
statistics; `step5b_registration_anova.py` recomputes the S1.9 three-group ANOVA + Tukey HSD
(classical equal-variance formulation; the authoritative outputs are in
`../reference_results/registration_quality_S1.9/`).

## 5. step4 — WhiteStripe intensity normalisation (S1.7)

`step4_whitestripe.py` (turned from the 2.1.1 notebook into a CLI with the algorithm
line-for-line unchanged): central 50% of voxels (avoiding edge interpolation artefacts) →
20th–99th percentile (removing CSF/background) → 200-bin histogram smoothed with a Gaussian
(σ=2) → the tallest peak is the white-matter peak → the mean of the white-matter stripe at
peak ±10% is mapped linearly to 1000 → clipped to [0, 3000].

```bash
# Batch normalisation (output filenames get the _ws suffix; existing files are skipped = resume)
python step4_whitestripe.py --input <the registered/ directory from step3> --output <output directory>

# Smoke test
python step4_whitestripe.py --input ... --output ... --max-files 2

# Quality assessment: white-matter peak alignment (target 1000; paper S9 white-matter peak
# CN 1018.8±7.9 / MCI 1018.1±12.9 / AD 1017.4±14.3)
python step4_whitestripe.py --evaluate --input <normalised directory>
```

> The step4 output directory is the final training data pointed to by `data.data_root` in
> `config.yaml` (the `*_fast_ws` directory).

## 6. Hand-off to training / evaluation

1. Generate the 954-subject list spreadsheet (`file_path` pointing at `*_ws.nii.gz`,
   `label` ∈ {0, 0.5, 1}) and enter it as `data.adni_excel` in `config.yaml`; an example
   format ships as `../data/data_lists/ADNI_954_subject_master_list_*.xlsx`.
2. The fixed test split `data/fixed_data_split.json` (seed=42, 144 subjects held out first)
   ships with the package; training and the grid search read it automatically.
3. MIRIAD / OASIS-2 go through the same pipeline (step2b → step3 → step4). Their directories
   are scanned directly and the label is taken from the subdirectory name
   (MIRIAD {0,1}; OASIS-2 {0, 0.5, 1}).

## 7. Preprocessing smoke-test checklist

| Smoke test | Command | Expected |
|---|---|---|
| Single registration | step3 `--single <substring>` | one file each in registered/warpfields/overlays, Dice ≥0.98 |
| Two-case normalisation | step4 `--max-files 2` | two `*_ws.nii.gz`; `--evaluate` white-matter peak ≈1000±50 |
| QC summary | step5/step5b | three-group Dice/CC/MAE statistics + ANOVA table (compare with S1.9) |
| Table 1 | step6a → step6b | demographics statistics + Table1.xlsx (compare p=0.026/0.113) |
