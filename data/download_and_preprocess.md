# Data acquisition and preprocessing pipeline

This document describes how to obtain the imaging data used in this study and how to
reproduce the preprocessing pipeline of paper §2.2 / supplementary S1.

> The imaging data are governed by the ADNI / MIRIAD / OASIS data-use agreements and are **not**
> distributed with this repository. After completing the steps below, fill the preprocessing
> artifact paths in `config.yaml` to run the training/inference scripts.

---

## 1. ADNI data (training / internal validation, 954 subjects)

1. Register and apply: https://adni.loni.usc.edu/
2. Download the T1-weighted 3T MR scans of the included subjects (318 each of CN/MCI/AD,
   954 in total). The subject Image Data ID list was determined by the sample-matching script
   (AD as the anchor, gender-hard/age-soft greedy matching; see paper §2.1 / S1.4). The
   identifiers in the bundled list file are **pseudonyms** (see the de-identification notice in
   the root README).
3. The sample-matching and three-group balancing code is archived in the authors' workspace
   (see paper §2.1 / S1.4 for the full algorithm; the refactored version ships here as
   `preprocess/step1b_sample_matching.py`).

## 2. MIRIAD data (external validation, 69 subjects)

1. Apply: https://miriad.drc.ion.ucl.ac.uk/
2. Download the 69 last-visit T1-weighted scans (23 HC + 46 AD).
3. A bulk-download script ships with the package: `preprocess/step0_download_miriad.py`
   (UCL XNAT sentinel downloader, cookie authentication, rate-limited and resumable).

## 3. OASIS-2 data (external validation, 147 subjects)

1. Apply: https://www.oasis-brains.org/
2. Download the longitudinal series. The validation cohort was built as follows: one
   mpr-1 scan per subject, grouped by CDR (0: CN; 0.5: very mild dementia; 1: mild dementia),
   giving 147 subjects (73/53/21) with no repeated subjects and hence no longitudinal leakage.
3. The de-identified basic-information list (MRI ID / CDR / MMSE) ships as
   `data/oasis2_basic_info.csv`; the original demographics table (373 rows) must be downloaded
   separately.

---

## 4. Preprocessing pipeline (identical to training)

Run the steps below in order; all commands and parameters follow paper §2.2 / S1:

```
(1) DICOM -> NIfTI conversion
    dcm2niix (2D slice merge; series with any dimension < 130 are quarantined automatically)

(2) Skull stripping
    MONAI Label server + 3D Slicer client, DeepEdit active-learning loop
    (SegResNet encoder (1,2,2,4,4) + instance norm + DiceCELoss + Adam lr=1e-4)
    validation Dice 0.96-0.98

(3) N4 bias-field correction + quantile histogram matching

(4) Nonlinear registration to MNI152
    ANTs SyN: rigid pre-registration + three-level multi-resolution (iterations 50/30/20),
    CC metric, flow smoothing sigma=3
    Registration quality: Dice 0.9947±0.0020, global Pearson r 0.7802±0.0243 (954 subjects)

(5) WhiteStripe intensity normalization
    central 50% region + 20-99th percentile CSF removal -> 200-bin histogram smoothed with
    Gaussian (sigma=2) -> peak ±10% white-matter stripe -> linear mapping to 1000 -> clip [0, 3000]
    training input uses ScaleIntensityRanged(a_min=0, a_max=1100) to map to [0, 1]

(6) Generate the data-list spreadsheets (columns file_path and label)
    file_path is the absolute path of the preprocessed NIfTI, or a filename relative to data_root
```

> Note: an early workspace document recorded the registration iterations as (100,70,50)
> (syn_standard parameters); the paper and this package both use **50/30/20** (syn_fast).

The original scripts of each step are archived in the authors' workspace; the refactored,
path-parameterized versions ship here under `preprocess/`.

---

## 5. Fixed test-set split

`data/fixed_data_split.json` records the fixed 144-subject test set (48 per class,
`random_state=42`). The split was generated and persisted before the first training so that
the grid search and the 5-fold experiments share the same test set. **Use the bundled split
file as-is** (do not regenerate) so that the test results match the paper.
