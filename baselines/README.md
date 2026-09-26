# baselines/ — Controlled baselines and the structural measurement panel

Paper **§2.7 (methods)**, **§3.5 (results)** and **supplementary S9**. This directory provides
the extraction of the structural measurements, the fitting/evaluation code and the
measurement-definition correctness checks for five controlled baselines.

---

## 1. Why controlled baselines are needed

So that the performance claim for "single-modality sMRI three-class classification" can be
independently audited, this directory builds two families of comparators that are
**same-source** with the main model:

| ID | Baseline | Input features | Purpose |
|---|---|---|---|
| B1 | Age + sex logistic regression | `age, sex_m` (2) | Demographic floor; doubles as the "performance is not demographic leakage" negative control |
| B2 | Structural measurement panel, logistic regression | 6 structural measurements | Representative of regional-volume approaches |
| B3 | Structural measurement panel, RBF-SVM | 6 structural measurements | Non-linear counterpart of the above |
| B4 | Panel + demographics, logistic regression | 8 features | Tests whether demographics add information |
| B5 | Panel + demographics, RBF-SVM | 8 features | As above, non-linear |

Both families **use only the same input modality as the main model** (a single T1-weighted
sMRI) and **the same preprocessing pipeline**, with no PET, CSF or cognitive scores — a
strictly same-source comparison.

---

## 2. The 6 structural measurement features

| # | Feature | Definition |
|---|---|---|
| 1 | `gmr_hippocampus_700` | Bilateral hippocampal gray-matter volume as a percentage of whole-brain gray-matter volume |
| 2 | `gmr_amygdala_700` | Bilateral amygdala gray-matter share |
| 3 | `gmr_parahippocampal_700` | Bilateral parahippocampal gyrus (anterior + posterior) gray-matter share |
| 4 | `gmr_mtl_700` | Gray-matter share of the **union** of the three above (the medial temporal lobe, MTL) |
| 5 | `vmr_latventricle` | Bilateral lateral-ventricle Jacobian-modulated volume as a percentage of whole-brain modulated volume |
| 6 | `brain_gm_ml` | Absolute whole-brain gray-matter volume (mL) |

---

## 3. Measurement definitions

### 3.1 Gray matter: template-space tissue occupancy

All images have been through the standard preprocessing pipeline and resampled to
**MNI152 1 mm template space** (182×218×182, WhiteStripe intensity normalisation).
Anatomical position is therefore aligned to a common coordinate system, so **the same set of
Harvard-Oxford masks points at the same anatomical structure for every subject**, and atrophy
shows up as "fewer voxels inside that structure look like gray matter".

```
                            #{ x ∈ ROI : I(x) > τ }              (tissue voxels in the region)
regional GM share (%) = 100 × ──────────────────────────────
                            #{ x ∈ whole brain : I(x) > τ }      (tissue voxels in the brain)
```

Voxel size is 1×1×1 mm³, so **the voxel count is numerically equal to mm³**.

### 3.2 Three independent grounds for the tissue threshold τ = 700 (it is not an arbitrary parameter)

1. **WS normalisation makes the WM peak highly consistent across subjects**, so a fixed
   threshold is comparable. Measured WM modes: CN 1018.8±7.9 (CV 0.77%),
   MCI 1018.1±12.9 (1.26%), AD 1017.4±14.3 (1.40%).
2. **τ = 700 ≈ 0.6875 × the WM mode**, lying between the gray-matter peak and CSF.
3. **At this threshold whole-brain tissue occupies 83–85% of the template brain**, inside the
   80–85% physiological volume range for gray + white matter in healthy adults — a criterion
   **independent of any atrophy finding**.

Sensitivity analysis: τ = 600 / 700 / 800 have all been computed and archived; the direction
of the conclusion is unchanged.

### 3.3 Lateral ventricles: Jacobian-modulated volume share

The lateral ventricles are CSF spaces, so a gray-matter threshold does not apply to them —
intensity inside the cavity is always at CSF level, so however large the cavity is,
`#{I > 700}` stays near zero. A Jacobian-modulated volume from the non-linear deformation
field is used instead:

```
                              Σ_{x ∈ lateral ventricles} |det(∂T/∂x)|
LV volume share (%) = 100 × ────────────────────────────────────────
                              Σ_{x ∈ whole brain}        |det(∂T/∂x)|
```

Using ANTs' own `create_jacobian_determinant_image(template, warp, do_log=False)`.
Measured on ADNI, AD/CN = 1.318, p = 7.3×10⁻²³, robustly reproducing ventricular enlargement;
this measure also serves as the **positive control** that the Jacobian metric has not failed
globally in this pipeline.

### 3.4 ROI labels (⚠ critical pitfall)

Taken from the Harvard-Oxford probabilistic atlas (maximum-probability threshold 25%, 1 mm):

| Structure | **Image label value** | Atlas XML index |
|---|---|---|
| Left / right hippocampus | 9 / 19 | 8 / 18 |
| Left / right amygdala | 10 / 20 | 9 / 19 |
| Parahippocampal gyrus (anterior / posterior) | 34 / 35 | 33 / 34 |
| Left / right lateral ventricle | 3 / 14 | 2 / 13 |
| Brainstem (control) | 8 | 7 |

> **Image label value = XML index + 1**
>
> `index="0"` in `HarvardOxford-Subcortical.xml` is "Left Cerebral White Matter" and the XML
> **has no background entry**, whereas label value 0 in the NIfTI file is background. If you
> wrongly assume "image value = XML index" and take the hippocampus as 8/18, you actually get
> the **brainstem** — a pitfall that silently produces wrong results.
>
> Two-hypothesis blind test (`verify_atlas_identity.py`; 21 known anatomical centroids,
> 30 mm tolerance): "value k = XML[k]" matches **15/21**, "value k = XML[k−1]" matches
> **21/21** → the off-by-one mapping holds. The decisive case: image value 8 gives
> 38,610 mm³ with centroid (0.6, −31.0, −34.2), which can only be the brainstem
> (the hippocampus is about 11,000 mm³ with centroid near (±25, −22, −14)).

---

## 4. Scripts

| Script | Function | Depends on |
|---|---|---|
| `extract_structural_panel_s9.py` | Structural measurement extraction for all three cohorts, 1170 subjects (≈13 min) | images + warp fields |
| `run_baselines.py` | Fitting, bootstrap CI and paired tests for the 5 baselines (≈1 min) | panel CSV + demographics |
| `run_baselines_tuned.py` | v58 tuned version = the paper's main definition (nested 5-fold B4 0.7115±0.0099, the value cited in the paper) | panel CSV + demographics |
| `run_baselines_5fold_fixedhp.py` | 5-fold reproduction with hyperparameters fixed (the S9 parenthetical values 0.5044–0.7110; output asserted against the archived log) | panel CSV + data split |
| `verify_atlas_identity.py` | Two-hypothesis blind test of the atlas label mapping (**correctness evidence ①②**) | atlas only |
| `verify_structural_panel.py` | Threshold-calibration grounds + directionality + threshold sensitivity (**correctness evidence ③④⑤**) | panel CSV only |

### Run order

```bash
# 1) Extract the structural measurements (needs the preprocessed images; ≈13 min)
python baselines/extract_structural_panel_s9.py

# 2) Fitting and statistics (outputs every value in paper Tables 3/4 and S9.4; ≈1 min)
python baselines/run_baselines.py

# 2b) 5-fold reproduction with hyperparameters fixed (the paper's S9 parenthetical values;
#     runs offline, ≈1 min)
python baselines/run_baselines_5fold_fixedhp.py

# 3) Correctness verification (runs standalone; check ① needs only the atlas, ② only the
#    output of step 1)
python baselines/verify_atlas_identity.py
python baselines/verify_structural_panel.py
```

> If you only want to reproduce the paper's numbers without re-extracting the measurements,
> you do not need to place any files manually: the package already contains a copy of
> `structural_panel.csv` in each of `data/` and `results/` (three copies in total including
> `reference_results/s9_baselines/`, bit-identical MD5). `data/` is read by
> run_baselines_tuned.py / verify_tuned_baselines.py / run_baselines_5fold_fixedhp.py, and
> `results/` by run_baselines.py and verify_structural_panel.py.

---

## 5. Chain of correctness evidence

| # | Evidence | Criterion | Measured |
|---|---|---|---|
| ① | Threshold is not arbitrary | Tissue fraction falls in the physiological range | 83–85% at τ=700, inside the 80–85% gray+white range ✓ |
| ② | Atlas mapping is correct | Anatomical hits under the two hypotheses | 15/21 (no shift) vs **21/21 (shift by 1)** ✓ |
| ③ | Directionality reproduces the literature | GM AD<CN, ventricular enlargement, MCI intermediate | hippocampus 0.901 / lateral ventricle 1.318 / all six features KW p<10⁻¹⁸ ✓ |
| ④ | Mask error is excluded | Direction stability across 5 mask variants | hippocampal Jacobian measure stays at 0.98–1.01 throughout → the failure is in the deformation field, not the mask ✓ |
| ⑤ | Threshold insensitivity | Direction consistency at τ=600/700/800 | all ROIs AD<CN at all three thresholds, no sign flips ✓ |
| ⑥ | Cross-cohort transferability | Zero-fine-tuning extrapolation AUC | MIRIAD 0.9395 / OASIS-2 0.9022 ✓ |

> The full diagnostic process behind evidence ④ is in paper supplementary **S9.6**: in the
> initial design every ROI used the Jacobian-modulated volume, but that measure fails for the
> **hippocampus** (AD/CN = 1.01), while under the same deformation field the amygdala (0.90),
> parahippocampal gyrus (0.94) and lateral ventricle (1.27) all point the right way. After
> exhausting 5 mask variants (probability threshold 25%→50%; erosion by 2/3/4 voxels; excluding
> voxels within 3/5 voxels of the lateral ventricle; adding a tissue threshold) the hippocampus
> still stayed at 0.98–1.01, so the failure was attributed to **the deformation field of the
> SyN "fast" registration itself** (the hippocampus is the most elongated gray-matter structure
> in the brain and is the easiest to smooth away during registration) rather than to a mask
> error. Switching to the template-space tissue-occupancy measure immediately restored the
> expected direction.

---

## 6. Output files (`results/`)

| File | Content |
|---|---|
| `structural_panel.csv` | Per-subject structural measurements, 1170 subjects × 42 columns (ADNI 954 / MIRIAD 69 / OASIS-2 147) |
| `baseline_test_set_performance.csv` | All metrics, CIs and paired differences against the ensemble, for the 5 baselines on the 144-subject test set |
| `cnn_ensemble_test_set_performance.csv` | Ensemble metrics on the same test set (for the paired comparison) |
| `baseline_5fold_cv.csv` | Fold-level AUC from 5-fold CV (including the fixed-hyperparameter version, same source as section [A] of `_cv5_fixedhp.log`) |
| `baseline_5fold_cv_fixedhp.csv` | Fixed-hyperparameter 5-fold values (output of run_baselines_5fold_fixedhp.py; per-fold values asserted against the archived log) |
| `baseline_external_cohorts.csv` | Structural-panel baseline AUC on the external cohorts |

---

## 7. Main results (144-subject fixed test set)

| Model | # features | Accuracy | Macro-F1 | Macro-AUC (95% CI) | CN / MCI / AD class AUC | ΔAUC vs ensemble |
|---|---|---|---|---|---|---|
| **CNN 5-fold ensemble** | 3D | **0.8681** | **0.8675** | **0.9630** (0.9357–0.9845) | 0.9529 / 0.9542 / 0.9818 | — |
| B1 age+sex LR | 2 | 0.3819 | 0.3590 | 0.5284 (0.4610–0.5940) | 0.5254 / 0.5091 / 0.5506 | +43.5 |
| B2 structural panel LR | 6 | 0.5139 | 0.5058 | 0.6941 (0.6275–0.7584) | 0.7480 / 0.5716 / 0.7626 | +26.9 |
| B3 structural panel SVM | 6 | 0.4861 | 0.4822 | 0.6492 (0.5817–0.7197) | 0.6957 / 0.5590 / 0.6929 | +31.4 |
| **B4 panel+demo LR** | 8 | 0.5139 | 0.5046 | **0.7079** (0.6500–0.7648) | 0.7917 / **0.5284** / 0.8036 | +25.5 |
| B5 panel+demo SVM | 8 | 0.5417 | 0.5330 | 0.7075 (0.6531–0.7661) | 0.8101 / **0.5200** / 0.7925 | +25.5 |

> The table above is the **paper's main definition = the standard grid-tuned version**
> (`run_baselines_tuned.py`: LR C∈{0.1,1,10}; RBF-SVM C∈{0.1,1,10} × gamma∈{scale,1e-3,1e-2};
> an inner 5-fold CV selects the parameters on the 810 non-test subjects;
> B1 C=1.0, B2 C=0.1, B3 C=1.0/gamma=scale, B4 C=0.1, B5 C=10/gamma=0.01).
> The untuned control (C=1.0/gamma=scale) is written by `run_baselines_tuned.py` to
> `baseline_test_set_performance_untuned_recheck.csv`; `run_baselines.py` implements the
> original untuned protocol.

Every ΔAUC has a paired bootstrap p < 0.001 (2000 paired stratified bootstrap resamples;
resolution floor 0.0005).

**Key readings**
1. B1 sits near chance (0.5284) → performance cannot be explained by age and sex, ruling out
   demographic leakage.
2. The best baseline, B4, reaches macro-AUC 0.7079; the ensemble is **25.5 percentage points**
   higher.
3. The structural panel's **MCI AUC is only 0.5200–0.5716 (near chance)**, versus 0.9542 for
   the ensemble → **the discriminative gain comes mainly from MCI**.

### External cohorts (zero-fine-tuning extrapolation)

| Cohort | n | View | Full panel (6 features) | CNN ensemble |
|---|---|---|---|---|
| MIRIAD | 69 | AD vs HC | **0.9395** | 0.8790 |
| OASIS-2 | 94 | CN vs AD | **0.9022** | 0.7567 |

> ⚠ **A limitation that must be reported honestly**: on the CN/AD two-class comparison in the
> external cohorts the structural measurement panel discriminates **better** than this CNN
> model, and that performance is driven by hippocampal and MTL volume. This shows that the
> model has **no advantage over coarse volumetric measurement in cross-centre transfer**
> (already stated in paper §4.3).

---

## 8. Reproduction environment

```bash
conda create -n ants_env python=3.10
conda activate ants_env
pip install ants nibabel SimpleITK scikit-learn pandas openpyxl scipy pyyaml
```

Measured environment: `ants_env` (Python 3.10.18; ants 0.6.1, SimpleITK 2.5.2, nibabel 5.3.2,
scikit-learn, pandas, openpyxl).

`config.yaml` requires: `data.data_root`, `data.adni_excel`, `data.adni_demo`,
`data.miriad_root`, `data.oasis2_root`, plus the three `atlas.*` entries.
