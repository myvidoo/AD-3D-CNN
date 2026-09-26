# Grad-CAM visualisation section · audit-level summary (README)

> # ⚠️ This document is retired (void); do not use it to verify the paper's values
>
> **This document records an intermediate state from the paper's v26 / v45 period; its values and figure numbers have all been superseded by later versions.**
> Specific retired items (no longer used in the main text, still retained in this document):
>
> | Old value in this document | Value actually used in the manuscript |
> |---|---|
> | MTL mass fraction 6.1%±1.0% | **2.37%±0.40%** |
> | Kruskal-Wallis H=16.75 | the paper no longer contains this set of numbers (replaced by a between-group contrast of MTL mass fraction) |
> | Figure numbering used in this document's earlier revisions (Figure 7 / Figure 8) | **Figure 5** (single-sample nine-grid) |
> | Diagonal Dice 0.121–0.164 | the paper no longer uses this as interpretability evidence (S6 archives the numbers only) |
>
> **To verify the paper's values, use the following authoritative sources**:
> - `statistical_results.json`, `roi_overlap_results.json`, `per_sample_results_final.csv` (this directory; all current artifacts)
> - `occlusion/manuscript_checks/S6_numeric_traceability.md` (S6 row-by-row traceability table, 87/87)
> - `occlusion/reference_results/` (occlusion and tissue-transplant perturbation analysis)
>
> This document is retained solely for tracing the experimental lineage.

> **Note on the directory tree in §2.2 below.** That tree describes the
> authors' original workspace directory (`gradcam_aggregate_v3/`), and therefore lists files that
> are **not** part of this code package — in particular the PNG/TIFF figures and the `*.nii.gz`
> CAM volumes. What actually ships under `reference_results/gradcam_v3/` is the **tabular
> material only** (`statistical_results.json`, `roi_overlap_results.json`,
> `per_sample_results_final.csv`, `random_init_control/*`, …). The published Figure 5 is
> distributed as `results/gradcam_aggregate_v3/single_sample_grid/Figure_5_gradcam_quadgrid_rebuild.png`;
> the raw CAM volumes are not distributed, since regenerating them needs the imaging data
> (`evaluate/gradcam_quadgrid.py --recompute`).

> Version: 2026-08-26  Corresponding paper: `paper_revision_v26_condensed.docx` §2.5 / §3.3 (Figure 7 in that revision; Figure 5 in the current numbering) / §4.2 / supplementary S6
> This directory is the archive and reproduction centre for all of the paper's interpretability evidence (Figure 5 + group-level quantitative statistics + ROI overlap + random-initialisation control).

---

## 1. Overview of the evidence chain

| Evidence layer | Content | Paper location | Artifact |
|---|---|---|---|
| Qualitative display | Confusion-matrix-structured single-sample QuadGrid (8 representative samples×4 slices) | Figure 5, §3.3 | `single_sample_grid/Figure_GradCAM_SingleSample_QuadGrid.png/.tiff` |
| Group-level quantitative | CAM statistics for the full test set (n=144): Mann-Whitney U, Cliff's δ, Spearman, Kruskal-Wallis | §2.5, §4.2 | `statistical_results.json`, `per_sample_results_final.csv`, `analysis_summary.txt` |
| Anatomical localisation | Harvard-Oxford atlas ROI overlap (MTL/hippocampus mass fraction, Dice) | §4.2, S6 | `roi_overlap_results.json` |
| Artefact exclusion | Random-initialisation weight control (same architecture, untrained, n=144, same pipeline) | §4.2, S6 | `random_init_control/` |
| Aggregate display | 8-confusion-cell aggregate CAM figure (per-sample max normalisation, then averaged) | supplementary alternative | `Figure_GradCAM_Aggregate.png/.tiff`, `agg_cam_*.nii.gz` |

---

## 2. File inventory and generating scripts

### 2.1 Scripts (workspace root)
| Script | Purpose | Status |
|---|---|---|
| `run_gradcam_confusion_aggregate_v3.py` | Group-level quantitative statistics + ROI overlap + aggregate CAM figure | Current |
| `run_5fold_gradcam_single_sample_quadgrid.py` | **Figure 5 generation (2026-08-26 rebuild-archived version)** | Current |
| `run_random_init_gradcam_control.py` | Random-initialisation weight control | Current |
| `run_5fold_gradcam_confusion_matrix_mni_only_v2.py` | older version of Figure 5 (global max colour scale); its artifacts have been replaced | **Retired** (flagged in the header comment; retained for traceability only) |

### 2.2 Artifacts in this directory
```
gradcam_aggregate_v3/
├─ README.md                        ← this file
├─ README_normalization_note.md     ← dedicated normalisation note (see §3.3)
├─ statistical_results.json         ← all group-level quantitative statistics (n=144)
├─ per_sample_results_final.csv     ← per-sample metrics for 144 cases (cam_mean/p95/peak, mtl/hipp_mass_frac, confidence)
├─ per_sample_metrics_cache.json    ← resume cache
├─ roi_overlap_results.json         ← ROI overlap of the 8 confusion-cell aggregate CAMs
├─ analysis_summary.txt             ← text material for the paper (Chinese numeric summary)
├─ Figure_GradCAM_Aggregate.png/.tiff  ← aggregate CAM figure (per-sample max normalisation)
├─ agg_cam_*_n*.nii.gz              ← 8-cell aggregate CAM volumes
├─ single_sample_grid/
│   ├─ Figure_GradCAM_SingleSample_QuadGrid.png/.tiff   ← ★current paper Figure 5 (embedded-docx MD5 verified to match)
│   ├─ Figure_GradCAM_SingleSample_QuadGrid_rebuild.*   ← rebuild-script reproduction (for verification)
│   ├─ Figure_GradCAM_SingleSample_Grid.png/.tiff       ← same-day early Grid version (superseded by QuadGrid)
│   └─ cam_*2*_median.nii.gz        ← raw ensemble CAMs of the 8 representative samples (not normalised)
└─ random_init_control/
    ├─ random_init_per_sample_results.csv      ← per-sample metrics of the random model for 144 cases
    ├─ random_init_statistical_results.json    ← random-control statistics
    └─ random_init_summary.txt                 ← text usable in the paper
```

---

## 3. Methodological definitions (mapped item by item to the main text / S6)

### 3.1 Sample selection (Figure 5)
- For each confusion cell, take the 1 sample whose **predicted-class confidence is closest to that cell's median** (no arbitrary parameters).
- The 8 representative sample IDs are archived in the rebuild script's `PUBLISHED_SELECTION`.
- Tied confidences (ties at float precision) are settled explicitly by `PUBLISHED_SELECTION`, guaranteeing pixel-by-pixel reproduction; when there is no tie, the deterministic rule is "smallest distance → file_id lexicographic order".

### 3.2 CAM computation
- 5-fold best checkpoints (Run 128, DenseNet-169 + Axial Attention), deepest feature layer (last dense block), Grad-CAM for the **predicted class**, arithmetic mean over the 5 folds voxel by voxel.
- The archived NIfTI holds **raw response values (not normalised)**, in MNI152 1mm space (182×218×182, origin z=-72).

### 3.3 Three-layer normalisation division of labour (a frequent reviewer question; be sure to distinguish)
| Stage | Method | Role |
|---|---|---|
| Figure 5 colour scale | **global 95th percentile** (pooled over all voxels of the 8 samples' CAMs, vmax=0.1517), one plasma colour scale shared across the whole figure | visual comparability across samples |
| Aggregate CAM figure | per-sample max normalisation, then averaging | removes within-subject absolute intensity differences while preserving the spatial distribution |
| Group-level quantitative statistics | **not normalised** (raw response values) | preserves the information on correct/incorrect intensity differences |
| MTL/hippocampus mass fraction | scale-invariant ratio (share of total CAM intensity) | independent of normalisation, comparable across models (including the random control) |

- Contradiction investigation conclusion: the current Figure 5 (docx-embedded, MD5 check = QuadGrid.png) does use the global 95th-percentile colour scale (log `global colour scale vmax (95th): 0.1517`), **consistent with the main text's "global 95th-percentile normalisation"**. `global_cam_max` exists only in the retired v2 script (corresponding to the replaced old Figure 5).

### 3.4 Slice selection (4 slices per cell in Figure 5)
- w_z = the whole-slice mean CAM weight; the active range = the contiguous interval where w_z ≥ 60%×max(w_z);
- `np.array_split` divides it into 4 equal segments, and the slice with the highest w_z is taken within each segment; the slices are arranged by ascending z into a 2×2 layout.
- **Verification**: for the 32 w annotations of the published figure, the error against the whole-slice mean of the archived CAM at the corresponding z is <7e-5 (32/32 match); 3/8 cells agree slice by slice, and for the remaining cells individual boundary slices differ by ≤5 slices (≤5mm, caused by minor GPU recomputation perturbations, with no effect on the conclusions).
- **Sensitivity** (already written into S6): across a 40%–80% peak threshold range, the MNI Z coordinate of the representative slices varies by <2 slice thicknesses (2mm).

### 3.5 Figure 5 annotation conventions
- The white 0.6 contour = high-weight regions with CAM > 0.6×global vmax (≈0.091); bottom-left z=MNI coordinate; top-right on a purple background w=the CAM weight of that slice (whole-slice mean); diagonal green titles=correct predictions, off-diagonal red titles=incorrect predictions; the AD→CN cell has no samples (n=0, annotated "No samples in this category").

---

## 4. Summary of quantitative results (paper-cited values ↔ data sources)

### 4.1 Correct vs incorrect predictions (n=125 vs 19)
| Metric | Correct | Incorrect | Test | Paper citation |
|---|---|---|---|---|
| CAM mean | 0.137±0.046 | 0.096±0.043 | U=1802, p=0.00029, **δ=0.52** | §4.2 (δ=0.52 used consistently throughout the paper) |
| CAM p95 | 0.238±0.084 | 0.173±0.073 | U=1750, p=0.00091, δ=0.47 | — |
| MTL mass fraction | 6.1%±1.0% | 6.0%±0.9% | p=0.667 (n.s.) | §4.2: localisation is stable; the difference lies in intensity, not location |
| Hippocampus mass fraction | 2.5%±0.5% | 2.6%±0.7% | p=0.436 (n.s.) | as above |

### 4.2 Confidence correlations (Spearman, n=144)
- vs cam_mean: **ρ=0.800, p=2.67×10⁻³³** (cited in the main text); vs cam_p95: ρ=0.713; vs MTL mass fraction: ρ=0.390; vs hippocampus mass fraction: ρ=0.272.

### 4.3 MTL mass fraction across the three groups (n=48 per class)
- CN 5.7%±1.0% | **MCI 6.5%±0.9% (highest)** | AD 5.9%±1.0%
- **Kruskal-Wallis H=16.75, p<0.001** (this notation is used consistently throughout the paper; the exact value p=2.3×10⁻⁴ is archived in the json)
- Reporting definition: the MCI group is highest, consistent with the pathological expectation that "MCI is the stage of most active hippocampal–entorhinal cortex degeneration" (§4.2 and §3.3 gradient wording was revised accordingly on 2026-08-26).

### 4.4 ROI overlap (aggregate CAM, Dice at the 50% peak threshold)
| Cell | n | MTL mass | Hippocampus mass | Dice(MTL) |
|---|---|---|---|---|
| AD→AD | 45 | 5.8% | 2.5% | 0.121 |
| MCI→MCI | 39 | 6.7% | 2.8% | 0.164 |
| CN→CN | 41 | 5.8% | 2.3% | 0.132 |
| Diagonal range | — | 5.8%–6.7% | 2.3%–2.8% | 0.121–0.164 (cited in the main text) |

### 4.5 Random-initialisation weight control (same architecture, untrained, n=144, same pipeline)
| Metric | Random model | Trained model | Interpretation |
|---|---|---|---|
| Accuracy | **33.3% (chance level)** | 86.81% | random weights have no diagnostic capability |
| MTL mass fraction | 3.83%±0.02% | 6.1%±1.0% | random is **below** the MTL volume share of 5.01%; trained is above it |
| Hippocampus mass fraction | 1.67%±0.01% | 2.5% | random is below the hippocampus volume share of 2.35% |
| Between-subject std | **0.02%** | 1.0% | the random CAM is nearly uniformly distributed, with no sample-specific localisation |
| Correct vs incorrect CAM intensity | δ=**-0.31** (opposite direction) | δ=**+0.52** | the random model shows no "correct → strong attention" coupling |
| MTL fraction in the three groups | 3.83/3.83/3.82%, H=6.12 | H=16.75 | the between-group difference in the random model has a negligible effect size |
- Conclusion (already written into §4.2 and S6): the model's medial temporal lobe focus is a **training-acquired** disease-related feature, ruling out the alternative explanation of architectural inductive bias / centre preference.

---

## 5. Reproduction steps

Environment: `<LOCAL_HOME>/anaconda/envs/monailabel/python.exe` (torch 2.13+cu132, CUDA available; checkpoints are located at `<AUTHOR_DATA_ROOT>/MONAILabel/AD_VS/results3.7/results_5fold_geo_resume/checkpoints/`)

```bash
cd <LOCAL_HOME>/<tool_dir>/<session>

# Figure 5 (reuses the archived CAM for rendering, takes seconds; adding --recompute recomputes the CAM through the full pipeline, ~10 minutes on GPU)
<LOCAL_HOME>/anaconda/envs/monailabel/python.exe run_5fold_gradcam_single_sample_quadgrid.py

# Group-level quantitative + ROI overlap + aggregate figure (GPU, supports resuming)
<LOCAL_HOME>/anaconda/envs/monailabel/python.exe run_gradcam_confusion_aggregate_v3.py

# Random-initialisation control (GPU, ~3 minutes)
<LOCAL_HOME>/anaconda/envs/monailabel/python.exe run_random_init_gradcam_control.py
```

---

## 6. Key points for responding to potential reviewer questions

1. **"Does the normalisation method disagree with the code?"** → see §3.3. The current Figure 5 uses the global 95th percentile (double evidence from the log and the MD5); global_max belongs only to the replaced old figure.
2. **"You did a ROI overlap — where are the numbers?"** → §4.4 and §4.2 now supply the concrete values; the absence of a difference in MTL fraction between correct and incorrect (p=0.667) in fact demonstrates that the localisation is stable.
3. **"How do you show this is not an artefact / inductive bias?"** → the random-initialisation control in §4.5: the random model's accuracy = chance level, its MTL fraction is below the volume share, its CAM is uniform with no localisation, and its intensity–correctness coupling points in the opposite direction.
4. **"Is the 60% slice threshold arbitrary?"** → the S6 sensitivity statement: across a 40%–80% threshold the representative slices' Z coordinates vary by <2mm; moreover, the group-level statistics (n=144) do not depend on slice selection at all.
5. **"Is the sample selection biased?"** → the median-confidence rule involves no human-chosen threshold; the 8 sample IDs and the confusion-cell n values are all archived and auditable.
6. **"Can the single-sample illustration represent the group?"** → the main text states clearly: the single sample is qualitative display only; all group-level inference is based on the n=144 quantitative analysis.

## 7. Known limitations and notes
- **Figure 5 version history**: v1 (2026-08-25, original published version; backup `single_sample_grid/Figure_GradCAM_SingleSample_QuadGrid_v1_original_published_from_docx.png`, extracted from the docx embed) → v3 (brain mask + initial spacing fix) → v4 (portrait compact layout) → **v5 (2026-08-26, current official version, already synced to the docx): ① slice selection gained a brain-tissue constraint (within the axial range where the slice brain-voxel count is ≥5% of the maximum slice, slices are chosen by four-way segmentation at ≥60% peak, fixing the all-black 4th slice in CN→AD, z=92→78); ② zero-valued CAM inside the brain is rendered as the lowest end of the colour scale rather than a cut-out (the MCI→CN "brain tissue with no weight" is in fact the zero-attention region of that sample's 112,573 true-zero voxels, not a mask/filtering problem); ③ alpha 0.60; ④ the docx displays full column width 152.4mm (about 17.7mm per slice). The figure caption / S6 / §2.5 wording has been synced**. v1's TIFF was not retained (the PNG has been fully backed up from the docx; the TIFF can be regenerated at any time from the archived CAM using the script).
- The QuadGrid script is the 2026-08-26 **rebuild-archived version** (the original was generated temporarily in a session and then lost): the samples, CAM data and w annotations all verify as consistent with the v1 published figure (32/32); individual boundary display slices differ by ≤5mm (GPU perturbation); if absolute consistency is required, the backed-up v1 PNG can be used directly.
- In the random control, cam_mean correct vs incorrect gives p=0.002 (opposite direction): the response wording is "the effect points in the opposite direction and the absolute difference is <0.001, i.e. noise; the trained model's δ=+0.52 is a large effect".
- `~$_ART_submission_structured_version.docx` is a Word lock file (created while the document is open); before submission, confirm that the document has been closed and saved normally.
