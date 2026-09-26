# Normalization and reproducibility note (merged)

The content of this file was merged on 2026-08-26 into **README.md** in the same directory
(audit-level summary). Please refer directly to:

- Three-layer normalization division of labour and contradiction analysis → `README.md` §3.3
- QuadGrid script rebuild and definition verification → `README.md` §2, §3.4, and the
  docstring of `run_5fold_gradcam_single_sample_quadgrid.py` in the workspace root

Key conclusion (kept for reference): the paper's Figure (docx-embedded, MD5 = QuadGrid.png)
uses the **global 95th-percentile** color scale (log vmax = 0.1517), consistent with the
main text's "global 95th-percentile normalization"; `global_cam_max` exists only in the
retired script `run_5fold_gradcam_confusion_matrix_mni_only_v2.py` (the replaced old figure).
