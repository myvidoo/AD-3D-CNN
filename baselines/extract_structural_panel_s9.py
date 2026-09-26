# -*- coding: utf-8 -*-
"""Controlled baselines (paper §2.7 / §3.5 / supplementary S9) — subject-level structural measurement panel extraction.

This script generates the **6 structural measurement features** used by the paper's
"baselines two/three (B2/B3)" and "baselines four/five (B4/B5)",
covering ADNI 954 + MIRIAD 69 + OASIS-2 147 = 1170 subjects.

================================================================================
[measurement design] why measurements can be taken directly in template space
================================================================================
All images have been through the standard preprocessing pipeline and resampled to MNI152 1 mm
template space (182×218×182, WhiteStripe intensity normalisation). Anatomical position is
aligned to a common coordinate system, so **the same set of Harvard-Oxford masks points at the
same anatomical structure for every subject**, and atrophy shows up as "fewer voxels inside
that structure look like gray matter".

This measurement therefore **does not rely on the non-linear deformation (warp) field
(Jacobian)** — the "local volume change relative to the template" that the warp field provides
is already captured by the change of tissue class in template space. Measurements on this
pipeline show the latter to be more reliable (see `baselines/README.md` §3.1 "Gray matter:
template-space tissue occupancy" and §3.3 "Lateral ventricles: Jacobian-modulated volume
share", and the methodological account in paper supplementary S9.6).

  (A) Gray-matter structures (hippocampus / amygdala / parahippocampal gyrus / their union, the MTL)
      Regional gray-matter volume share (%) = 100 × #{x ∈ ROI : I(x) > τ} / #{x ∈ whole brain : I(x) > τ}
      Voxel size is 1 mm³, so the voxel count is the volume (mm³).

      Three independent grounds for the tissue threshold τ = 700 (it is not an arbitrary parameter):
        ① WhiteStripe normalisation makes the WM peak (histogram mode) highly consistent across
           subjects (measured CN 1018.8±7.9, MCI 1018.1±12.9, AD 1017.4±14.3, CV 0.77%–1.40%),
           so a fixed threshold is comparable;
        ② τ = 700 ≈ 0.6875 × the WM mode, lying between the gray-matter peak and CSF;
        ③ at this threshold whole-brain tissue occupies about 83–85% of the template brain, inside
           the 80–85% physiological range for gray + white matter in healthy adults (a criterion
           independent of any atrophy finding).
      Sensitivity analysis: τ = 600 / 700 / 800 have all been computed and archived.

  (B) Lateral ventricles (CSF spaces, where a gray-matter threshold does not apply)
      Ventricular enlargement shows up as the cavity boundary moving outwards, while intensity
      inside the cavity is always at CSF level: however large the cavity is, #{I > 700} stays
      near 0. A Jacobian-modulated volume is therefore used instead:
        V = Σ_{x ∈ lateral ventricles} |det(∂T/∂x)|       (voxel 1 mm³)
      Measured on ADNI, AD/CN = 1.32, p = 7.3×10⁻²³, robustly reproducing ventricular
      enlargement. This measure also serves as the **positive control** that the Jacobian metric
      has not failed globally in this pipeline.

================================================================================
[ROI labels] Harvard-Oxford (maximum-probability threshold 25%, 1 mm)
================================================================================
  ⚠ image label value = XML index + 1
     index=0 in HarvardOxford-Subcortical.xml is "Left Cerebral White Matter", and the XML
     **has no background entry**; whereas label value 0 in the NIfTI file is background. Hence
     the overall shift by 1.

     Two-hypothesis blind test (against 21 known anatomical centroids, tolerance 30 mm):
       Hypothesis A "value k = XML[k]"      15/21 hits
       Hypothesis B "value k = XML[k−1]"    21/21 hits   ← holds
     The decisive case: image value 8 = 38,610 mm³ with centroid (0.6,−31.0,−34.2) → it can only
     be the brainstem, not the hippocampus

  Subcortical label values: hippocampus 9/19, amygdala 10/20, lateral ventricle 3/14, brainstem 8
  Cortical label values:   parahippocampal gyrus anterior 34, posterior 35

  The label mapping is self-checked anatomically by MNI centroid (bilateral-union definition):
    hippocampus (1.0,−21.4,−14.4), amygdala (1.8,−4.3,−18.0), parahippocampal gyrus (0.5,−16.9,−25.4)

================================================================================
[output] 1180 columns × per-subject records → structural_panel.csv (42 columns)
================================================================================
  gmr_{roi}_{τ}      regional gray-matter volume share (%), τ ∈ {600,700,800}, main definition τ=700
  gm_{roi}_{τ}       absolute regional tissue voxel count (for auditing)
  tissue_brain_{τ}   whole-brain tissue voxel count (normalisation denominator)
  V_latventricle     lateral-ventricle Jacobian-modulated volume
  V_brain_jac        whole-brain Jacobian-modulated volume
  vmr_latventricle   lateral-ventricle volume share (%) = V_latventricle / V_brain_jac × 100
  brain_gm_ml        absolute whole-brain gray-matter volume (mL)
  brain_tissue_frac  whole-brain tissue as a fraction of the template brain mask (%) (for auditing)
  wm_mode            white-matter peak mode (audit column for threshold calibration)

Run:
    python baselines/extract_structural_panel_s9.py
"""
import os
import io
import sys
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config, resolve_path  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import ants          # noqa: E402
import nibabel as nib  # noqa: E402

# ---------------------------------------------------------------------------
# methodological constants
# ---------------------------------------------------------------------------
TAUS = (600, 700, 800)   # the three sensitivity-analysis settings
TAU_MAIN = 700           # main definition (grounds in the module docstring)

# image label values (= XML index + 1)
LBL_HIPPOCAMPUS = [9, 19]
LBL_AMYGDALA = [10, 20]
LBL_LATVENTRICLE = [3, 14]
LBL_PARAHP = [34, 35]    # cortical atlas: anterior 34 / posterior 35


def build_masks(cfg):
    """Build the ROI masks and the whole-brain mask (invariants of template space)."""
    sub = ants.image_read(resolve_path(cfg, "atlas.harvard_oxford_sub")).numpy()
    cort = ants.image_read(resolve_path(cfg, "atlas.harvard_oxford_cort")).numpy()
    tmpl_img = ants.image_read(resolve_path(cfg, "atlas.mni_template"))
    td = tmpl_img.numpy()

    rois = {
        "hippocampus": np.isin(sub, LBL_HIPPOCAMPUS),
        "amygdala": np.isin(sub, LBL_AMYGDALA),
        "parahippocampal": np.isin(cort, LBL_PARAHP),
        "latventricle": np.isin(sub, LBL_LATVENTRICLE),
    }
    # the MTL is the union of the three (not a single anatomical structure but the operational
    # definition of the "region of early AD pathological accumulation")
    rois["mtl"] = rois["hippocampus"] | rois["amygdala"] | rois["parahippocampal"]

    # whole-brain mask: template intensity > the 5% quantile within the brain (a fixed mask,
    # constant across subjects)
    brain = td > np.percentile(td[td > 0], 5)
    return rois, brain, tmpl_img


def build_jobs(cfg):
    """Enumerate every subject of the three cohorts (with image paths and the matching warpfield paths)."""
    jobs = []

    # ---- ADNI: driven by the manifest Excel ----
    df = pd.read_excel(resolve_path(cfg, "data.adni_excel"))
    adni_root = cfg["data"]["data_root"]
    for _, r in df.iterrows():
        fp = str(r["file_path"])
        if not os.path.isabs(fp):
            fp = os.path.join(adni_root, fp)
        lab = float(r["label"])
        cls = "0" if lab == 0 else ("0.5" if lab == 0.5 else "1")
        stem = os.path.basename(fp).replace("_ws.nii.gz", "")
        jobs.append(("ADNI", stem, lab, fp,
                     os.path.join(adni_root, cls, "warpfields", stem + "_warp.nii.gz")))

    # ---- MIRIAD: direct directory scan, label subdirectories {0: HC, 1: AD} ----
    mir_root = cfg["data"]["miriad_root"]
    for cls, lab in [("0", 0), ("1", 1)]:
        d = os.path.join(mir_root, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith("_ws.nii.gz"):
                continue
            stem = f.replace("_ws.nii.gz", "")
            jobs.append(("MIRIAD", stem, float(lab), os.path.join(d, f),
                         os.path.join(mir_root, cls, "warpfields", stem + "_warp.nii.gz")))

    # ---- OASIS-2: direct directory scan, label subdirectories {0: CN, 0.5: MCI, 1: AD} ----
    oa_root = cfg["data"]["oasis2_root"]
    for cls, lab in [("0", 0.0), ("0.5", 0.5), ("1", 1.0)]:
        d = os.path.join(oa_root, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith("_ws.nii.gz"):
                continue
            stem = f.replace("_ws.nii.gz", "")
            jobs.append(("OASIS2", stem, lab, os.path.join(d, f),
                         os.path.join(oa_root, cls, "warpfields", stem + "_warp.nii.gz")))
    return jobs


def main():
    ap = argparse.ArgumentParser(description="S9 structural-measurement panel extraction")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing non-empty structural_panel.csv")
    ap.add_argument("--min-success-frac", type=float, default=0.5,
                    help="abort (non-zero exit, panel not written) if fewer than this fraction of "
                         "subjects could be extracted; default 0.5")
    args = ap.parse_args()

    cfg = load_config()
    out = cfg["output_dir"]

    rois, brain, tmpl_img = build_masks(cfg)
    brain_n = int(brain.sum())
    print("ROI template-space voxel counts:", {k: int(v.sum()) for k, v in rois.items()}, "brain", brain_n)
    print(f"MTL occupies {rois['mtl'].sum() / brain_n * 100:.2f}% of the whole brain  "
          f"hippocampus {rois['hippocampus'].sum() / brain_n * 100:.2f}%\n")

    jobs = build_jobs(cfg)
    print(f"total subjects {len(jobs)}: " + str(pd.Series([j[0] for j in jobs]).value_counts().to_dict()))

    rows, t0, fails = [], time.time(), []
    for i, (cohort, subject, label, img_p, warp_p) in enumerate(jobs):
        try:
            if not os.path.exists(img_p):
                raise FileNotFoundError("img " + img_p)
            if not os.path.exists(warp_p):
                raise FileNotFoundError("warp " + warp_p)

            I = np.asanyarray(nib.load(img_p).dataobj).astype(np.float32)
            J = ants.create_jacobian_determinant_image(tmpl_img, warp_p, do_log=False).numpy()

            rec = dict(cohort=cohort, subject=subject, label=label)

            # ---- audit column: white-matter peak mode (used to check the threshold calibration) ----
            v = I[brain]
            hist, edges = np.histogram(v, bins=256, range=(float(v.min()), float(v.max())))
            ctr = (edges[:-1] + edges[1:]) / 2
            hi = ctr > np.percentile(v, 60)          # WM sits in the high-intensity region
            rec["wm_mode"] = float(ctr[hi][np.argmax(hist[hi])])

            # ---- (A) regional gray-matter volume share (for each τ) ----
            for tau in TAUS:
                nb = float((I[brain] > tau).sum())   # normalisation denominator: whole-brain tissue voxel count
                rec[f"tissue_brain_{tau}"] = nb
                for k, m in rois.items():
                    n_roi = float((I[m] > tau).sum())
                    rec[f"gm_{k}_{tau}"] = n_roi
                    rec[f"gmr_{k}_{tau}"] = n_roi / nb * 100.0

            # ---- (B) Jacobian-modulated volume (lateral ventricle + whole brain) ----
            rec["V_latventricle"] = float(J[rois["latventricle"]].sum())
            rec["V_brain_jac"] = float(J[brain].sum())
            rec["vmr_latventricle"] = rec["V_latventricle"] / rec["V_brain_jac"] * 100.0

            # ---- absolute whole-brain gray-matter volume (mL) and audit fractions ----
            rec["brain_gm_ml"] = rec[f"tissue_brain_{TAU_MAIN}"] / 1000.0
            rec["brain_tissue_frac"] = rec[f"tissue_brain_{TAU_MAIN}"] / brain_n * 100.0

            rows.append(rec)
        except Exception as e:                                   # noqa: BLE001
            fails.append((cohort, subject, f"{type(e).__name__}: {e}"))
            print(f"  FAIL {cohort}/{subject}: {type(e).__name__} {e}")

        if (i + 1) % 50 == 0:
            el = time.time() - t0
            eta = (el / (i + 1) * (len(jobs) - i - 1)) / 60
            print(f"  {i + 1}/{len(jobs)}  {el / 60:.1f} min  ETA {eta:.1f} min")

    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "structural_panel.csv")
    n_total, n_ok = len(jobs), len(rows)
    frac = (n_ok / n_total) if n_total else 0.0

    # The failure list is diagnostic and is written even when the run is aborted below.
    if fails:
        fp = os.path.join(out, "structural_panel_failures.csv")
        pd.DataFrame(fails, columns=["cohort", "subject", "err"]).to_csv(
            fp, index=False, encoding="utf-8-sig")
        print(f"failure list -> {fp}")

    # A run that extracted little or nothing is a failure, not a result. Without this check the
    # script exited 0 after writing an all-but-empty panel over the authoritative one shipped with
    # the package — which is read by baselines/verify_structural_panel.py and the baseline scripts.
    if frac < args.min_success_frac:
        raise SystemExit(
            f"[ERROR] extracted {n_ok}/{n_total} subjects ({frac:.1%}), below the required "
            f"{args.min_success_frac:.0%}; {p} was NOT written.\n"
            f"        The usual cause is missing inputs: this script reads the registered image\n"
            f"        <data_root>/<label>/<subject>.nii.gz together with its deformation field\n"
            f"        <data_root>/<label>/warpfields/<subject>_warp.nii.gz, both produced by the\n"
            f"        preprocessing pipeline (preprocess/step3). The failure list above records\n"
            f"        the exact paths it looked for.")

    # Never silently replace the shipped panel.
    if os.path.exists(p) and os.path.getsize(p) > 0 and not args.force:
        raise SystemExit(
            f"[ERROR] {p} already exists ({os.path.getsize(p)} bytes) and would be overwritten.\n"
            f"        It is the authoritative structural panel shipped with this package. Re-run\n"
            f"        with --force if you really intend to replace it.")

    pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\nDone, {n_ok} subjects ({len(fails)} failures) -> {p}")


if __name__ == "__main__":
    main()
