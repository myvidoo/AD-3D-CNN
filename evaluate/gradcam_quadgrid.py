# -*- coding: utf-8 -*-
"""
Paper Figure 5: single-sample QuadGrid visualisation of the confusion-matrix structure.

Corresponds to the paper §2.5 / Figure 5 (1 representative sample per confusion-matrix cell, with the confidence closest to the median,
5-fold ensemble Grad-CAM, plasma colour scale, global 95th-percentile normalisation, white 0.6 contour).

Requires: 5-fold weights + ADNI test images (when using --recompute); the rendering mode can reuse stored CAMs.

Usage:
    python evaluate/gradcam_quadgrid.py               # render (reuse the stored CAMs)
    python evaluate/gradcam_quadgrid.py --recompute   # recompute the CAMs (requires GPU + imaging data)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gradcam_roi as agg
from config import resolve_path, first_existing

OUT_DIR = agg.OUTPUT_DIR / "single_sample_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Archived CAM (shipped with the package): when launched from outside the package, OUT_DIR does not
# contain it, so fall back to the in-package results/ directory (see results/README.md section 5).
OUT_ARCHIVE = Path(resolve_path(agg._CFG, "output_dir")) / "gradcam_aggregate_v3" / "single_sample_grid"
PRED_CSV = resolve_path(agg._CFG, "data.pred_csv")

CLASS_NAMES_LIST = ['CN', 'MCI', 'AD']
CELL_TITLES_COLOR = {'diag': '#2e7d32', 'off': '#c62828'}
W_BANNER_COLOR = '#3b0f4f'

# ---------- Representative-sample rule (paper §2.5 / supplementary S6) ----------
#
# The paper's rule for the Figure 5 3x3 grid is: within each (true, predicted) cell, take the
# single sample whose predicted-class confidence is closest to that cell's median confidence. It
# has no free parameter — that is the property the paper relies on ("no arbitrary parameter, so the
# selection cannot be biased") — and it is what this script applies by default.
#
# Two caveats, both stated in README §7 item 10:
#   * Exact ties. Several samples can sit at the same minimal distance, which happens in saturated
#     cells where many confidences are equal to within float64 resolution. The rule alone then does
#     not single one out; the tie is broken by taking the first such sample in predictions-CSV
#     order. The script prints which cells were tied.
#   * Recomputation. Re-running the ensemble can shift confidences by ~1e-7 (observed drift between
#     two runs of the same checkpoint was 1.5e-7), which is enough to change which sample is
#     closest to the median. A re-rendered figure may therefore show a different — equally
#     representative — subject than the published one. Figure 5 is an illustrative single-sample
#     view; the population-level statistics reported in the paper are unaffected, because they are
#     computed over all 144 test samples (evaluate/gradcam_roi.py).
#
# PUBLISHED_SELECTION records the samples actually shown in the shipped figure, so that the exact
# published rendering can be reproduced. It is used only with --match-published (or
# `gradcam.use_published_selection: true`); it is NOT part of the paper's rule.
PUBLISHED_SELECTION = {
    ('CN', 'CN'):  {'id': 'ADNI-0229_S_P1EFCEA', 'conf': 0.9999999},
    ('CN', 'MCI'): {'id': 'ADNI-1285_S_P1A0040', 'conf': 0.9952969},
    ('CN', 'AD'):  {'id': 'ADNI-0701_S_PAE1EB3', 'conf': 0.97735274},
    ('MCI', 'CN'): {'id': 'ADNI-0141_S_P3D4447', 'conf': 0.9410653},
    ('MCI', 'MCI'): {'id': 'ADNI-1131_S_P779760', 'conf': 0.9999881},
    ('MCI', 'AD'): {'id': 'ADNI-1064_S_PB35FC6', 'conf': 0.90778273},
    ('AD', 'MCI'): {'id': 'ADNI-0423_S_P7FD945', 'conf': 0.91607285},
    ('AD', 'AD'):  {'id': 'ADNI-0291_S_PF9E504', 'conf': 0.9999989},
}

# Only samples this close to the minimum distance count as tied. Deliberately tight: a loose
# tolerance (1e-6 in an earlier revision) lumps together samples at visibly different distances and
# makes the rule look ambiguous where it is not.
TIE_EPS = 1e-12
# Tolerance for matching a published confidence anchor in --match-published mode: wide enough to
# survive the ~1e-7 drift of a re-run, far tighter than the spacing between distinct samples.
CONF_TOL = 1e-5


def published_selection():
    """Published-sample anchors for --match-published, overridable from config.yaml.

        gradcam:
          published_selection:
            "CN->CN": {"id": "<file_id substring>"}    # a bare string means the same as {"id": ...}
            "AD->AD": {"conf": 0.9999989}
    """
    sel = {cell: dict(spec) for cell, spec in PUBLISHED_SELECTION.items()}
    override = (agg._CFG.get('gradcam') or {}).get('published_selection') or {}
    for key, val in override.items():
        tn, _, pn = str(key).partition('->')
        cell = (tn.strip(), pn.strip())
        if cell not in sel:
            raise ValueError(
                f"gradcam.published_selection key {key!r} is not a valid grid cell; "
                f"expected one of {sorted('->'.join(c) for c in sel)}")
        sel[cell].update(val if isinstance(val, dict) else {'id': val})
    return sel


def select_median_confidence_samples(pred_csv, match_published=False):
    """Representative sample per (true, predicted) cell, by the paper's rule: the sample whose
    predicted-class confidence is closest to the cell's median confidence.

    With `match_published=True` the samples shown in the shipped figure are pinned instead, which
    reproduces that exact rendering but reintroduces a recorded, non-rule-based choice.

    Returns (selected, cell_counts, tie_cells, unresolved):
      tie_cells  — cells where more than one sample sat at the minimal distance, so the rule alone
                   did not single one out (the first in predictions-CSV order was taken)
      unresolved — cells whose published sample was not found (only in --match-published mode)
    """
    df = pd.read_csv(pred_csv)
    df['confidence'] = df.apply(
        lambda r: [r['prob_cn'], r['prob_mci'], r['prob_ad']][int(r['pred_label'])], axis=1)
    df['file_id'] = df['file_path'].apply(lambda p: Path(p).name.replace('.nii.gz', ''))

    published = published_selection() if match_published else {}
    selected, cell_counts, tie_cells, unresolved = {}, {}, [], []
    for t in range(3):
        for p in range(3):
            tn, pn = CLASS_NAMES_LIST[t], CLASS_NAMES_LIST[p]
            cell = df[(df.true_label == t) & (df.pred_label == p)]
            cell_counts[(tn, pn)] = len(cell)
            if len(cell) == 0:
                selected[(tn, pn)] = None
                continue

            med = cell['confidence'].median()
            dist = (cell['confidence'] - med).abs()
            tied = cell[dist <= dist.min() + TIE_EPS]
            row, matched = None, False

            if match_published:
                # Search the whole cell, not just the tied set: the pinned sample is by definition
                # the one the authors showed, which need not coincide with the rule's own answer.
                spec = published.get((tn, pn)) or {}
                if spec.get('id'):
                    hit = cell[cell['file_id'].str.contains(str(spec['id']), regex=False)]
                    if len(hit):
                        row, matched = hit.iloc[0], True
                if row is None and spec.get('conf') is not None:
                    hit = cell[(cell['confidence'] - float(spec['conf'])).abs() < CONF_TOL]
                    if len(hit) == 1:
                        row, matched = hit.iloc[0], True
                if row is None:
                    unresolved.append((tn, pn))

            if row is None:
                if len(tied) > 1:
                    tie_cells.append((tn, pn))
                # Deterministic tie-break: the first tied sample in predictions-CSV order. A stable
                # sort keeps that order for equal values, so the pick does not depend on the file
                # names themselves.
                row = tied.sort_values('confidence', kind='stable').iloc[0]

            selected[(tn, pn)] = {
                'file_id': row['file_id'], 'file_path': row['file_path'],
                'confidence': float(row['confidence']), 'n': len(cell),
                'published_sample': matched,
            }
    return selected, cell_counts, tie_cells, unresolved


def compute_and_save_cams(selected, config, device):
    models = agg.load_ensemble_models(config, device)
    for (tn, pn), s in selected.items():
        if s is None:
            continue
        out = OUT_DIR / f"cam_{tn}2{pn}_median.nii.gz"
        path = agg._resolve_data_path(s['file_path'])
        input_tensor = agg.prepare_model_input(path, config)
        pred_class, _ = agg.ensemble_inference(models, input_tensor, device)
        cam = agg.compute_ensemble_gradcam(models, input_tensor, pred_class, device)
        ref = nib.load(path)
        nib.save(nib.Nifti1Image(cam.astype(np.float32), ref.affine), str(out))
        print(f"  {tn}->{pn}: {s['file_id'][:50]} peak={cam.max():.4f} -> {out.name}")


def load_cams(selected):
    cams = {}
    for (tn, pn), s in selected.items():
        if s is None:
            cams[(tn, pn)] = None
            continue
        f = Path(first_existing(OUT_DIR / f"cam_{tn}2{pn}_median.nii.gz",
                               OUT_ARCHIVE / f"cam_{tn}2{pn}_median.nii.gz"))
        if not f.exists():
            raise FileNotFoundError(f"missing CAM file: {f} (run --recompute first)")
        cams[(tn, pn)] = nib.load(str(f)).get_fdata(dtype=np.float32)
    return cams


def pick_slices(cam, brain3d, n_slices=4, weight_threshold=0.6, min_brain_frac=0.05):
    w_z = cam.mean(axis=(0, 1))
    brain_cnt = brain3d.sum(axis=(0, 1))
    eligible = brain_cnt >= min_brain_frac * brain_cnt.max()
    w_el = np.where(eligible, w_z, -np.inf)
    thresh = weight_threshold * w_el.max()
    active = np.where(eligible & (w_z >= thresh))[0]
    if len(active) < n_slices:
        active = np.where(eligible)[0]
    picked = []
    for seg in np.array_split(active, n_slices):
        zi = int(seg[np.argmax(w_z[seg])])
        picked.append((zi, float(w_z[zi])))
    picked.sort(key=lambda t: t[0])
    return picked


def render_quadgrid(selected, cams, save_png, save_tiff):
    pooled = np.concatenate([c.ravel() for c in cams.values() if c is not None])
    vmax = float(np.percentile(pooled, 95))
    contour_lv = 0.6 * vmax
    print(f"[INFO] global colour scale vmax (95th): {vmax:.4f}  contour: {contour_lv:.4f}")

    ORIGIN_Z, SPACING_Z = -72.0, 1.0

    PW = 0.85
    PH = PW * 218.0 / 182.0
    G_IN = 0.07
    CW = 2 * PW + G_IN
    CH = 2 * PH + G_IN
    GX, GY = 0.30, 0.40
    ML, MR = 0.50, 0.92
    MT, MB = 0.78, 0.55
    W = ML + 3 * CW + 2 * GX + MR
    H = MB + 3 * CH + 2 * GY + MT

    fig = plt.figure(figsize=(W, H), facecolor='white')

    def fx(x): return x / W
    def fy(y): return y / H

    for r, tn in enumerate(CLASS_NAMES_LIST):
        for c, pn in enumerate(CLASS_NAMES_LIST):
            key = (tn, pn)
            cam = cams.get(key)
            is_diag = (tn == pn)
            title_color = CELL_TITLES_COLOR['diag' if is_diag else 'off']

            cell_x = ML + c * (CW + GX)
            cell_ytop = H - MT - r * (CH + GY)
            cell_y = cell_ytop - CH

            fig.text(fx(cell_x + CW / 2), fy(cell_ytop + 0.055),
                     f"{tn}\u2192{pn}", ha='center', va='bottom',
                     fontsize=10.5, fontweight='bold', color=title_color)

            if cam is None:
                for i in range(2):
                    for j in range(2):
                        px = cell_x + j * (PW + G_IN)
                        py = cell_y + (1 - i) * (PH + G_IN)
                        axp = fig.add_axes([fx(px), fy(py), fx(PW), fy(PH)])
                        axp.set_facecolor('#f0f0f0')
                        axp.set_xticks([]); axp.set_yticks([])
                        for sp in axp.spines.values():
                            sp.set_visible(False)
                fig.text(fx(cell_x + CW / 2), fy(cell_y + CH / 2),
                         'No samples\nin this\ncategory',
                         ha='center', va='center', fontsize=9, color='#555555')
                continue

            s = selected[key]
            mri_path = agg._resolve_data_path(s['file_path'])
            mri = nib.load(mri_path).get_fdata(dtype=np.float32)
            nz = mri[mri > 0]
            lo, hi = np.percentile(nz, [1, 99])
            mri_n = np.clip((mri - lo) / (hi - lo + 1e-8), 0, 1)

            fg = ndimage.binary_fill_holes(mri > 0)
            lab, n_cc = ndimage.label(fg)
            if n_cc > 1:
                sizes = ndimage.sum(fg, lab, range(1, n_cc + 1))
                fg = lab == (int(np.argmax(sizes)) + 1)
            brain3d = ndimage.binary_erosion(fg, iterations=1, border_value=0)

            for k, (zi, w) in enumerate(pick_slices(cam, brain3d)):
                px = cell_x + (k % 2) * (PW + G_IN)
                py = cell_y + (1 - k // 2) * (PH + G_IN)
                ax = fig.add_axes([fx(px), fy(py), fx(PW), fy(PH)])
                ax.set_facecolor('black')
                img2d = mri_n[:, :, zi].T
                brain2d = brain3d[:, :, zi].T
                cam2d = np.where(brain2d, cam[:, :, zi].T, 0.0)
                ax.imshow(img2d, cmap='gray', vmin=0, vmax=1, origin='lower', interpolation='nearest')
                ax.imshow(np.ma.masked_where(~brain2d, cam2d), cmap='plasma',
                          vmin=0, vmax=vmax, alpha=0.60, origin='lower', interpolation='nearest')
                try:
                    ax.contour(cam2d > contour_lv, levels=[0.5], colors='white',
                               linewidths=0.4, alpha=0.8, origin='lower')
                except Exception:
                    pass
                z_phys = ORIGIN_Z + zi * SPACING_Z
                ax.text(0.04, 0.03, f"z={z_phys:.0f}", transform=ax.transAxes,
                        fontsize=6, color='white', ha='left', va='bottom')
                ax.text(0.96, 0.97, f"w={w:.4f}", transform=ax.transAxes,
                        fontsize=6, color='white', ha='right', va='top',
                        bbox=dict(facecolor=W_BANNER_COLOR, edgecolor='none', boxstyle='square,pad=0.22'))
                ax.set_xticks([]); ax.set_yticks([])

    for c, pn in enumerate(CLASS_NAMES_LIST):
        fig.text(fx(ML + c * (CW + GX) + CW / 2), fy(H - 0.52),
                 f"Predicted {pn}", ha='center', va='center',
                 fontsize=11.5, fontweight='bold', color='#222222')
    for r, tn in enumerate(CLASS_NAMES_LIST):
        cell_ytop = H - MT - r * (CH + GY)
        fig.text(fx(0.16), fy(cell_ytop - CH / 2),
                 f"True {tn}", ha='center', va='center', rotation=90,
                 fontsize=11.5, fontweight='bold', color='#222222')

    fig.text(0.5, fy(H - 0.24),
             'Ensemble Inference Visualization (Single-Sample Example)',
             ha='center', va='center', fontsize=13, fontweight='bold')

    grid_h = 3 * CH + 2 * GY
    cax_h = grid_h * 0.62
    cax_y = MB + (grid_h - cax_h) / 2
    cax = fig.add_axes([fx(ML + 3 * CW + 2 * GX + 0.13), fy(cax_y), fx(0.10), fy(cax_h)])
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Grad-CAM intensity\n(global 95th-pct norm.)', fontsize=7, labelpad=3)
    cbar.ax.tick_params(labelsize=6.5, length=2)

    fig.text(0.5, fy(0.34),
             'Each cell: one representative sample (median confidence). '
             'Green title = correct, red = incorrect. '
             '4 axial slices from \u226560% peak CAM weight range (brain-only).',
             ha='center', va='center', fontsize=6, color='#555555')
    fig.text(0.5, fy(0.16),
             'plasma color scale, global 95th-percentile normalization. '
             'White contours: \u226560% of global max. w = slice CAM weight',
             ha='center', va='center', fontsize=6, color='#555555')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_png, dpi=300, facecolor='white')
    fig.savefig(save_tiff, dpi=600, facecolor='white', format='tiff')
    plt.close(fig)
    print(f"[OK] PNG: {save_png}  ({W:.2f}x{H:.2f} in)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true',
                    help='reload the 5-fold models and recompute the CAMs (requires GPU); by default the stored CAMs are reused and only rendering is performed')
    ap.add_argument('--suffix', default='_rebuild',
                    help="output file-name suffix (default '_rebuild'; pass an empty string '' to write the final file names)")
    ap.add_argument('--match-published', action='store_true',
                    help="pin the 8 samples shown in the published figure instead of applying the "
                         "paper's rule (closest to the cell's median confidence). Use this to "
                         "reproduce the shipped rendering exactly; it reintroduces a recorded, "
                         "non-rule-based choice for the cells where the rule leaves a tie")
    args = ap.parse_args()

    print("=" * 70)
    print("Paper Figure 5: single-sample QuadGrid generation")
    print("=" * 70)

    print("\n[1/3] selecting the representative sample of each cell...")
    selected, cell_counts, tie_cells, unresolved = select_median_confidence_samples(
        PRED_CSV, match_published=args.match_published)
    print("  rule: " + ("samples shown in the published figure (--match-published)"
                        if args.match_published else
                        "paper 2.5/S6 - the sample closest to the cell's median confidence"))
    for key, s in selected.items():
        if s:
            tag = ""
            if args.match_published and not s.get('published_sample'):
                tag = "   <-- NOT the published sample"
            print(f"  {key[0]}->{key[1]} (n={s['n']}): {s['file_id'][:55]} "
                  f"conf={s['confidence']:.7f}{tag}")
        else:
            print(f"  {key[0]}->{key[1]}: n=0")
    if tie_cells and not args.match_published:
        print("  [NOTE] more than one sample at the minimal distance, took the first in CSV order: "
              + ", ".join(f"{a}->{b}" for a, b in tie_cells))

    if unresolved:
        cells = ", ".join(f"{a}->{b}" for a, b in unresolved)
        print(f"\n[WARN] --match-published could not locate the published sample for: {cells}.\n"
              f"       A rule-based sample is shown for those cells instead, so the figure will\n"
              f"       differ from the shipped one there. Supply your own anchors via\n"
              f"       `gradcam.published_selection` in config.yaml to pin them.")

    if args.recompute:
        # Fail early and clearly if the predictions CSV does not belong to the configured data_root:
        # otherwise the mismatch surfaces much later as an opaque FileNotFoundError inside MONAI.
        absent = [p for p in (agg._resolve_data_path(s['file_path'])
                              for s in selected.values() if s)
                  if not Path(p).exists()]
        if absent:
            raise SystemExit(
                "[ERROR] the selected images do not exist after joining data_root.\n"
                f"        missing: {absent[0]}\n"
                + (f"        (and {len(absent) - 1} more)\n" if len(absent) > 1 else "")
                + f"        `data.pred_csv` ({PRED_CSV}) was produced for a different file naming than\n"
                f"        `data.data_root` ({agg.DATA_ROOT}). Regenerate predictions for your own data\n"
                f"        first and point `data.pred_csv` at the result:\n"
                f"            python inference/predict_ensemble.py\n"
                f"            # then set data.pred_csv to <output_dir>/ensemble_test_predictions.csv")

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = agg.load_best_config_from_csv(agg.BEST_RUN_ID)
        print("\n[2/3] recomputing the 5-fold ensemble Grad-CAM (GPU)...")
        compute_and_save_cams(selected, config, device)
    else:
        print("\n[2/3] reusing the stored CAM NIfTI files (--recompute to recompute)...")

    cams = load_cams(selected)

    print("\n[3/3] rendering the QuadGrid figure...")
    suffix = args.suffix
    render_quadgrid(
        selected, cams,
        OUT_DIR / f"Figure_5_gradcam_quadgrid{suffix}.png",
        OUT_DIR / f"Figure_5_gradcam_quadgrid{suffix}.tiff",
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
