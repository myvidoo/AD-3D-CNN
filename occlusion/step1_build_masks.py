# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 1: mask construction (CPU, about 1 minute).

Corresponds to the paper: §2.5 / §3.3, Table 3 (regional volume fractions).

Builds 5 kinds of mask and writes them to disk:
  · mtl          medial temporal lobe = hippocampus + amygdala + parahippocampal gyrus (anterior/posterior)
  · hippocampus  the hippocampus proper
  · ctrl_c1      volume-matched non-MTL cortical control region (cort_23)
  · ball_union   equal-volume combination of random spheres (10 spheres)
  · brain        whole brain (MNI template > 5th percentile)
Also includes the C2/C3 controls and an anatomical self-check of the MNI centroids.

⚠ Harvard-Oxford label mapping (a frequent source of error)
    The NIfTI image label **value** = the XML ``<label index>`` **+ 1**.
    Correct values: hippocampus 9/19, amygdala 10/20, parahippocampal gyrus anterior 34 / posterior 35, lateral ventricle 4/15, brain-stem 8.
    Mistakenly using 8/18 silently picks up the **brain-stem** (volume ≈ 38,610 mm³, not the hippocampus).
    The evidence chain for correctness is in baselines/verify_atlas_identity.py (21/21 anatomical hits).

⚠ Volume-fraction definition: use the **raw mask** / ``brain.sum()``; the mask itself is **not**
intersected with the brain mask
(intersecting with the brain would turn the MTL fraction from 1.8301% into 1.7275%, which does not match Table 3 of the paper).

Input: Harvard-Oxford atlas + MNI152 template (config.yaml → atlas)
Output: <work>/masks/masks_v2.npz + masks_v2_meta.json

Run: python occlusion/step1_build_masks.py"""

# Source: 27_occlusion_experiment/step1_masks_v2.py (the computation logic is unchanged line by line; only the
# "environment bootstrap layer" was changed to be provided uniformly by occlusion_config; see
# occlusion/README.md for the refactoring notes).

import json
import sys
import numpy as np
import ants

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from occlusion_config import (  # noqa: E402
    ATLAS_CORT, ATLAS_SUB, MASK_DIR, SEED,
    TEMPLATE, ensure_dirs, setup_stdout,
)

setup_stdout()
ensure_dirs()

# ---------- ★ corrected label indices (image value = XML index + 1) ----------
SUB_LABELS = {'Left Hippocampus': 9, 'Right Hippocampus': 19,
              'Left Amygdala': 10, 'Right Amygdala': 20,
              'Left Lateral Ventricle': 4, 'Right Lateral Ventricle': 15,
              'Brain-Stem': 8}
CORT_LABELS = {'Parahippocampal Gyrus, anterior division': 34,
               'Parahippocampal Gyrus, posterior division': 35,
               'Temporal Pole': 8,
               'Frontal Orbital Cortex': 22}

seed = 42
rng = np.random.default_rng(seed)

print("=" * 78)
print("Step 1 (v2) mask construction — corrected HO label indices")
print("=" * 78)

sub_a = ants.image_read(str(ATLAS_SUB))
cort_a = ants.image_read(str(ATLAS_CORT))
tmpl = ants.image_read(str(TEMPLATE))
sub, cort, tmpl_data = sub_a.numpy(), cort_a.numpy(), tmpl.numpy()
SHAPE = sub.shape
VOX_ML = 1e-3
print("shape =", SHAPE, " spacing =", tmpl.spacing)

# ---------- brain mask (the paper's original definition) ----------
brain = tmpl_data > np.percentile(tmpl_data[tmpl_data > 0], 5)
print("brain mask voxels = %d (%.1f mL)" % (brain.sum(), brain.sum() * VOX_ML))

# ---------- main masks ----------
# ★ Definition used by the paper: the volume fraction uses the **raw mask** / brain.sum(), and the
#   mask itself is **not** intersected with brain
#   (diag confirmed: raw MTL=31764 → 1.8301%, exactly matching the paper's 1.830%;
#    with & brain it would be 29983 → 1.7275%, which does not match)
hipp = np.isin(sub, [SUB_LABELS['Left Hippocampus'], SUB_LABELS['Right Hippocampus']])
amyg = np.isin(sub, [SUB_LABELS['Left Amygdala'], SUB_LABELS['Right Amygdala']])
phg = np.isin(cort, [CORT_LABELS['Parahippocampal Gyrus, anterior division'],
                     CORT_LABELS['Parahippocampal Gyrus, posterior division']])
mtl = hipp | amyg | phg            # raw, not & brain (consistent with the paper's T3)
hipp_b = hipp

print("\n--- main masks (same definition as the paper's T3, raw without intersection) ---")
for nm, m in [('hippocampus', hipp), ('amygdala', amyg),
              ('parahippocampal gyrus', phg), ('MTL', mtl)]:
    print("  %-8s voxels=%6d  volume=%7.3f mL  of whole brain=%.4f%%"
          % (nm, m.sum(), m.sum() * VOX_ML, 100 * m.sum() / brain.sum()))

print("\n  ★ paper T3 reference: MTL 1.830%% / hippocampus 0.649%%")
print("  ★ recomputed here  : MTL %.4f%% / hippocampus %.4f%%"
      % (100 * mtl.sum() / brain.sum(), 100 * hipp.sum() / brain.sum()))

# ---------- anatomical self-check of the MNI centroids (reusing the paper's assert approach) ----------
print("\n--- MNI centroid self-check ---")


def centroid_mni(mask, img):
    """Centroid → physical (MNI) coordinates. physical = origin + direction @ (index * spacing)"""
    idx = np.argwhere(mask).mean(axis=0)
    phys = np.array(img.origin) + np.array(img.direction) @ (idx * np.array(img.spacing))
    return phys


# Definition used by the paper's analysis_summary.txt: the y axis is displayed with a negative sign (direction is [0,-1,0])
for nm, m, img, ref in [('hippocampus', hipp, sub_a, (-1.0, -21.4, -14.4)),
                        ('amygdala', amyg, sub_a, (-1.8, -4.3, -18.0)),
                        ('parahippocampal gyrus', phg, cort_a, (-0.5, -16.9, -25.4))]:
    c = centroid_mni(m, img)
    c_disp = (c[0], -c[1], c[2])       # aligned with the paper's display convention
    print("  %-8s MNI centroid = (%5.1f, %5.1f, %5.1f)   published in the paper %s"
          % (nm, c_disp[0], c_disp[1], c_disp[2], ref))

hc = centroid_mni(hipp, sub_a)
hc_disp = (hc[0], -hc[1], hc[2])
assert abs(hc_disp[1] - (-21.4)) < 1.5, "abnormal hippocampus y centroid: %.1f" % hc_disp[1]
assert abs(hc_disp[2] - (-14.4)) < 1.5, "abnormal hippocampus z centroid: %.1f" % hc_disp[2]
assert int(hipp.sum()) == 11263, "hippocampus voxel count should be 11263, actual %d" % hipp.sum()
assert int(mtl.sum()) == 31764, "MTL voxel count should be 31764, actual %d" % mtl.sum()
assert abs(100 * mtl.sum() / brain.sum() - 1.830) < 0.005, "MTL fraction does not reproduce the paper"
print("  ✔ all anatomical + volume assertions passed (hippocampus 11263 / MTL 31764 / 1.830%)")

# ---------- C3 hippocampus proper ----------
c3 = hipp

# ---------- C1 volume-matched control ----------
print("\n--- control C1 (volume-matched to MTL=%d voxels) ---" % mtl.sum())
TARGET_VOX = int(mtl.sum())
EXCLUDE_CORT = {34, 35, 8, 9, 10}          # parahippocampal gyrus anterior/posterior, temporal pole, anterior superior/middle temporal
EXCLUDE_SUB = {9, 19, 10, 20, 8, 11, 21}   # hippocampus, amygdala, brain-stem, lateral ventricle

cands = []
for lab in range(1, int(cort.max()) + 1):
    if lab in EXCLUDE_CORT:
        continue
    m = cort == lab                        # raw, same definition as the main masks
    if m.sum() > 0:
        cands.append(('cort_%d' % lab, m))
for lab in range(1, int(sub.max()) + 1):
    if lab in EXCLUDE_SUB:
        continue
    m = sub == lab
    if m.sum() > 0:
        cands.append(('sub_%d' % lab, m))

cands.sort(key=lambda kv: abs(kv[1].sum() - TARGET_VOX))
print("  the 6 candidates closest in volume to MTL (MTL-adjacent regions already excluded):")
for nm, m in cands[:6]:
    print("    %-10s voxels=%6d  (MTL=%d, diff %+d)" % (nm, m.sum(), TARGET_VOX, m.sum() - TARGET_VOX))
c1_name, c1 = cands[0]
print("  → selected C1 = %s (voxels %d, deviation %.1f%%)"
      % (c1_name, c1.sum(), 100 * (c1.sum() - TARGET_VOX) / TARGET_VOX))

# ---------- C2 random spheres (changed to a combination of multiple small spheres) ----------
print("\n--- control C2 random spheres (combination of multiple small spheres, equal total volume) ---")
# Why v1 failed: an equal-volume single sphere has radius 26.6mm, and no position inside the brain
# can accommodate it in full without touching the MTL.
# v2 switches to N small spheres, each of volume = TARGET/N, with the radius reduced to ~9mm, so
# they can be placed far from the MTL in the temporal/parietal/occipital lobes. The constraints are
# relaxed to: sphere centre inside the brain + centre at least 40mm from the MTL centroid
# (the sphere is no longer required to fall entirely inside the brain; instead the actual occluded
# volume is counted after sphere & brain).
N_BALL = 10
TARGET_PER_BALL = TARGET_VOX // N_BALL
r = (3 * TARGET_PER_BALL / (4 * np.pi)) ** (1 / 3.0)
print("  target total occluded voxels = %d; %d voxels per sphere → radius %.2f mm" % (TARGET_VOX, TARGET_PER_BALL, r))

ri = int(np.ceil(r))
zz, yy, xx = np.meshgrid(np.arange(-ri, ri + 1), np.arange(-ri, ri + 1),
                         np.arange(-ri, ri + 1), indexing='ij')
sphere_off = (zz ** 2 + yy ** 2 + xx ** 2) <= r ** 2
print("  single-sphere template voxels = %d" % sphere_off.sum())

mtl_centroid = np.argwhere(mtl).mean(axis=0)
brain_idx = np.argwhere(brain & ~mtl)      # sphere-centre candidates: inside the brain and not MTL
ball_masks, ball_centers, tries = [], [], 0
while len(ball_centers) < N_BALL and tries < 500000:
    tries += 1
    c = brain_idx[rng.integers(0, len(brain_idx))]
    # sphere centre at least 40mm from the MTL centroid, ensuring it is anatomically far from the MTL
    if np.linalg.norm(c - mtl_centroid) < 40:
        continue
    sl = tuple(slice(c[k] - ri, c[k] + ri + 1) for k in range(3))
    if any(sl[k].start < 0 or sl[k].stop > SHAPE[k] for k in range(3)):
        continue
    sub_m = brain[sl]
    if sub_m.shape != sphere_off.shape:
        continue
    # the sphere-brain intersection must be >= 60% of the sphere volume (so that most of the sphere does not fall outside the skull)
    if np.sum(sub_m & sphere_off) < 0.60 * sphere_off.sum():
        continue
    if np.any(mtl[sl][sphere_off]):          # must not overlap the MTL
        continue
    # spheres must not overlap each other
    if any(np.any(m[sl][sphere_off]) for m in ball_masks):
        continue
    m = np.zeros(SHAPE, dtype=bool)
    m[sl] = sphere_off
    ball_masks.append(m)
    ball_centers.append(tuple(int(v) for v in c))

print("  generated %d/%d spheres (%d attempts)" % (len(ball_masks), N_BALL, tries))
ball_union = np.zeros(SHAPE, dtype=bool)
for m in ball_masks:
    ball_union |= m
avg_ball = float(np.mean([m.sum() for m in ball_masks])) if ball_masks else 0.0
if ball_union.sum() > 0:
    print("  mean voxels per sphere = %.1f ; total sphere-union voxels = %d (= %.2fx MTL)"
          % (avg_ball, ball_union.sum(), ball_union.sum() / max(mtl.sum(), 1)))
else:
    print("  [SEVERE WARNING] still failed to generate spheres; control C2 is unusable!")

# ---------- summary ----------
out = {
    'brain': brain, 'mtl': mtl, 'hippocampus': c3,
    'amygdala': amyg, 'parahippocampal': phg,
    'ctrl_c1': c1, 'ctrl_c1_name': np.array([c1_name]),
    'ball_union': ball_union, 'template': tmpl_data,
}
for i, m in enumerate(ball_masks):
    out['ball_%02d' % i] = m

np.savez_compressed(MASK_DIR / 'masks_v2.npz', **out)
print("\nsaved:", MASK_DIR / 'masks_v2.npz')

meta = dict(
    shape=list(SHAPE), vox_ml=VOX_ML,
    label_mapping='HO image value = XML index + 1 (v45 corrected version)',
    brain_vox=int(brain.sum()),
    mtl_vox=int(mtl.sum()), mtl_pct_of_brain=float(100 * mtl.sum() / brain.sum()),
    hipp_vox=int(c3.sum()), hipp_pct_of_brain=float(100 * c3.sum() / brain.sum()),
    paper_mtl_pct=1.830, paper_hipp_pct=0.649,
    enrichment_mtl=float(2.376 / (100 * mtl.sum() / brain.sum())),
    enrichment_hipp=float(0.856 / (100 * c3.sum() / brain.sum())),
    c1_name=str(c1_name), c1_vox=int(c1.sum()),
    n_balls=len(ball_masks), ball_radius_mm=float(r),
    ball_vox_total=int(ball_union.sum()), ball_centers=[list(c) for c in ball_centers],
    seed=seed,
)
json.dump(meta, open(MASK_DIR / 'masks_v2_meta.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print("saved:", MASK_DIR / 'masks_v2_meta.json')

print("\n" + "=" * 78)
print("mask summary")
print("=" * 78)
print("%-24s %9s %11s %10s" % ('mask', 'voxels', 'volume (mL)', 'vs MTL'))
print("-" * 78)
for nm, m in [('MTL (main mask)', mtl), ('hippocampus (C3)', c3),
              ('control C1 (%s)' % c1_name, c1), ('sphere union (C2)', ball_union)]:
    print("%-24s %9d %11.3f %9.2fx" % (nm, m.sum(), m.sum() * VOX_ML,
                                      m.sum() / max(mtl.sum(), 1)))
print("=" * 78)
