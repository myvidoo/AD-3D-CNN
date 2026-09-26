# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 2: main occlusion inference sweep (GPU, about 4 minutes for 144 cases).

Corresponds to the paper: §3.3 and supplementary S6.

Protocol (aligned with the "standardised occlusion protocol" given by the external reviewer)
----
Main mask = MTL (same definition as Table 3, masks_v2.npz, 31,764 voxels). 3 filling strategies run in parallel:
  F1 constant fill   replacement with the whole-brain in-brain mean (most common, most defensible)
  F2 statistically matched noise  Gaussian noise with the region's mean/SD, so that the "abnormality of the fill value" itself does not become a cue
  F3 cross-subject transplant  replacement with same-region voxels from **another subject** (different true label) (strongest evidence)
Control groups (all F1 constant fill, for comparability under the same definition):
  C1 volume-matched non-MTL cortical region (cort_23, 32,849 voxels, 1.03x)
  C2 equal-volume random spheres (10 spheres, 31,910 voxels, 1.00x)
  C3 hippocampus proper (11,263 voxels, 0.35x) -> used for the dose-response
Dose-response: MTL is shrunk by **rank selection** on the Euclidean distance to the boundary, in five
          steps of 25/40/60/80/100%
          (a threshold-based approach degenerates to d75=d100 at 1 mm voxels, hence rank selection).

⚠ Determinism: this script enforces cuDNN deterministic algorithms so that the two forward passes
   ("unoccluded / occluded") are strictly comparable
   (the difference in cuDNN algorithm selection between batch=1 and batch=2 can reach 2.4×10⁻⁴).
⚠ This script only performs inference and writes results to disk; it does no statistics. See step3* for the statistics.

Input: <work>/masks/masks_v2.npz + ADNI 144 cases + weights/
Output: <work>/results/occlusion_raw.npz (1440 records = 144 samples × 10 conditions)

Switches: OCC_MAX_SAMPLES=N runs only the first N cases (⚠ this changes the F3 donor pool); OCC_SMOKE=1 runs only 3 cases.

Run: python occlusion/step2_occlusion_sweep.py"""

# Source: 27_occlusion_experiment/step2_occlude.py (the computation logic is unchanged line by line; only the
# "environment bootstrap layer" was changed to be provided uniformly by occlusion_config; see
# occlusion/README.md for the refactoring notes).

import json
import os
import sys
import time
import numpy as np
import torch

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from occlusion_config import (  # noqa: E402
    BEST_RUN_ID, CKPT_DIR, CLASS_NAMES, FIXED_DATA_SPLIT_JSON,
    GLOBAL_SEED, MASK_DIR, MAX_SAMPLES, NUM_CLASSES,
    OUT_DIR, SEED, apply_determinism, device,
    ensure_dirs, get_data_transforms, get_model, load_best_config_from_csv,
    load_checkpoint, prepare_model_for_gradcam, setup_stdout,
)

setup_stdout()
ensure_dirs()

# ==================== ★ deterministic inference settings ====================
# Diagnostics confirmed: repeated inference on the same tensor with the same
# batch size is bit-identical (difference=0), but batch=1 and batch=2 differ by 2.4e-04 — this is
# the result of cuDNN automatically selecting a convolution algorithm according to batch size.
# In the occlusion experiment the two forward passes "unoccluded" and "occluded" must be strictly
# comparable, hence deterministic algorithms are enforced.
apply_determinism()


rng = np.random.default_rng(SEED)

print("=" * 80)
print("Step 2  occlusion inference")
print("=" * 80)

from monai.data import Dataset

torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
config = load_best_config_from_csv(BEST_RUN_ID)
print("[OK] device =", device, "| Run", BEST_RUN_ID, config.get('model_name'),
      "attn =", config.get('attention_type'))

# ==================== load the 5-fold models ====================
def load_ensemble(cfg):
    ms = []
    for k in range(5):
        p = CKPT_DIR / ("fold_%d_best_geo.pth" % (k + 1))
        m = get_model(cfg['model_name'], NUM_CLASSES, device,
                      cfg.get('dropout_rate', 0.0), cfg.get('attention_type', 'none'))
        ck = load_checkpoint(str(p), map_location=device)
        sd = ck['model_state_dict']
        if hasattr(m, 'base_model') and not any(kk.startswith('base_model.') for kk in sd):
            m.base_model.load_state_dict(sd)
        else:
            m.load_state_dict(sd)
        ms.append(prepare_model_for_gradcam(m).eval())
    return ms


t0 = time.time()
models = load_ensemble(config)
print("[OK] 5-fold models loaded (%.1fs)" % (time.time() - t0))


def infer_probs(x):
    """Equal-weight logit average over the 5 folds → softmax. Exactly the same as step0."""
    with torch.no_grad():
        logits = torch.stack([m(x) for m in models], dim=0).mean(dim=0)
        return torch.softmax(logits, dim=1)[0].cpu().numpy()


# ==================== data and masks ====================
split = json.load(open(FIXED_DATA_SPLIT_JSON, encoding='utf-8'))
test_items = split['test_split']
_, test_transforms = get_data_transforms(augmentation_intensity=0,
                                         a_max=config.get('a_max', 1100))

mz = np.load(MASK_DIR / 'masks_v2.npz')
brain = mz['brain']
mtl = mz['mtl']
hipp = mz['hippocampus']
c1 = mz['ctrl_c1']
ball = mz['ball_union']
c1_name = str(mz['ctrl_c1_name'][0])
print("[OK] masks: MTL=%d  hippocampus=%d  C1(%s)=%d  spheres=%d  brain=%d"
      % (mtl.sum(), hipp.sum(), c1_name, c1.sum(), ball.sum(), brain.sum()))

# ==================== dose-response: MTL shrinking by distance ====================
print("\n[dose steps] shrinking by Euclidean distance to the MTL boundary")
from scipy import ndimage

# compute, for each voxel inside the MTL, its distance to the boundary
dist_in = ndimage.distance_transform_edt(mtl)
dvals = dist_in[mtl]
print("  distance quantiles inside the MTL: 25%%=%.2f  50%%=%.2f  75%%=%.2f  max=%.2f mm"
      % tuple(np.percentile(dvals, [25, 50, 75, 100])))
print("  distance values (integer mm):", np.unique(np.round(dvals, 3))[:12], "...")

# ⚠️ The distance is in integer mm (EDT on 1mm isotropic voxels), so a threshold-based approach
#    degenerates to d75==d100. A **rank-selection** method that takes the top k% after sorting by
#    distance is therefore used instead, guaranteeing that every step is strictly monotonically decreasing.
order = np.argsort(-dist_in[mtl])          # distance from large to small
mtl_flat_idx = np.flatnonzero(mtl.reshape(-1))
n_mtl = len(mtl_flat_idx)

dose_masks = {}
for frac, tag in [(0.25, 'd25'), (0.40, 'd40'), (0.60, 'd60'), (0.80, 'd80'), (1.00, 'd100')]:
    k = int(round(frac * n_mtl))
    keep = mtl_flat_idx[order[:k]]
    m = np.zeros(mtl.shape, dtype=bool)
    m.reshape(-1)[keep] = True
    dose_masks[tag] = m
    print("  %-4s the %3.0f%% of voxels closest to the boundary → voxels=%6d  threshold distance≥%.1f mm  (%.4f%% of whole brain)"
          % (tag, frac * 100, m.sum(), dist_in[m].min(), 100 * m.sum() / brain.sum()))

# monotonicity assertion
vols = [dose_masks[t].sum() for t in ('d25', 'd40', 'd60', 'd80', 'd100')]
assert all(vols[i] < vols[i + 1] for i in range(len(vols) - 1)), \
    "dose-step volumes must be strictly increasing, actual %s" % vols
print("  ✔ dose-step volumes strictly increasing:", vols)

# ==================== build the condition table ====================
# each condition = (mask, filling strategy)
CONDITIONS = []
for tag in ('d100', 'd80', 'd60', 'd40', 'd25'):
    CONDITIONS.append(dict(name='MTL_%s_F1const' % tag, mask=tag, fill='const'))
CONDITIONS.append(dict(name='MTL_d100_F2noise', mask='d100', fill='noise'))
CONDITIONS.append(dict(name='MTL_d100_F3transplant', mask='d100', fill='transplant'))
CONDITIONS.append(dict(name='C1_%s_F1const' % c1_name, mask='c1', fill='const'))
CONDITIONS.append(dict(name='C2_ball_F1const', mask='ball', fill='const'))
CONDITIONS.append(dict(name='C3_hipp_F1const', mask='hipp', fill='const'))

MASK_OF = {'d100': mtl, 'd80': dose_masks['d80'], 'd60': dose_masks['d60'],
           'd40': dose_masks['d40'], 'd25': dose_masks['d25'],
           'c1': c1, 'ball': ball, 'hipp': hipp}

print("\n[condition table] %d conditions in total" % len(CONDITIONS))
for c in CONDITIONS:
    m = MASK_OF[c['mask']]
    print("  %-28s mask=%-5s voxels=%6d  fill=%s" % (c['name'], c['mask'], m.sum(), c['fill']))

# ==================== preload all images (to avoid repeated IO) ====================
print("\n[preload] loading the 144 images into memory...")
t0 = time.time()
images = []
metas = []
for it in test_items:
    d = Dataset(data=[{'image': it['file_path'], 'label': int(it['label'])}],
                transform=test_transforms)
    arr = d[0]['image']  # (1,182,218,182) float32
    images.append(arr)
    metas.append(dict(file_id=Path(it['file_path']).name.replace('.nii.gz', ''),
                      file_path=it['file_path'], true_label=int(it['label'])))
print("[OK] %d cases loaded (%.1fs), single-case shape %s" % (len(images), time.time() - t0,
                                                              tuple(images[0].shape)))

# ==================== filling functions ====================
def make_filled(vol_np, mask, fill, donor=None):
    """vol_np: (182,218,182) float32; returns a copy with the occlusion applied"""
    out = vol_np.copy()
    if fill == 'const':
        # replacement with the whole-brain in-brain mean (defensible: introduces no new information)
        val = float(vol_np[brain].mean())
        out[mask] = val
    elif fill == 'noise':
        # statistically matched Gaussian noise: the mean/SD of the original voxels in the same region
        vals = vol_np[mask]
        mu, sd = float(vals.mean()), float(vals.std())
        out[mask] = rng.normal(mu, sd, size=int(mask.sum())).astype(np.float32)
    elif fill == 'transplant':
        # cross-class transplant: same-region voxels from another subject (different true label)
        assert donor is not None
        out[mask] = donor[mask]
    else:
        raise ValueError(fill)
    return out


# ==================== main loop ====================
SMOKE = os.environ.get('OCC_SMOKE', '0') == '1'
if SMOKE:
    images = images[:3]
    metas = metas[:3]
    print("\n[SMOKE TEST] running only the first 3 cases")
# subset switch (OCC_MAX_SAMPLES=N, for a quick check of the GPU code path).
# ⚠ When N < 144, the donor pool of the F3 cross-subject transplant shrinks with the sample set,
#   so that condition is no longer bitwise comparable with the full run; the other 9 conditions
#   remain bitwise comparable.
if MAX_SAMPLES:
    images = images[:MAX_SAMPLES]
    metas = metas[:MAX_SAMPLES]
    print("\n[SUBSET] OCC_MAX_SAMPLES=%d, running only the first %d cases" % (MAX_SAMPLES, MAX_SAMPLES))

print("\n[inference] starting the occlusion sweep ...")
records = []
# pre-select transplant donors: build a donor list for each class pool (different true label)
by_label = {}
for i, m in enumerate(metas):
    by_label.setdefault(m['true_label'], []).append(i)

t0 = time.time()
n_done = 0
total = len(images) * len(CONDITIONS)

for si, (img_t, meta) in enumerate(zip(images, metas)):
    vol = img_t[0].numpy()  # (182,218,182)
    x0 = img_t.unsqueeze(0).to(device)
    p0 = infer_probs(x0)     # re-check the baseline (should agree with step0)

    for cond in CONDITIONS:
        mask = MASK_OF[cond['mask']]
        donor = None
        if cond['fill'] == 'transplant':
            # randomly pick one donor from the pool of subjects with a different true label
            pool = [j for lb, js in by_label.items() if lb != meta['true_label'] for j in js]
            dj = pool[rng.integers(0, len(pool))]
            donor = images[dj][0].numpy()
        filled = make_filled(vol, mask, cond['fill'], donor)
        xf = torch.from_numpy(filled).unsqueeze(0).unsqueeze(0).to(device)
        pf = infer_probs(xf)

        records.append(dict(
            file_id=meta['file_id'], true_label=meta['true_label'],
            cond=cond['name'], mask_vox=int(mask.sum()), fill=cond['fill'],
            p0=[float(v) for v in p0], p_after=[float(v) for v in pf],
            mask_name=cond['mask'],
        ))
        n_done += 1
        if n_done % 200 == 0:
            el = time.time() - t0
            print("  progress %d/%d (%.1f%%)  elapsed %.0fs  estimated remaining %.0fs"
                  % (n_done, total, 100 * n_done / total, el,
                     el / n_done * (total - n_done)))

print("\n[OK] inference finished, %d records in total, took %.1f minutes"
      % (len(records), (time.time() - t0) / 60))

# ==================== write to disk ====================
COND_NAMES = [c['name'] for c in CONDITIONS]
np.savez_compressed(
    OUT_DIR / 'occlusion_raw.npz',
    cond_names=np.array(COND_NAMES),
    file_ids=np.array([r['file_id'] for r in records]),
    true_labels=np.array([r['true_label'] for r in records]),
    cond_idx=np.array([COND_NAMES.index(r['cond']) for r in records]),
    mask_vox=np.array([r['mask_vox'] for r in records]),
    p0=np.array([r['p0'] for r in records], dtype=np.float32),
    p_after=np.array([r['p_after'] for r in records], dtype=np.float32),
)
print("saved:", OUT_DIR / 'occlusion_raw.npz')

# metadata
meta_out = dict(
    n_samples=len(images), n_conditions=len(CONDITIONS),
    conditions=[dict(name=c['name'], mask=c['mask'], fill=c['fill'],
                     mask_vox=int(MASK_OF[c['mask']].sum())) for c in CONDITIONS],
    dose_vox={k: int(v.sum()) for k, v in dose_masks.items()},
    mask_vox=dict(mtl=int(mtl.sum()), hipp=int(hipp.sum()),
                  c1=int(c1.sum()), ball=int(ball.sum()), brain=int(brain.sum())),
    c1_name=c1_name, seed=SEED, run_id=BEST_RUN_ID,
    fill_policy=dict(const='whole-brain in-brain mean', noise='region-matched Gaussian noise',
                     transplant='same-region voxels from a subject with a different true label'),
    note='p0 is a repeated-inference re-check and should be exactly identical to step0_baseline.json',
)
json.dump(meta_out, open(OUT_DIR / 'occlusion_meta.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print("saved:", OUT_DIR / 'occlusion_meta.json')

# ==================== consistency self-check ====================
b0 = json.load(open(OUT_DIR / 'baseline_p0.json', encoding='utf-8'))
ref = {r['file_id']: np.array(r['p0']) for r in b0}
diffs, argmax_mis = [], 0
for r in records:
    if r['cond'] != COND_NAMES[0]:
        continue
    a = np.array(r['p0']); b = ref[r['file_id']]
    diffs.append(np.abs(a - b).max())
    if a.argmax() != b.argmax():
        argmax_mis += 1
dmax = float(max(diffs))
print("\n[self-check] this run's p0 against the Step0 baseline")
print("       max|Δp|          = %.2e" % dmax)
print("       argmax mismatches = %d / %d" % (argmax_mis, len(diffs)))
# Note: this script has cudnn.deterministic enabled, so p0 should be exactly identical to Step0 (<1e-6).
#       If a difference of order 1e-4 remains, Step0 did not enable deterministic algorithms; the
#       present run then takes precedence.
tol = 1e-6
if dmax < tol:
    print("       verdict: ✓ exactly identical to Step0 (deterministic algorithms are aligned)")
else:
    print("       verdict: ⚠ Step0 differs from this run by an order of %.1e (Step0 did not enable cudnn.deterministic)" % dmax)
    print("             → **this run's p0 takes precedence** (this run uses deterministic inference, and the same batch definition before and after occlusion)")
assert argmax_mis == 0, "inconsistency at the argmax level, needs investigation"
assert dmax < 1e-3, "p0 difference too large (%.2e), needs investigation" % dmax
print("=" * 80)
