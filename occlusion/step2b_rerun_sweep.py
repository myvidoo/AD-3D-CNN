# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 2b: full-rerun verification + donor-identity logging (GPU, about 4 minutes).

Corresponds to the paper: supplementary S6 item (1) "tiered reproducibility verification".

Purpose
----
1. **Bitwise re-check** whether <work>/results/occlusion_raw.npz (1440 records) is reproducible;
2. Log the **donor identity** (index / file_id / true_label) of the F3 cross-subject transplant as a
   first-class record
   — the original step2 did not save the donor, so the donor class could only be reconstructed
   afterwards from the rng sequence;
3. Check whether differences in the numpy version affect the donor sequence.

Design principles
----
* **Line-by-line equivalent** to step2_occlusion_sweep.py: the same determinism settings, 5-fold
  ensemble, masks, dose steps, condition order, filling functions, rng call order and parameters.
  Any deviation would make the comparison meaningless.
* **Only writes new files**, and never overwrites existing artifacts:
      <work>/results/occlusion_raw_rerun.npz
      <work>/results/donor_map.json
      <work>/results/rerun_verify.json
* Per-sample checkpoint (including the rng state), so an interrupted run can resume.

Switch: OCC_RERUN_TAG=run2 -> all artifacts get a _run2 suffix (so repeated reruns do not overwrite each other).

Run: python occlusion/step2b_rerun_sweep.py"""

# Source: 27_occlusion_experiment/step2b_rerun_full.py (the computation logic is unchanged line by line; only the
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
    BEST_RUN_ID, CKPT_DIR, FIXED_DATA_SPLIT_JSON, GLOBAL_SEED,
    MASK_DIR, NUM_CLASSES, OUT_DIR, SEED,
    apply_determinism, device, ensure_dirs, get_data_transforms,
    get_model, load_best_config_from_csv, load_checkpoint, prepare_model_for_gradcam,
    setup_stdout,
)

setup_stdout()
ensure_dirs()

# ==================== ★ deterministic inference settings (exactly the same as step2) ====================
apply_determinism()


# ★ Artifact suffix: with the default '' it is exactly identical to the existing naming (backward
#   compatible); with OCC_RERUN_TAG=run2 all artifacts (including the checkpoint) get a _run2
#   suffix, allowing repeated reruns without overwriting each other.
TAG = os.environ.get('OCC_RERUN_TAG', '')
SUF = ('_' + TAG) if TAG else ''
if SUF:
    print("[TAG] artifact suffix for this run = %s" % SUF)
rng = np.random.default_rng(SEED)

print("=" * 84)
print("Step 2b  full-rerun verification + donor-identity logging")
print("=" * 84)
print("numpy = %s | torch = %s | cuda = %s"
      % (np.__version__, torch.__version__, torch.cuda.is_available()))

from monai.data import Dataset
from scipy import ndimage

torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
config = load_best_config_from_csv(BEST_RUN_ID)
print("[OK] device = %s | Run %d | %s | attn = %s"
      % (device, BEST_RUN_ID, config.get('model_name'), config.get('attention_type')))


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
    """Equal-weight logit average over the 5 folds → softmax. Exactly the same as step0 / step2."""
    with torch.no_grad():
        logits = torch.stack([m(x) for m in models], dim=0).mean(dim=0)
        return torch.softmax(logits, dim=1)[0].cpu().numpy()


# ==================== data and masks ====================
split = json.load(open(FIXED_DATA_SPLIT_JSON, encoding='utf-8'))
test_items = split['test_split']
_, test_transforms = get_data_transforms(augmentation_intensity=0,
                                         a_max=config.get('a_max', 1100))

mz = np.load(MASK_DIR / 'masks_v2.npz')
brain, mtl, hipp = mz['brain'], mz['mtl'], mz['hippocampus']
c1, ball = mz['ctrl_c1'], mz['ball_union']
c1_name = str(mz['ctrl_c1_name'][0])
print("[OK] masks: MTL=%d hippocampus=%d C1(%s)=%d spheres=%d brain=%d"
      % (mtl.sum(), hipp.sum(), c1_name, c1.sum(), ball.sum(), brain.sum()))

# ==================== dose steps (rank-selection method exactly as in step2) ====================
dist_in = ndimage.distance_transform_edt(mtl)
order = np.argsort(-dist_in[mtl])
mtl_flat_idx = np.flatnonzero(mtl.reshape(-1))
n_mtl = len(mtl_flat_idx)
dose_masks = {}
for frac, tag in [(0.25, 'd25'), (0.40, 'd40'), (0.60, 'd60'), (0.80, 'd80'), (1.00, 'd100')]:
    k = int(round(frac * n_mtl))
    keep = mtl_flat_idx[order[:k]]
    m = np.zeros(mtl.shape, dtype=bool)
    m.reshape(-1)[keep] = True
    dose_masks[tag] = m
vols = [int(dose_masks[t].sum()) for t in ('d25', 'd40', 'd60', 'd80', 'd100')]
assert all(vols[i] < vols[i + 1] for i in range(len(vols) - 1)), vols
print("[OK] dose-step volumes strictly increasing: %s" % vols)

# ==================== condition table (the order must be exactly the same as step2) ====================
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
COND_NAMES = [c['name'] for c in CONDITIONS]
print("[OK] condition table with %d entries: %s" % (len(CONDITIONS), COND_NAMES))

# ==================== preload images ====================
print("\n[preload] loading the 144 cases...")
t0 = time.time()
images, metas = [], []
for it in test_items:
    d = Dataset(data=[{'image': it['file_path'], 'label': int(it['label'])}],
                transform=test_transforms)
    images.append(d[0]['image'])
    metas.append(dict(file_id=Path(it['file_path']).name.replace('.nii.gz', ''),
                      file_path=it['file_path'], true_label=int(it['label'])))
print("[OK] %d cases loaded (%.1fs), shape %s" % (len(images), time.time() - t0, tuple(images[0].shape)))
assert len(images) == 144, "the number of samples should be 144, actual %d" % len(images)


def make_filled(vol_np, mask, fill, donor=None):
    """Exactly as in step2: filling is applied to the **already normalised tensor**, with no second normalisation."""
    out = vol_np.copy()
    if fill == 'const':
        out[mask] = float(vol_np[brain].mean())
    elif fill == 'noise':
        vals = vol_np[mask]
        mu, sd = float(vals.mean()), float(vals.std())
        out[mask] = rng.normal(mu, sd, size=int(mask.sum())).astype(np.float32)
    elif fill == 'transplant':
        assert donor is not None
        out[mask] = donor[mask]
    else:
        raise ValueError(fill)
    return out


# ==================== main loop (with checkpoint-based resuming) ====================
by_label = {}
for i, m in enumerate(metas):
    by_label.setdefault(m['true_label'], []).append(i)

CKPT = OUT_DIR / ('_rerun_ckpt%s.npz' % SUF)
records = []
start = 0
if CKPT.exists():
    _ck = np.load(CKPT, allow_pickle=True)
    start = int(_ck['done'])
    records = list(_ck['records'])
    rng.bit_generator.state = _ck['rng_state'].item()
    print("\n[resume] restored from checkpoint: %d/%d cases already done" % (start, len(images)))

total = len(images) * len(CONDITIONS)
t0 = time.time()
for si in range(start, len(images)):
    img_t, meta = images[si], metas[si]
    vol = img_t[0].numpy()
    x0 = img_t.unsqueeze(0).to(device)
    p0 = infer_probs(x0)

    for cond in CONDITIONS:
        mask = MASK_OF[cond['mask']]
        donor = None
        dj, dlab = -1, -1
        if cond['fill'] == 'transplant':
            pool = [j for lb, js in by_label.items() if lb != meta['true_label'] for j in js]
            dj = int(pool[rng.integers(0, len(pool))])
            dlab = int(metas[dj]['true_label'])
            donor = images[dj][0].numpy()
        filled = make_filled(vol, mask, cond['fill'], donor)
        xf = torch.from_numpy(filled).unsqueeze(0).unsqueeze(0).to(device)
        pf = infer_probs(xf)

        records.append(dict(
            file_id=meta['file_id'], true_label=meta['true_label'],
            cond=cond['name'], mask_vox=int(mask.sum()), fill=cond['fill'],
            p0=[float(v) for v in p0], p_after=[float(v) for v in pf],
            mask_name=cond['mask'],
            donor_index=dj,
            donor_file_id=(metas[dj]['file_id'] if dj >= 0 else ''),
            donor_label=dlab,
        ))

    np.savez_compressed(CKPT, done=si + 1, records=np.array(records, dtype=object),
                        rng_state=np.array(rng.bit_generator.state, dtype=object))
    el = time.time() - t0
    if (si + 1) % 10 == 0 or si + 1 == len(images):
        print("  sample %3d/%d  elapsed %.0fs  estimated remaining %.0fs  | this sample p0 argmax=%d true label=%d"
              % (si + 1, len(images), el, el / max(si + 1 - start, 1) * (len(images) - si - 1),
                 int(p0.argmax()), meta['true_label']))

print("\n[OK] inference finished, %d records, took %.1f minutes"
      % (len(records), (time.time() - t0) / 60))
assert len(records) == total, "the number of records should be %d, actual %d" % (total, len(records))

# ==================== write to disk (new files, no overwriting) ====================
np.savez_compressed(
    OUT_DIR / ('occlusion_raw_rerun%s.npz' % SUF),
    cond_names=np.array(COND_NAMES),
    file_ids=np.array([r['file_id'] for r in records]),
    true_labels=np.array([r['true_label'] for r in records]),
    cond_idx=np.array([COND_NAMES.index(r['cond']) for r in records]),
    mask_vox=np.array([r['mask_vox'] for r in records]),
    p0=np.array([r['p0'] for r in records], dtype=np.float32),
    p_after=np.array([r['p_after'] for r in records], dtype=np.float32),
)
print("saved:", OUT_DIR / ('occlusion_raw_rerun%s.npz' % SUF))

DONOR = {r['file_id']: dict(donor_index=r['donor_index'], donor_file_id=r['donor_file_id'],
                            donor_label=r['donor_label'])
         for r in records if r['cond'] == 'MTL_d100_F3transplant'}
json.dump(DONOR, open(OUT_DIR / ('donor_map%s.json' % SUF), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print("saved:", OUT_DIR / ('donor_map%s.json' % SUF), "(%d recipients)" % len(DONOR))

# ==================== bitwise comparison with the existing results ====================
print("\n" + "=" * 84)
print("bitwise comparison: occlusion_raw_rerun.npz  vs  the existing occlusion_raw.npz")
print("=" * 84)
old = np.load(OUT_DIR / 'occlusion_raw.npz', allow_pickle=True)
V = {}
ok_align = (list(old['cond_names']) == COND_NAMES
            and list(old['file_ids']) == [r['file_id'] for r in records]
            and list(old['true_labels']) == [r['true_label'] for r in records]
            and list(old['cond_idx']) == [COND_NAMES.index(r['cond']) for r in records])
print("record alignment (condition names/order/sample order/condition index): %s" % ("✅ exactly identical" if ok_align else "❌ mismatch"))
V['aligned'] = bool(ok_align)
assert ok_align, "record order is not aligned, the comparison is meaningless"

NEW_P0 = np.array([r['p0'] for r in records], dtype=np.float64)
NEW_PA = np.array([r['p_after'] for r in records], dtype=np.float64)
NEW_MV = np.array([r['mask_vox'] for r in records], dtype=np.float64)

for key, b in (('p0', NEW_P0), ('p_after', NEW_PA), ('mask_vox', NEW_MV)):
    a = old[key].astype(np.float64)
    diff = np.abs(a - b)
    mx = float(diff.max())
    nbad = int((diff > 0).sum())
    print("%-10s max|Δ| = %.3e   nonzero differences = %d / %d" % (key, mx, nbad, diff.size))
    V[key] = dict(max_abs_diff=mx, n_nonzero=nbad, n_total=int(diff.size))
    if key in ('p0', 'p_after'):
        V[key]['n_argmax_mismatch'] = int((a.argmax(1) != b.argmax(1)).sum())

# per-condition differences
print("\nper-condition maximum absolute p_after difference:")
per_cond = {}
for ci_, cn_ in enumerate(COND_NAMES):
    sel = np.array([COND_NAMES.index(r['cond']) for r in records]) == ci_
    d = np.abs(old['p_after'][sel].astype(np.float64) - NEW_PA[sel])
    per_cond[cn_] = dict(max_abs_diff=float(d.max()), n_nonzero=int((d > 0).sum()))
    print("  %-24s max|Δ| = %.3e   nonzero = %d/%d" % (cn_, d.max(), int((d > 0).sum()), d.size))
V['per_condition'] = per_cond

overall = max(V['p0']['max_abs_diff'], V['p_after']['max_abs_diff'])
print("\n★ verdict: %s" % ("✅ fully reproducible (p0 and p_after bitwise identical)"
                           if overall == 0.0 else
                           "⚠ a difference of order %.3e exists, needs diagnosis" % overall))

# donor class distribution (ground truth)
CLS = ['CN', 'MCI', 'AD']
recv = {r['file_id']: int(r['true_label']) for r in records}
print("\ndonor class distribution (the true records of this rerun, recipient class × donor class):")
print("  %-6s %-8s %6s" % ("recipient", "donor", "n"))
dist = {}
for f, dv in DONOR.items():
    key = '%s<-%s' % (CLS[recv[f]], CLS[dv['donor_label']])
    dist[key] = dist.get(key, 0) + 1
for k in sorted(dist):
    print("  %-6s %-8s %6d" % (k.split('<-')[0], k.split('<-')[1], dist[k]))
print("  total %d cases (a recipient accepting from itself is impossible, hence no CN<-CN / MCI<-MCI / AD<-AD)" % sum(dist.values()))
V['donor_distribution'] = dist

json.dump(V, open(OUT_DIR / ('rerun_verify%s.json' % SUF), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2, default=float)
print("\nsaved:", OUT_DIR / ('rerun_verify%s.json' % SUF))
print("=" * 84)
