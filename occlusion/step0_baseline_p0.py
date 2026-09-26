# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 0: freeze the baseline p0 and verify spatial alignment with the input tensor (GPU, about 1 minute).

Corresponds to the paper: §2.5 / §3.3 / §4.3 and supplementary S6.

Purpose
----
1. Verify that this pipeline reproduces the accuracy reported in the paper on the 144-case fixed
   test set (determinism check);
2. Verify the input tensor dimensions and their alignment with MNI 1 mm space (this determines
   whether the ROI masks can be applied directly);
3. Output the per-sample baseline probability p0 (<work>/results/baseline_p0.json) for the
   subsequent occlusion comparison.

Input: ADNI 144-case fixed test-set images (config.yaml → data.data_root) + weights/
Output: <work>/results/baseline_p0.json

Run: python occlusion/step0_baseline_p0.py"""

# Source: 27_occlusion_experiment/step0_baseline.py (the computation logic is unchanged line by line; only the
# "environment bootstrap layer" was changed to be provided uniformly by occlusion_config; see
# occlusion/README.md for the refactoring notes).

import json
import sys
import time
import numpy as np
import torch

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from occlusion_config import (  # noqa: E402
    ATLAS_CORT, ATLAS_SUB, BEST_RUN_ID, CKPT_DIR,
    CLASS_NAMES, FIXED_DATA_SPLIT_JSON, GLOBAL_SEED, NUM_CLASSES,
    OUT_DIR, TEMPLATE, apply_determinism, device,
    ensure_dirs, get_data_transforms, get_model, load_best_config_from_csv,
    load_checkpoint, prepare_model_for_gradcam, setup_stdout,
)

setup_stdout()
ensure_dirs()

# ==================== ★ deterministic inference settings ====================
# Kept consistent with step2_occlude.py: cuDNN selects different convolution algorithms according
# to batch size, and a single-case versus two-case batch differs by 2.4e-04 in practice. The
# occlusion experiment requires the two forward passes "unoccluded/occluded" to be strictly
# comparable, so deterministic algorithms are enabled on both sides (this also guarantees that p0
# from this script and p0 from Step2 are bitwise identical).
apply_determinism()


print("=" * 76)
print("Step 0  freeze the baseline and verify environment alignment")
print("=" * 76)

print("[OK] training module imported successfully")
print("     device =", device)

torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ==================== configuration ====================
config = load_best_config_from_csv(BEST_RUN_ID)
print("[OK] Run %d configuration: %s" % (BEST_RUN_ID, config))

# ==================== load the 5-fold models ====================
def load_ensemble(config):
    models = []
    for k in range(5):
        p = CKPT_DIR / ("fold_%d_best_geo.pth" % (k + 1))
        m = get_model(config['model_name'], NUM_CLASSES, device,
                      config.get('dropout_rate', 0.0),
                      config.get('attention_type', 'none'))
        ck = load_checkpoint(str(p), map_location=device)
        sd = ck['model_state_dict']
        if hasattr(m, 'base_model') and not any(kk.startswith('base_model.') for kk in sd):
            m.base_model.load_state_dict(sd)
        else:
            m.load_state_dict(sd)
        m = prepare_model_for_gradcam(m)
        m = m.eval()
        models.append(m)
    return models


t0 = time.time()
models = load_ensemble(config)
print("[OK] 5-fold models loaded (%.1fs)" % (time.time() - t0))

# ==================== data loading ====================
split = json.load(open(FIXED_DATA_SPLIT_JSON, encoding='utf-8'))
test_items = split['test_split']
print("[OK] test set n =", len(test_items))

from monai.data import Dataset
_, test_transforms = get_data_transforms(augmentation_intensity=0,
                                         a_max=config.get('a_max', 1100))

# ---- single-case trial run: verify dimensions and spatial alignment ----
probe = test_items[0]
ds = Dataset(data=[{'image': probe['file_path'], 'label': int(probe['label'])}],
             transform=test_transforms)
item = ds[0]
img = item['image'].unsqueeze(0).to(device)
print()
print("--- input tensor ---")
print("  shape =", tuple(img.shape))
print("  dtype =", img.dtype, " range = [%.4f, %.4f]" % (img.min(), img.max()))

# ---- alignment check against the atlases ----
import ants
sub_a = ants.image_read(str(ATLAS_SUB))
cort_a = ants.image_read(str(ATLAS_CORT))
tmpl = ants.image_read(str(TEMPLATE))
def _fi(x): return [float(v) for v in x]
print("  input image  spacing=%s  origin=%s" % (_fi(ants.image_read(probe['file_path']).spacing), _fi(ants.image_read(probe['file_path']).origin)))
print("  HO-sub    shape=%s  spacing=%s" % (tuple(sub_a.shape), _fi(sub_a.spacing)))
print("  HO-cort   shape=%s  spacing=%s" % (tuple(cort_a.shape), _fi(cort_a.spacing)))
print("  MNI template shape=%s  spacing=%s" % (tuple(tmpl.shape), _fi(tmpl.spacing)))
print("  image shape =", tuple(ants.image_read(probe['file_path']).shape))

# ==================== baseline inference over the full test set ====================
print()
print("--- baseline inference over the full test set ---")
records = []
t0 = time.time()
with torch.no_grad():
    for i, it in enumerate(test_items):
        p = it['file_path']
        tl = int(it['label'])
        d = Dataset(data=[{'image': p, 'label': tl}], transform=test_transforms)
        x = d[0]['image'].unsqueeze(0).to(device)
        logits = torch.stack([m(x) for m in models], dim=0).mean(dim=0)  # equal-weight logit average over the 5 folds
        prob = torch.softmax(logits, dim=1)[0]
        pred = int(prob.argmax().item())
        records.append(dict(
            file_id=Path(p).name.replace('.nii.gz', ''),
            file_path=p,
            true_label=tl,
            pred_label=pred,
            correct=int(tl == pred),
            p0=[float(v) for v in prob.cpu().numpy()],
            confidence=float(prob.max().item()),
            input_shape=list(x.shape),
        ))
        if (i + 1) % 20 == 0:
            print("  completed %d/%d  (%.1fs)" % (i + 1, len(test_items), time.time() - t0))

acc = np.mean([r['correct'] for r in records])
print()
print("=" * 76)
print("baseline result: accuracy = %.4f  (%.2f%%)   n = %d" % (acc, 100 * acc, len(records)))
print("reported in the paper: 86.81%% (125/144)")
print("verdict:", "✓ reproduction matches" if abs(acc - 0.8681) < 0.002 else "✗ mismatch, needs investigation")
print("=" * 76)

# per-class recall
for k, nm in enumerate(CLASS_NAMES):
    m = [r for r in records if r['true_label'] == k]
    if m:
        rec = np.mean([r['correct'] for r in m])
        print("  %-4s n=%3d recall=%.4f" % (nm, len(m), rec))

json.dump(records, open(OUT_DIR / 'baseline_p0.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print()
print("saved:", OUT_DIR / 'baseline_p0.json')
print("⚠ this file now uses the deterministic inference definition (cudnn.deterministic=True), consistent with Step2.")
