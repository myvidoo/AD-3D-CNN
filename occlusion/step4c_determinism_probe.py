# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 4c: cross-process determinism probe (GPU, about 1 minute).

Corresponds to the paper: supplementary S6 item (1) "tiered reproducibility verification".

The single question to answer (the last contradiction left by step4a/4b):
  · step0 ↔ step2  two independent processes → p0 differs by 0 (bitwise identical)
  · step2 ↔ step2b two independent processes → p0 differs by 9.449×10⁻⁴
  Since "repeated forward passes within a process are bitwise identical" and "data loading is
  bitwise identical" have already been established, the difference can only be at the
  **cross-process** level. This probe measures it directly.

Three parallel checks (each measured for every case):
  A repeated forward pass within the process ×3        expected 0 (re-checking the determinism settings)
  B re-read from disk and transform again      expected 0 (data-pipeline determinism)
  C forward pass via a different tensor-construction path     tests whether "tensor origin / layout" affects the numbers
  D cross-process: this process's p0 vs the previous process's p0  ★ the core measurement

The environment fingerprint (torch / cuDNN version, GPU, memory) is recorded at the same time, to
distinguish "algorithm selection" from "environment change".

⚠ Run it twice in a row and the script will automatically compare the two records. It only writes new files.

Output: <work>/results/det_probe.jsonl (appended), det_probe_last.json

Run: python occlusion/step4c_determinism_probe.py   # run twice in a row"""

# Source: 27_occlusion_experiment/step4c_determinism_probe.py (the computation logic is unchanged line by line;
# only the "environment bootstrap layer" was changed to be provided uniformly by
# occlusion_config; see occlusion/README.md for the refactoring notes).

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
    NUM_CLASSES, OUT_DIR, apply_determinism, device,
    ensure_dirs, get_data_transforms, get_model, load_best_config_from_csv,
    load_checkpoint, prepare_model_for_gradcam, setup_stdout,
)

setup_stdout()
ensure_dirs()

apply_determinism()


print("=" * 88)
print("Step 4c  cross-process determinism probe")
print("=" * 88)

from monai.data import Dataset

torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
config = load_best_config_from_csv(BEST_RUN_ID)

models = []
for k in range(5):
    m = get_model(config['model_name'], NUM_CLASSES, device,
                  config.get('dropout_rate', 0.0), config.get('attention_type', 'none'))
    sd = load_checkpoint(str(CKPT_DIR / ("fold_%d_best_geo.pth" % (k + 1))),
                         map_location=device)['model_state_dict']
    if hasattr(m, 'base_model') and not any(kk.startswith('base_model.') for kk in sd):
        m.base_model.load_state_dict(sd)
    else:
        m.load_state_dict(sd)
    models.append(prepare_model_for_gradcam(m).eval())
print("[OK] 5-fold models loaded | device = %s" % device)


def infer_probs(x):
    with torch.no_grad():
        logits = torch.stack([m(x) for m in models], dim=0).mean(dim=0)
        return torch.softmax(logits, dim=1)[0].cpu().numpy()


split = json.load(open(FIXED_DATA_SPLIT_JSON, encoding='utf-8'))
test_items = split['test_split']
_, test_transforms = get_data_transforms(augmentation_intensity=0,
                                         a_max=config.get('a_max', 1100))

# ★ pick 3 cases: the MCI case with the largest difference in step4a + two ordinary cases
WANT = [
    'ADNI-0156_S_PB43C7E_MCI_M_Accelerated_SAG_IR-SPGR_ws',          # the largest max|Δp0| in step4a
    'ADNI-1608_S_PCDBC60_AD_F_MPRAGE_SENSE2_ws',                      # one of the 3 AD→CN flips
    'ADNI-1040_S_PB8AEC4_CN_F_Accelerated_Sagittal_MPRAGE_(MSV21)_ws',
]
id_of = {Path(it['file_path']).name.replace('.nii.gz', ''): it for it in test_items}
picks = [id_of[w] for w in WANT if w in id_of]
if not picks:
    picks = test_items[:3]
print("[OK] %d probe cases" % len(picks))

fp = dict(pid=os.getpid(), t=time.strftime('%Y-%m-%d %H:%M:%S'),
          torch=torch.__version__, cudnn=torch.backends.cudnn.version(),
          numpy=np.__version__,
          gpu=torch.cuda.get_device_name(0),
          mem_total_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
          mem_alloc_after_load_gb=round(torch.cuda.memory_allocated(0) / 1e9, 3),
          use_det=torch.are_deterministic_algorithms_enabled(),
          cublas_cfg=os.environ.get('CUBLAS_WORKSPACE_CONFIG', '<not set>'))
print("environment fingerprint: %s" % fp)

results = {}
for it in picks:
    fid = Path(it['file_path']).name.replace('.nii.gz', '')
    tgt = {'image': it['file_path'], 'label': int(it['label'])}

    img_t = Dataset(data=[tgt], transform=test_transforms)[0]['image']
    x = img_t.unsqueeze(0).to(device)

    # A repeated forward pass within the same process ×3
    ps = [infer_probs(x) for _ in range(3)]
    same = float(max(np.abs(ps[0] - ps[1]).max(), np.abs(ps[0] - ps[2]).max(),
                     np.abs(ps[1] - ps[2]).max()))

    # B re-read from disk + transform again
    img_t2 = Dataset(data=[tgt], transform=test_transforms)[0]['image']
    reload_d = float(np.abs(img_t.numpy() - img_t2.numpy()).max())

    # C switch construction path (numpy → contiguous array → tensor → two unsqueezes)
    x2 = torch.from_numpy(np.ascontiguousarray(img_t[0].numpy())).unsqueeze(0).unsqueeze(0).to(device)
    p_alt = infer_probs(x2)
    alt_d = float(np.abs(ps[0] - p_alt).max())

    results[fid] = dict(true_label=int(it['label']), p0=[float(v) for v in ps[0]],
                        same_proc_maxdiff=same, reload_maxdiff=reload_d,
                        alt_construct_maxdiff=alt_d)
    print("  %-58s A(repeat) %.2e | B(re-read) %.2e | C(switch path) %.2e | p0.top1=%.6f argmax=%d"
          % (fid[:58], same, reload_d, alt_d, ps[0].max(), int(ps[0].argmax())))

rec = dict(fingerprint=fp, results=results)
LOG = OUT_DIR / 'det_probe.jsonl'
prev = []
if LOG.exists():
    prev = [json.loads(l) for l in open(LOG, encoding='utf-8') if l.strip()]
with open(LOG, 'a', encoding='utf-8') as f:
    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
print("\nappended:", LOG, "(%d historical records)" % len(prev))

print("\n" + "=" * 88)
print("D  cross-process comparison")
print("=" * 88)
if not prev:
    print("  this is run number 1, so there is no counterpart yet. **Please run this script once more**.")
else:
    p = prev[-1]
    print("  counterpart process: pid=%s  %s  cudnn=%s  mem_alloc_load=%.3fGB"
          % (p['fingerprint']['pid'], p['fingerprint']['t'],
             p['fingerprint']['cudnn'], p['fingerprint']['mem_alloc_after_load_gb']))
    print("  this      process: pid=%s  %s  cudnn=%s  mem_alloc_load=%.3fGB"
          % (fp['pid'], fp['t'], fp['cudnn'], fp['mem_alloc_after_load_gb']))
    worst = 0.0
    for fid, r in results.items():
        if fid not in p['results']:
            continue
        a = np.array(r['p0'], dtype=np.float64)
        b = np.array(p['results'][fid]['p0'], dtype=np.float64)
        d = float(np.abs(a - b).max())
        worst = max(worst, d)
        print("  %-58s cross-process max|Δp0| = %.3e  argmax %d vs %d"
              % (fid[:58], d, a.argmax(), b.argmax()))
    print("\n  ★ verdict: %s" % ("cross-process p0 is bitwise identical (=0) → the difference must come from script/invocation differences and needs further investigation"
                                 if worst == 0.0 else
                                 "cross-process p0 shows a jitter of order %.3e → confirms **runtime non-determinism**, "
                                 "of the same order as the 2.4e-04 in det2" % worst))

json.dump(rec, open(OUT_DIR / 'det_probe_last.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print("=" * 88)
