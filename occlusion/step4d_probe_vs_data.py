# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 4d: probe values × the two data versions × the file timeline (CPU, seconds).

Corresponds to the paper: supplementary S6 item (1).

step4c has established that in the current environment p0 is bitwise identical (difference 0) in all
four situations: **within a process, across processes, re-read from disk, and via an alternative
tensor-construction path**. So the 9.449×10⁻⁴ difference between step2 and step2b can only come
from **the two runs having different environments**. This script reconciles the three parties and
gives the file timeline:

    OLD   = results/occlusion_raw.npz        (step2 artifact)
    NEW   = results/occlusion_raw_rerun.npz  (step2b artifact)
    PROBE = the last record of results/det_probe.jsonl      (step4c probe)

Output: <work>/results/probe_vs_data.json

Run: python occlusion/step4d_probe_vs_data.py"""

# Source: 27_occlusion_experiment/step4d_probe_vs_data.py (the computation logic is unchanged line by line; only
# the "environment bootstrap layer" was changed to be provided uniformly by occlusion_config; see
# occlusion/README.md for the refactoring notes).

import json
import sys
import time
import numpy as np

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from occlusion_config import (  # noqa: E402
    OUT, setup_stdout,
)

setup_stdout()

print("=" * 92)
print("Step 4d  probe values × the two data versions × the file timeline")
print("=" * 92)


def load_p0(fname):
    raw = np.load(OUT / fname, allow_pickle=True)
    fi = [str(x) for x in raw['file_ids']]
    ci = np.asarray(raw['cond_idx'])
    P0 = raw['p0'].astype(np.float64)
    d = {}
    for k, f in enumerate(fi):
        if ci[k] == 0:                       # row 0 of each sample block
            d[f] = P0[k]
    return d


OLD = load_p0('occlusion_raw.npz')
NEW = load_p0('occlusion_raw_rerun.npz')

LOG = OUT / 'det_probe.jsonl'
probe = {}
if LOG.exists():
    lines = [json.loads(l) for l in open(LOG, encoding='utf-8') if l.strip()]
    probe = lines[-1]['results']
    FP = lines[-1]['fingerprint']
    print("probe environment fingerprint: torch=%s cudnn=%s numpy=%s | CUBLAS_WORKSPACE_CONFIG=%s | %s"
          % (FP['torch'], FP['cudnn'], FP['numpy'], FP['cublas_cfg'], FP['gpu']))

print()
print("--- one: file timeline (the results/ directory) ---")
files = ['baseline_p0.json', 'occlusion_raw.npz', 'occlusion_meta.json',
         'occlusion_raw_rerun.npz', 'donor_map.json', 'rerun_verify.json',
         'rerun_diag.json', 'rerun_consistency.json', 'det_probe.jsonl']
rows = []
for f in files:
    p = OUT / f
    if p.exists():
        mt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))
        rows.append((mt, f, p.stat().st_size))
for mt, f, sz in sorted(rows):
    print("  %s  %-28s %10.1f KB" % (mt, f, sz / 1024))

for f in ('occlusion_raw.npz', 'baseline_p0.json'):
    p = OUT / f
    if p.exists():
        gap = (time.time() - p.stat().st_mtime) / 86400.0
        print("  → %s was generated %.1f days ago" % (f, gap))

print()
print("--- two: three-party reconciliation (the 3 probe cases) ---")
print("  %-52s %-16s %-16s %-16s" % ("file_id", "PROBE top1", "OLD top1", "NEW top1"))
R = {'columns': ['file_id', 'probe_vs_old', 'probe_vs_new', 'old_vs_new'], 'rows': []}
for fid, rec in probe.items():
    p = np.array(rec['p0'], dtype=np.float64)
    o = OLD.get(fid)
    n = NEW.get(fid)
    if o is None or n is None:
        print("  %-52s ⚠ data missing" % fid[:52]); continue
    dp_o = float(np.abs(p - o).max())
    dp_n = float(np.abs(p - n).max())
    do_n = float(np.abs(o - n).max())
    print("  %-52s %-16.6f %-16.6f %-16.6f" % (fid[:52], p.max(), o.max(), n.max()))
    print("  %-52s |Δ| PROBE-OLD %.3e | PROBE-NEW %.3e | OLD-NEW %.3e"
          % ("", dp_o, dp_n, do_n))
    R['rows'].append(dict(file_id=fid, probe_vs_old=dp_o, probe_vs_new=dp_n, old_vs_new=do_n))

print()
print("--- three: verdict ---")
if R['rows']:
    so = max(r['probe_vs_old'] for r in R['rows'])
    sn = max(r['probe_vs_new'] for r in R['rows'])
    print("  max difference probe vs OLD = %.3e" % so)
    print("  max difference probe vs NEW = %.3e" % sn)
    if sn == 0.0 and so > 0:
        verdict = ('✅ probe ≡ NEW (re-run today) and ≠ OLD (legacy step2 artifact)\n'
                   '     ⟹ OLD comes from an **earlier runtime environment** (different torch/cuDNN/driver versions or settings),\n'
                   '       not from a code difference; **the current environment is fully self-reproducible** (in-process / cross-process / re-read from disk / alternative construction path all give 0)')
    elif so == 0.0 and sn > 0:
        verdict = "⚠ the probe ≡ OLD and ≠ NEW ⟹ that step2b run today deviated; the step2b runtime state needs checking"
    elif so == 0.0 and sn == 0.0:
        verdict = "⚠ the probe ≡ OLD ≡ NEW ⟹ all three pairwise directions agree, which conflicts with the step4a conclusion and needs re-checking"
    else:
        verdict = "⚠ the probe agrees with neither version ⟹ a third runtime state exists; further investigation is needed"
    print("\n  " + verdict)
    R['verdict'] = verdict

print()
print("--- four: OLD vs NEW all-sample difference distribution (re-checking step4a) ---")
fs = [f for f in OLD if f in NEW]
d = np.array([np.abs(OLD[f] - NEW[f]).max() for f in fs])
top = np.argsort(-d)[:8]
print("  n=%d | cases with difference=0 %d | cases with >1e-6 %d | max %.3e"
      % (len(fs), int((d == 0).sum()), int((d > 1e-6).sum()), d.max()))
print("  the 8 cases with the largest difference:")
for i in top:
    print("    %-58s %.3e" % (fs[i][:58], d[i]))

json.dump(R, open(OUT / 'probe_vs_data.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2, default=float)
print("\nsaved: results/probe_vs_data.json")
print("=" * 92)
