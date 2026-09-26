# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 4e: bitwise comparison of two full reruns in the same environment (CPU, seconds).

Corresponds to the paper: supplementary S6 item (1) "bitwise identical within a session".

    run1 = results/occlusion_raw_rerun.npz       (OCC_RERUN_TAG=, the first one)
    run2 = results/occlusion_raw_rerun_run2.npz  (OCC_RERUN_TAG=run2)
The two share exactly the same code, environment and samples, and differ only in when they ran
→ expected to be bitwise identical (difference 0).
The donor identities written by the two runs are also checked for exact agreement.

Input: occlusion_raw_rerun{,_run2}.npz + donor_map{,_run2}.json
Output: <work>/results/repeat_check.json

Run: python occlusion/step4e_repeat_check.py"""

# Source: 27_occlusion_experiment/step4e_repeat_check.py (the computation logic is unchanged line by line; only
# the "environment bootstrap layer" was changed to be provided uniformly by occlusion_config; see
# occlusion/README.md for the refactoring notes).

import json
import sys
import numpy as np

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from occlusion_config import (  # noqa: E402
    OUT, SAT_THR, setup_stdout,
)

setup_stdout()

A = OUT / 'occlusion_raw_rerun.npz'
B = OUT / 'occlusion_raw_rerun_run2.npz'
DA = OUT / 'donor_map.json'
DB = OUT / 'donor_map_run2.json'

print("=" * 92)
print("Step 4e  bitwise comparison of two full reruns in the same environment (run1 vs run2)")
print("=" * 92)

if not B.exists():
    raise SystemExit("the run2 artifact does not exist: %s" % B)


def pack(p):
    r = np.load(p, allow_pickle=True)
    return dict(cond=[str(x) for x in r['cond_names']],
                fid=[str(x) for x in r['file_ids']],
                tl=np.asarray(r['true_labels']),
                ci=np.asarray(r['cond_idx']),
                mv=np.asarray(r['mask_vox'], dtype=np.float64),
                p0=r['p0'].astype(np.float64),
                pa=r['p_after'].astype(np.float64))


a, b = pack(A), pack(B)
if a['cond'] != b['cond'] or a['fid'] != b['fid'] or list(a['ci']) != list(b['ci']) \
        or list(a['tl']) != list(b['tl']):
    raise SystemExit("the two versions' records are not aligned, aborting")
K = len(a['cond'])
M = len(a['fid']) // K
print("records aligned ✅ | %d samples × %d conditions = %d rows" % (M, K, len(a['fid'])))

R = {}
for key in ('p0', 'pa', 'mv'):
    d = np.abs(a[key] - b[key])
    R[key] = dict(max_abs_diff=float(d.max()), n_nonzero=int((d > 0).sum()),
                  n_total=int(d.size))
    print("  %-4s  max|Δ| = %.3e   nonzero differences = %d / %d"
          % (key, d.max(), int((d > 0).sum()), d.size))

mis_p0 = int((a['p0'].argmax(1) != b['p0'].argmax(1)).sum())
mis_pa = int((a['pa'].argmax(1) != b['pa'].argmax(1)).sum())
R['argmax_mismatch'] = dict(p0=mis_p0, p_after=mis_pa, n_cells=int(a['p0'].shape[0]))
print("  argmax mismatches: p0 %d / %d | p_after %d / %d"
      % (mis_p0, a['p0'].shape[0], mis_pa, a['pa'].shape[0]))

# saturation stratification (based only on the first condition row of the baseline p0)
P0m_a = a['p0'].reshape(M, K, 3)[:, 0, :]
P0m_b = b['p0'].reshape(M, K, 3)[:, 0, :]
sa, sb = P0m_a.max(1) > SAT_THR, P0m_b.max(1) > SAT_THR
R['saturation'] = dict(n_sat_run1=int(sa.sum()), n_sat_run2=int(sb.sum()),
                       n_flip=int((sa != sb).sum()))
print("  saturated n: run1 %d / run2 %d  →  flipped %d"
      % (sa.sum(), sb.sum(), int((sa != sb).sum())))

print()
print("--- donor identity (donor_map.json vs donor_map_run2.json) ---")
dA = json.load(open(DA, encoding='utf-8'))
dB = json.load(open(DB, encoding='utf-8'))
same_keys = set(dA) == set(dB)
mism = []
if same_keys:
    for f in dA:
        x, y = dA[f], dB[f]
        if (x['donor_index'], x['donor_file_id'], x['donor_label']) != \
           (y['donor_index'], y['donor_file_id'], y['donor_label']):
            mism.append(f)
R['donor'] = dict(same_receptor_set=bool(same_keys), n_receptors=len(dA),
                  n_mismatch=len(mism), mismatch_ids=mism[:20])
print("  recipient sets identical = %s | recipients %d | per-sample mismatches = %d"
      % (same_keys, len(dA), len(mism)))
for f in mism[:10]:
    print("    ★ %-56s %s vs %s" % (f[:56], dA[f], dB[f]))

ok = (R['p0']['max_abs_diff'] == 0.0 and R['pa']['max_abs_diff'] == 0.0
      and R['mv']['max_abs_diff'] == 0.0 and len(mism) == 0
      and not R['saturation']['n_flip'])
print()
print("=" * 92)
print("★ overall verdict: %s" % ("✅ the two full reruns are **bitwise exactly identical** (1440×3 probabilities, masks, argmax, "
                                 "saturation stratification and donor identities all the same) ⟹ this pipeline is repeatable in the current environment"
                                 if ok else "⚠ a difference exists, needs diagnosis"))
print("=" * 92)

json.dump(R, open(OUT / 'repeat_check.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2, default=float)
print("saved: results/repeat_check.json")
