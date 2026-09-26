# -*- coding: utf-8 -*-
"""Occlusion experiment archive self-verification: read-only recomputation + paper numeric anchor assertions.

This script is the **executable acceptance evidence** for the ``occlusion/`` archive: anyone who
obtains this repository, with only numpy / scipy / matplotlib / PyYAML (no GPU, no raw images, no
model weights), can prove two things in one go—

1. **the archived code runs**: rerun the entire CPU analysis chain on the frozen inputs in a
   temporary working directory, with all exit codes 0;
2. **the produced data is correct**: the recomputed results agree field by field with the
   authoritative artifacts ``reference_results/`` frozen and shipped with the repository, and every
   numerical value cited by the paper / supplementary material can be verified in place against the
   frozen artifacts.

Usage::

    python occlusion/verify_occlusion.py                   # full verification (recommended)
    python occlusion/verify_occlusion.py --skip-recompute  # only check the anchors and the S6 numeric traceability
    python occlusion/verify_occlusion.py --work DIR        # use a fixed working directory
    python occlusion/verify_occlusion.py --clean           # delete the working directory after verification
    python occlusion/verify_occlusion.py --write-manifest  # rebuild the input MD5 manifest

Verification is divided into four stages

=====  =========================  ==========================================================
stage  name                       content
=====  =========================  ==========================================================
1      archive integrity          all required files present; frozen inputs match ``MANIFEST.md5``
2      paper numeric anchors      every value cited in paper §2.5/§3.3/§4.3 and supplementary S6 asserted in place
3      read-only recomputation    rerun the CPU chain in a temporary directory; compare recomputed ⇄ frozen artifacts field by field
4      S6 numeric traceability    ``manuscript_checks/check_s6_traceability.py``: traceability table ⇄ frozen artifacts row by row
=====  =========================  ==========================================================

Exit code: 0 = everything passed; 1 = at least one failure.

⚠ By default this script **writes no file into the repository** (except with ``--write-manifest``);
   all intermediate artifacts land in the working directory; the script **deletes** no file, and the
   working directory is given by ``--work``, defaulting to a one-off directory created under the
   system temporary directory.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# In a conda environment, MKL coexisting with another OpenMP runtime terminates the child process
# with "OMP: Error #15"; setting it here means the env = dict(os.environ) below passes it to every child process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

OCC_DIR = Path(__file__).resolve().parent
REF = OCC_DIR / "reference_results"
MC = OCC_DIR / "manuscript_checks"
MANIFEST = REF / "MANIFEST.md5"

# ==================== frozen inputs (all dependencies needed by the stage-3 recomputation) ====================
# reference_results/<name>  ->  <work>/<dst_sub>/<name>
FROZEN_INPUTS = [
    ("occlusion_raw.npz", "results"),
    ("occlusion_raw_rerun.npz", "results"),
    ("occlusion_raw_rerun_run2.npz", "results"),
    ("occlusion_meta.json", "results"),
    ("baseline_p0.json", "results"),
    ("donor_map.json", "results"),
    ("donor_map_run2.json", "results"),
    ("det_probe.jsonl", "results"),
    ("masks_v2.npz", "masks"),
    ("masks_v2_meta.json", "masks"),
]

# files that must be present in the archive (including frozen inputs, scripts and the paper text)
REQUIRED = [n for n, _ in FROZEN_INPUTS] + [
    "occlusion_stats.json",
    "perclass_stats.json",
    "patch_stats.json",
    "minimal_pkg_stats.json",
    "rerun_diag.json",
    "rerun_consistency.json",
    "probe_vs_data.json",
    "repeat_check.json",
    "rerun_verify.json",
    "det_probe_last.json",
    "figures/fig_occlusion_by_class.png",
    "figures/fig_occlusion_controls.png",
    "figures/fig_occlusion_dose_response.png",
    "figures/fig_occlusion_flip.png",
    "figures/fig_occlusion_paired.png",
    "figures/fig_occlusion_recall.png",
]

# ==================== stage 3: the CPU analysis chain ====================
# (script name, [expected output files relative to results], output kind)
# kind 'json' -> field-by-field numeric comparison; 'figure' -> pixel-level comparison; 'report' -> only required to be generated successfully
CHAIN = [
    ("step3_occlusion_stats.py", ["occlusion_stats.json"], "json"),
    ("step3c_per_class_stats.py", ["perclass_stats.json"], "json"),
    ("step3d_patch_flow.py", ["patch_stats.json"], "json"),
    ("step3e_minimal_package.py", ["minimal_pkg_stats.json"], "json"),
    ("step4_figures.py", [
        "figures/fig_occlusion_by_class.png",
        "figures/fig_occlusion_controls.png",
        "figures/fig_occlusion_dose_response.png",
        "figures/fig_occlusion_flip.png",
        "figures/fig_occlusion_paired.png",
        "figures/fig_occlusion_recall.png",
    ], "figure"),
    ("step4a_rerun_diag.py", ["rerun_diag.json"], "json"),
    ("step4b_consistency_recompute.py", ["rerun_consistency.json"], "json"),
    ("step4d_probe_vs_data.py", ["probe_vs_data.json"], "json"),
    ("step4e_repeat_check.py", ["repeat_check.json"], "json"),
    ("step5_report.py", ["../occlusion_report.md"], "report"),
]

# ==================== stage 2: paper numeric anchors ====================
# (authoritative artifact file, 'JSON dotted path', expected value, absolute tolerance)
# The expected values are all taken from the numbers actually written in the paper / supplementary
# material (or from the raw stored value uniquely determined by that number).
ANCHORS = [
    # ---- saturation stratification: defining the paper's main convention (S6 (4): the main convention is the non-saturated subset) ----
    ("occlusion_stats.json", "saturation/threshold", 0.999999, 0.0),
    ("occlusion_stats.json", "saturation/n_saturated", 64, 0),
    ("occlusion_stats.json", "saturation/n_non_saturated", 80, 0),
    # ---- baseline (§3.2 test-set Acc 86.81%) ----
    ("occlusion_stats.json", "baseline_acc", 0.8680555555555556, 1e-15),

    # ---- M1 main convention: MTL 100% occlusion / F1 constant fill / non-saturated n=80 (§3.3, S6 (4)) ----
    ("occlusion_stats.json", "M1_delta_p_true/MTL_d100_F1const/n", 80, 0),
    ("occlusion_stats.json", "M1_delta_p_true/MTL_d100_F1const/median_delta",
     9.083747863769531e-05, 1e-18),
    ("occlusion_stats.json", "M1_delta_p_true/MTL_d100_F1const/mean_delta",
     0.01790935106906497, 1e-14),
    ("occlusion_stats.json", "M1_delta_p_true/MTL_d100_F1const/W", 1068.0, 0.0),
    ("occlusion_stats.json", "M1_delta_p_true/MTL_d100_F1const/p_value",
     0.00810772420285838, 1e-16),
    ("occlusion_stats.json", "M1_delta_p_true/MTL_d100_F1const/effect_r",
     0.29600514796038196, 1e-14),
    ("occlusion_stats.json", "M1_delta_p_true/MTL_d100_F1const/n_pos", 53, 0),
    ("occlusion_stats.json", "M1_delta_p_true/MTL_d100_F1const/n_neg", 27, 0),

    # ---- the three filling strategies side by side (S6 (3): F1/F2/F3 are very easily mislabelled) ----
    ("occlusion_stats.json", "M1_delta_p_true/MTL_d100_F2noise/mean_delta",
     0.0071292712651465475, 1e-14),
    ("occlusion_stats.json", "M1_delta_p_true/MTL_d100_F2noise/p_value",
     0.053841762361592396, 1e-16),
    ("occlusion_stats.json", "M1_delta_p_true/MTL_d100_F3transplant/mean_delta",
     0.028545542317082943, 1e-14),
    ("occlusion_stats.json", "M1_delta_p_true/MTL_d100_F3transplant/p_value",
     0.002321838273704062, 1e-16),
    ("occlusion_stats.json", "M1_delta_p_true/MTL_d100_F3transplant/effect_r",
     0.3405131683964539, 1e-14),

    # ---- all-samples convention (S6 (4) requires it to be given alongside the main convention) ----
    ("occlusion_stats.json", "M1_delta_p_true_full/MTL_d100_F1const/n", 115, 0),
    ("occlusion_stats.json", "M1_delta_p_true_full/MTL_d100_F1const/p_value",
     0.0001299528240298809, 1e-16),

    # ---- M2 prediction flip rate (S6 (4); see S6 (1) for the reproducibility convention behind "0 flips") ----
    ("occlusion_stats.json", "M2_flip/MTL_d100_F1const/n_flip", 10, 0),
    ("occlusion_stats.json", "M2_flip/MTL_d100_F1const/n_fix", 5, 0),
    ("occlusion_stats.json", "M2_flip/MTL_d100_F1const/n_break", 5, 0),
    ("occlusion_stats.json", "M2_flip/MTL_d100_F1const/acc_delta", 0.0, 0.0),

    # ---- M3 dose-response (S6 (6) dose-response: the two-sided 0.083 is the correct statement; do not apply a blanket rule) ----
    ("occlusion_stats.json", "M3_dose_response/vox/[0]", 7941, 0),
    ("occlusion_stats.json", "M3_dose_response/vox/[4]", 31764, 0),
    ("occlusion_stats.json", "M3_dose_response/spearman_rho_median",
     0.8999999999999998, 1e-14),
    ("occlusion_stats.json", "M3_dose_response/spearman_p_median",
     0.03738607346849874, 1e-15),
    ("occlusion_stats.json", "M3_dose_response/strictly_monotonic_mean", False, 0),

    # ---- M4 sample-level coupling (§3.3 regional information dependence; S6 (9)) ----
    ("occlusion_stats.json", "M4_coupling/mtl_mass_frac/rho",
     0.011837787154242852, 1e-14),
    ("occlusion_stats.json", "M4_coupling/mtl_mass_frac/p", 0.9169966555721426, 1e-14),
    ("occlusion_stats.json", "M4_coupling/correctness_group/p",
     0.0003037878436617813, 1e-15),

    # ---- M5 control comparison (S6 (5) direct paired comparison: MTL vs hippocampus p = 0.19, limitation ③ ⇒ must not claim MTL is specific to AD) ----
    ("occlusion_stats.json", "M5_vs_controls/C1_cort_23_F1const/p",
     0.035660828255257064, 1e-14),
    ("occlusion_stats.json", "M5_vs_controls/C2_ball_F1const/p",
     0.012800332449170703, 1e-14),
    ("occlusion_stats.json", "M5_vs_controls/C3_hipp_F1const/p",
     0.19040349475275842, 1e-14),
    ("occlusion_stats.json", "verdict/code", "A-", None),
    ("occlusion_stats.json", "verdict/sig_down", True, None),
    ("occlusion_stats.json", "verdict/hipp_better", False, None),

    # ---- P2b: the provenance of 97.1% / 94.5% / 86.0% (S6 (11); patch_stats → P4 → flow_by_condition) ----
    ("patch_stats.json",
     "P4_prob_destination/flow_by_condition/MTL_d100_F1const/share_to_CN",
     0.767044562410661, 1e-15),
    ("patch_stats.json",
     "P4_prob_destination/flow_by_condition/MTL_d100_F2noise/share_to_CN",
     0.9714310595125109, 1e-15),
    ("patch_stats.json",
     "P4_prob_destination/flow_by_condition/C2_ball_F1const/share_to_CN",
     0.9452284714102398, 1e-15),
    ("patch_stats.json",
     "P4_prob_destination/flow_by_condition/C3_hipp_F1const/share_to_CN",
     0.8598631932949115, 1e-15),

    # ---- limitation ①: AD McNemar p = 0.25, not significant (S6 (11) limitation note; must not be deleted) ----
    ("perclass_stats.json", "ad_specificity/ad_break", 3, 0),
    ("perclass_stats.json", "ad_specificity/mcnemar_p", 0.25, 1e-15),
    ("perclass_stats.json", "ad_specificity/significant", False, None),

    # ---- P2a: the effect ceiling of the E3 coupling (S6 (9); the paper writes [−0.38, +0.34]) ----
    ("minimal_pkg_stats.json", "E3_coupling/MCI/n", 30, 0),
    ("minimal_pkg_stats.json", "E3_coupling/MCI/rho", -0.01935483870967742, 1e-15),
    ("minimal_pkg_stats.json", "E3_coupling/MCI/p", 0.9191412148908162, 1e-14),
    ("minimal_pkg_stats.json", "E3_coupling/MCI/min_detectable_r",
     0.49235588997361734, 1e-14),
    ("minimal_pkg_stats.json", "E3_coupling/AD/rho", -0.44820512820512814, 1e-15),
    ("minimal_pkg_stats.json", "E3b_ceiling/MCI/ci/[0]", -0.3769952909748862, 1e-14),
    ("minimal_pkg_stats.json", "E3b_ceiling/MCI/ci/[1]", 0.3433082561157216, 1e-14),

    # ---- E1 entropy (S6 (8) prediction entropy; active misreading: flipped samples lose entropy and gain confidence) ----
    ("minimal_pkg_stats.json", "E1_entropy/_MAIN_nonsat/mean",
     0.006921561276288771, 1e-15),
    ("minimal_pkg_stats.json", "E1_entropy/_MAIN_nonsat/n", 80, 0),
    ("minimal_pkg_stats.json", "E1_entropy/_MAIN_nonsat/n_down", 28, 0),
    ("minimal_pkg_stats.json", "E1_entropy/_MAIN_nonsat/n_up", 52, 0),

    # ---- E2 the donor identity of the F3 cross-subject transplant can be fully reconstructed (S6 (7); the reproducibility premise of the strongest evidence) ----
    ("minimal_pkg_stats.json", "E2_f3/donor_reconstruction/n_recovered", 144, 0),

    # ---- authoritative mask-volume convention (§2.5, S6 (2); MTL 1.83% / hippocampus 0.649%) ----
    ("masks_v2_meta.json", "brain_vox", 1735619, 0),
    ("masks_v2_meta.json", "mtl_vox", 31764, 0),
    ("masks_v2_meta.json", "hipp_vox", 11263, 0),
    ("masks_v2_meta.json", "mtl_pct_of_brain", 1.830125159957341, 1e-12),
    ("masks_v2_meta.json", "hipp_pct_of_brain", 0.6489327438798492, 1e-12),
    ("masks_v2_meta.json", "c1_vox", 32849, 0),
    ("masks_v2_meta.json", "ball_vox_total", 31910, 0),
]


# ==================== general utilities ====================

def setup_stdout():
    """Force UTF-8 + line buffering, to avoid cp936 encoding failures on the Windows console."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def getp(obj, dotted):
    """Fetch a nested value by 'a/b/c'; within a segment '\\[i\\]' denotes a list index."""
    cur = obj
    for seg in str(dotted).split("/"):
        seg = seg.strip()
        if not seg:
            continue
        if seg.startswith("[") and seg.endswith("]"):
            cur = cur[int(seg[1:-1])]
        elif isinstance(cur, list):
            cur = cur[int(seg)]
        else:
            cur = cur[seg]
    return cur


def flatten(obj, prefix=""):
    """Recursively flatten into {dotted path: leaf value}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, prefix + "/" + str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, prefix + "/" + str(i)))
    else:
        out[prefix] = obj
    return out


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


def values_equal(a, b, rtol=1e-9, atol=1e-12):
    """Lenient equality test for numbers/strings/booleans/None."""
    if a is None and b is None:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if _isnan(a) and _isnan(b):
            return True
        if _isnan(a) or _isnan(b):
            return False
        return abs(a - b) <= atol + rtol * max(abs(a), abs(b))
    return a == b


# ==================== stage 1: archive integrity ====================

def stage1_integrity():
    print("=" * 88)
    print("Stage 1  archive integrity")
    print("=" * 88)
    fails, notes = [], []

    missing = [n for n in REQUIRED if not (REF / n).exists()]
    if missing:
        fails.append("missing %d authoritative artifact(s): %s" % (len(missing), ", ".join(missing)))
    else:
        print("[OK]   all authoritative artifacts present: %d items" % len(REQUIRED))

    scripts = sorted(p.name for p in OCC_DIR.glob("*.py"))
    others = sorted(p.name for p in MC.glob("*.py")) if MC.is_dir() else []
    print("[OK]   archive scripts: %d at the top level + %d in manuscript_checks" % (len(scripts), len(others)))
    if "occlusion_config.py" not in scripts or "verify_occlusion.py" not in scripts:
        fails.append("occlusion_config.py / verify_occlusion.py are missing")

    if MANIFEST.exists():
        want = {}
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, name = line.split(None, 1)
            want[name.strip()] = digest
        bad = []
        for name, digest in sorted(want.items()):
            f = REF / name
            if not f.exists():
                bad.append("%s (file missing)" % name)
            elif md5_of(f) != digest:
                bad.append("%s (MD5 mismatch)" % name)
        if bad:
            fails.append("MANIFEST.md5 verification failed for %d item(s): %s" % (len(bad), "; ".join(bad[:6])))
        else:
            print("[OK]   MANIFEST.md5 verification passed: %d inputs byte-identical" % len(want))
    else:
        notes.append("reference_results/MANIFEST.md5 not found (it can be generated with --write-manifest)")
        print("[SKIP] MANIFEST.md5 not found, skipping the input byte check")

    return fails, notes


def write_manifest():
    """Rebuild reference_results/MANIFEST.md5 (the byte-level manifest of the frozen inputs)."""
    lines = ["# occlusion archive frozen-input manifest (MD5)",
             "# generated by: python occlusion/verify_occlusion.py --write-manifest",
             "# note: these files are the inputs to and the comparison baseline of the CPU analysis chain; modifying them would break reproducibility."]
    names = sorted(set(
        [n for n, _ in FROZEN_INPUTS]
        + list(REQUIRED)
    ))
    n = 0
    for name in names:
        f = REF / name
        if not f.exists():
            continue
        lines.append("%s  %s" % (md5_of(f), name))
        n += 1
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("written %s (%d items)" % (MANIFEST, n))


# ==================== stage 2: paper numeric anchors ====================

def stage2_anchors():
    print()
    print("=" * 88)
    print("Stage 2  paper numeric anchors")
    print("=" * 88)
    fails = []
    cache, n_ok = {}, 0
    for fname, dotted, expect, tol in ANCHORS:
        if fname not in cache:
            cache[fname] = json.loads((REF / fname).read_text(encoding="utf-8"))
        try:
            got = getp(cache[fname], dotted)
        except Exception as exc:
            fails.append("%s %s failed to fetch: %r" % (fname, dotted, exc))
            continue
        ok = (got == expect) if tol is None else values_equal(got, expect, rtol=0.0, atol=tol)
        if ok:
            n_ok += 1
        else:
            fails.append("%s %s: expected %r, actual %r" % (fname, dotted, expect, got))
    print("[%s] numeric anchors %d/%d passed" % ("OK  " if not fails else "FAIL", n_ok, len(ANCHORS)))
    for f in fails:
        print("       x %s" % f)
    return fails, []


# ==================== stage 3: read-only recomputation ====================

def _resolve_out(work, rel):
    """Resolve a relative output path from CHAIN ('figures/*' is relative to <work>, the rest to <work>/results)."""
    if rel.startswith("figures/"):
        return work / rel
    return (work / "results" / rel).resolve()


def _prepare_work(work):
    for sub in ("results", "masks", "figures", "logs", "mplcache"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for name, sub in FROZEN_INPUTS:
        src = REF / name
        if not src.exists():
            return "frozen input missing: %s" % name
        shutil.copy2(src, work / sub / name)
        n += 1
    return None if n else "no input to copy"


def _subprocess_env(work):
    env = dict(os.environ)
    for k in ("OCC_MAX_SAMPLES", "OCC_SMOKE", "OCC_RERUN_TAG", "AD3DCNN_OCC_DATA_ROOT"):
        env.pop(k, None)
    env["AD3DCNN_OCC_WORK"] = str(work)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLCONFIGDIR"] = str(work / "mplcache")
    return env


def _compare_json(tag, got_p, ref_p, fails, notes):
    got = json.loads(Path(got_p).read_text(encoding="utf-8"))
    ref = json.loads(Path(ref_p).read_text(encoding="utf-8"))
    fa, fb = flatten(got), flatten(ref)
    keys = sorted(set(fa) | set(fb))
    diffs = []
    for k in keys:
        if k not in fa:
            diffs.append("%s missing from the recomputation" % k)
        elif k not in fb:
            diffs.append("%s missing from the baseline" % k)
        elif not values_equal(fa[k], fb[k]):
            diffs.append("%s: baseline %r ≠ recomputed %r" % (k, fb[k], fa[k]))
    if diffs:
        fails.append("%s field-by-field comparison mismatch: %d/%d places (examples: %s)"
                     % (tag, len(diffs), len(keys), "; ".join(diffs[:4])))
        print("       x %s  %d/%d fields mismatch" % (tag, len(diffs), len(keys)))
    else:
        print("       - %s  %d fields agree value by value (including NaN/boolean/string)" % (tag, len(keys)))
    return len(diffs) == 0


def _compare_figure(tag, got_p, ref_p, fails, tol=0.01,
                    strong=8.0 / 255.0, max_strong_frac=0.001, hard=0.125):
    """Compare a regenerated figure against the frozen one.

    A strict max|Δ| criterion is too brittle for anti-aliased rasters: two renderings of
    byte-identical data can differ by a few 1/255 steps where a dashed line crosses a marker
    cluster, and that alone would fail an otherwise perfect reproduction. This check therefore
    reports both measures and fails only on a *substantive* difference:

      * more than `max_strong_frac` of pixels differ by more than `strong` (8/255), or
      * any single channel differs by more than `hard` (32/255).

    A genuine change - a moved marker, altered text, a different colour scale - alters far more
    than 0.1% of pixels, so it is still caught. `tol` is retained for call compatibility only.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import numpy as np
    a = mpimg.imread(str(got_p))
    b = mpimg.imread(str(ref_p))
    if a.shape != b.shape:
        fails.append("%s size mismatch: %r vs %r" % (tag, a.shape, b.shape))
        print("       x %s  size %r vs %r" % (tag, a.shape, b.shape))
        return False
    diff = np.abs(a.astype("float64") - b.astype("float64"))
    d = float(diff.max())
    strong_frac = float((diff > strong).mean())
    if strong_frac > max_strong_frac or d > hard:
        fails.append("%s pixel difference: %.4f%% of pixels differ by more than %.4f, max|Δ| = %.4f"
                     % (tag, 100.0 * strong_frac, strong, d))
        print("       x %s  pixels differing by >%.4f: %.4f%% (limit %.4f%%); max|Δ| = %.4f (hard limit %.4f)"
              % (tag, strong, 100.0 * strong_frac, 100.0 * max_strong_frac, d, hard))
        return False
    print("       - %s  %s, max|Δpixel| = %.4f, pixels differing by >%.4f: %.4f%% (limits %.4f%% / max %.4f)"
          % (tag, str(a.shape[:2]), d, strong, 100.0 * strong_frac, 100.0 * max_strong_frac, hard))
    return True


def stage3_recompute(work, python_exe):
    print()
    print("=" * 88)
    print("Stage 3  read-only recomputation (frozen inputs → rerun the CPU analysis chain → compare with the frozen artifacts field by field)")
    print("=" * 88)
    print("temporary work directory: %s" % work)
    print("interpreter             : %s" % python_exe)

    fails, notes = [], []
    err = _prepare_work(work)
    if err:
        return [err], notes

    env = _subprocess_env(work)

    # freshness baseline: record the mtime of the output files; it must change after the run.
    # mtime is used rather than "delete first" to detect stale artifacts, avoiding any dependence on delete permission.
    stale = {}
    for _s, outs, _k in CHAIN:
        for rel in outs:
            p = _resolve_out(work, rel)
            stale[str(p)] = p.stat().st_mtime_ns if p.exists() else None

    n_field_ok = 0
    for script, outputs, kind in CHAIN:
        sp = OCC_DIR / script
        if not sp.exists():
            fails.append("script missing: %s" % script)
            continue
        log = work / "logs" / (script + ".log")
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("$ %s %s\n" % (python_exe, sp))
            proc = subprocess.run([python_exe, str(sp)], cwd=str(OCC_DIR), env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace")
            fh.write(proc.stdout or "")
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").strip().splitlines()[-6:])
            fails.append("%s exit code %d; log %s; last lines: %s"
                         % (script, proc.returncode, log, tail))
            print("[FAIL] %-34s exit=%d" % (script, proc.returncode))
            print("       last lines %s" % tail.replace("\n", " | ")[:240])
            continue
        print("[OK]   %-34s exit=0  log %s" % (script, log.name))

        for rel in outputs:
            got_p = _resolve_out(work, rel)
            ref_p = REF / rel if rel.startswith("figures/") else REF / Path(rel).name
            if not got_p.exists():
                fails.append("%s did not produce %s" % (script, rel))
                print("       x did not produce %s" % rel)
                continue
            if stale.get(str(got_p)) == got_p.stat().st_mtime_ns:
                fails.append("%s: %s is a leftover artifact from a previous round (mtime not updated)" % (script, rel))
                print("       x %s is stale and was not updated" % rel)
                continue
            if kind == "json":
                if not ref_p.exists():
                    fails.append("%s is missing the comparison baseline %s" % (script, ref_p.name))
                    continue
                if _compare_json(Path(rel).name, got_p, ref_p, fails, notes):
                    n_field_ok += 1
            elif kind == "figure":
                if not ref_p.exists():
                    fails.append("%s is missing the comparison baseline %s" % (script, ref_p.name))
                    continue
                _compare_figure(Path(rel).name, got_p, ref_p, fails)
            elif kind == "report":
                size = got_p.stat().st_size
                if size < 5000:
                    fails.append("%s output too small (%d B)" % (rel, size))
                    print("       x %s is only %d B" % (rel, size))
                else:
                    notes.append("the step5 report was regenerated: %s (%d B)"
                                 % (got_p.name, size))
                    print("       - %s regenerated successfully (%d B)" % (got_p.name, size))

    print()
    print("[%s] recomputation conclusion: %d JSON artifacts field-by-field identical; %d failure(s)"
          % ("OK  " if not fails else "FAIL", n_field_ok, len(fails)))
    for f in fails:
        print("       x %s" % f)
    return fails, notes


# ==================== stage 4: S6 numeric traceability table check ====================

def stage4_manuscript(python_exe):
    print()
    print("=" * 88)
    print("Stage 4  S6 numeric traceability check (manuscript_checks/check_s6_traceability.py)")
    print("=" * 88)
    fails, notes = [], []
    sp = MC / "check_s6_traceability.py"
    tp = MC / "S6_numeric_traceability.md"
    if not sp.exists():
        return ["missing %s" % sp], notes
    if not tp.exists():
        return ["missing %s (it can be generated with check_s6_traceability.py --write)" % tp], notes
    proc = subprocess.run([python_exe, str(sp)], cwd=str(OCC_DIR),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace")
    out = proc.stdout or ""
    for line in out.splitlines():
        if ("[OK]" in line or "PASS" in line or "FAIL" in line):
            print("       %s" % line.strip()[:150])
    if proc.returncode != 0:
        fails.append("check_s6_traceability.py exit code %d" % proc.returncode)
    else:
        notes.append("S6 numeric traceability check exit code 0 (see its output for the row-by-row agreement count and the number of invariants)")
    return fails, notes


# ==================== main flow ====================

def main():
    setup_stdout()
    ap = argparse.ArgumentParser(
        description="occlusion experiment archive self-verification (read-only recomputation + paper numeric anchors)")
    ap.add_argument("--work", default=None,
                    help="temporary working directory (by default a one-off directory is created under the system temporary directory)")
    ap.add_argument("--clean", action="store_true",
                    help="delete the working directory after verification (by default it is kept, for manual inspection)")
    ap.add_argument("--skip-recompute", action="store_true",
                    help="skip stage 3 (no need to rerun the analysis chain)")
    ap.add_argument("--write-manifest", action="store_true",
                    help="rebuild reference_results/MANIFEST.md5 and exit")
    ap.add_argument("--python", default=None,
                    help="interpreter used to run the sub-scripts (default: the current interpreter)")
    ap.add_argument("--report", default=None, help="write the verification report to the given path")
    args = ap.parse_args()

    if args.write_manifest:
        write_manifest()
        return 0

    python_exe = args.python or sys.executable
    print("#" * 88)
    print("# occlusion experiment archive self-verification")
    print("#" * 88)
    print("repository      : %s" % OCC_DIR.parent)
    print("authoritative   : %s" % REF)
    print("interpreter     : %s" % python_exe)
    try:
        import numpy, scipy
        print("numpy / scipy   : %s / %s" % (numpy.__version__, scipy.__version__))
    except Exception as exc:
        print("numpy / scipy   : unavailable (%r)" % exc)
    try:
        import torch
        print("torch           : %s (cuda=%s)" % (torch.__version__,
                                                 torch.cuda.is_available()))
    except Exception:
        print("torch           : not installed — stage 3 only runs the CPU chain and does not need torch")

    all_fails, all_notes = [], []

    f, n = stage1_integrity()
    all_fails += f
    all_notes += n

    f, n = stage2_anchors()
    all_fails += f
    all_notes += n

    if args.skip_recompute:
        print()
        print("[SKIP] stage 3 (--skip-recompute)")
    else:
        # working-directory policy: **no file is ever deleted**.
        # By default a one-off temporary directory is used (freshly created each round, so there are
        # naturally no stale artifacts); when --work names a fixed directory it is reused, and stale
        # artifacts are detected via the mtime freshness check.
        if args.work:
            work = Path(args.work).resolve()
            work.mkdir(parents=True, exist_ok=True)
        else:
            work = Path(tempfile.mkdtemp(prefix="occ_verify_"))
        f, n = stage3_recompute(work, python_exe)
        all_fails += f
        all_notes += n
        if args.clean:
            shutil.rmtree(work, ignore_errors=True)
            print("(attempted to delete the temporary working directory: %s)" % work)
        else:
            print("(temporary working directory kept: %s; you may delete it yourself)" % work)

    f, n = stage4_manuscript(python_exe)
    all_fails += f
    all_notes += n

    print()
    print("#" * 88)
    if all_fails:
        print("# RESULT: FAILED (%d item(s))" % len(all_fails))
        for x in all_fails:
            print("#   x %s" % x)
    else:
        print("# RESULT: ALL PASSED — the archived code runs, and the produced data agrees with the authoritative artifacts")
    for x in all_notes:
        print("#   · %s" % x)
    print("#" * 88)

    if args.report:
        lines = ["# occlusion experiment archive self-verification report", "",
                 "- repository: `%s`" % OCC_DIR.parent,
                 "- interpreter: `%s`" % python_exe,
                 "- conclusion: **%s**" % ("all passed" if not all_fails else "%d item(s) failed" % len(all_fails)),
                 "", "## failures", ""]
        lines += (["- %s" % x for x in all_fails] or ["(none)"])
        lines += ["", "## notes", ""] + (["- %s" % x for x in all_notes] or ["(none)"])
        Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("report written to: %s" % args.report)

    return 1 if all_fails else 0


if __name__ == "__main__":
    sys.exit(main())
