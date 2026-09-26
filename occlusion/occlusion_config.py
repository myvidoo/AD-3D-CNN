# -*- coding: utf-8 -*-
"""Unified configuration and shared utilities for the occlusion causality analysis (paper §2.5 / §3.3 / §4.3, supplementary S6).

This module is the **sole environment entry point** for every script under
``occlusion/`` and plays the same role as the repository-root ``config.yaml``:
all paths, random seeds and interpretation thresholds are resolved centrally
here, so no script body contains any machine-specific path.

Resolution priority (high → low)::

    environment variables  >  the ``occlusion`` node of config.yaml  >  repository-relative defaults

List of environment variables (all optional)::

    AD3DCNN_OCC_WORK        experimental work directory (masks / results / figures are created under it)
    AD3DCNN_OCC_DATA_ROOT   root directory of preprocessed NIfTI images (overrides data.data_root in config.yaml)
    OCC_MAX_SAMPLES         run only the first N cases (for quick validation)
                            ⚠ shrinking the sample set changes the donor pool of the F3 cross-subject
                              transplant, so that condition is no longer bitwise comparable with the
                              full run; the remaining conditions are still comparable.
    OCC_SMOKE=1             run only the first 3 cases
    OCC_RERUN_TAG           artifact suffix (so that repeated step2b reruns do not overwrite each other)

Design notes
------------
* This module **does not depend on torch / monai at import time**, so the CPU
  scripts needed for statistics and figures alone can run in an environment
  with only numpy / scipy / PyYAML installed; torch-related capabilities are
  exposed through **lazily imported functions** (``get_model`` etc.).
* GPU scripts uniformly call ``apply_determinism()`` to enable cuDNN
  deterministic algorithms, guaranteeing that the two forward passes
  ("unoccluded / occluded") are strictly comparable.
* The model and data pipeline **reuse this repository's implementation
  directly** (``model/densenet169_attention.py``, ``common/utils.py``) and do
  not depend on any directory outside the repository.
"""

import json
import os
import sys
from pathlib import Path

# ---- OpenMP runtime-conflict fallback (must be set before importing numpy / scipy) ------------
# In a conda environment, if MKL's libiomp5md.dll coexists with another OpenMP runtime library,
# the process terminates outright with "OMP: Error #15" (this repository's GPU scripts already
# set this variable in the legacy training scripts, whereas the pure-CPU occlusion chain had
# previously omitted it). Child processes inherit this variable.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# -------------------------------------------------------------------------

import numpy as np

# ==================== repository location and import path ====================

OCC_DIR = Path(__file__).resolve().parent
REPO_ROOT = OCC_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import load_config, resolve_path  # noqa: E402

_CFG = load_config()
_OCC_CFG = _CFG.get("occlusion", {}) or {}


def _env_path(name, default):
    """Environment variable (absolute path) first, then config.yaml, finally the repository-relative default."""
    v = os.environ.get(name)
    if v:
        return Path(v).expanduser().resolve()
    return Path(default)


# ==================== directory conventions ====================

# ⚠ Like the other path keys, ``occlusion.work_dir`` in config.yaml is resolved **relative to the
#   directory containing config.yaml (i.e. the repository root)**, so it must be joined with
#   ``REPO_ROOT`` here rather than with ``OCC_DIR``.
#   (Writing ``OCC_DIR / ...`` would yield the incorrectly nested path ``occlusion/occlusion/work``.)
WORK_DIR = _env_path("AD3DCNN_OCC_WORK",
                     REPO_ROOT / _OCC_CFG.get("work_dir", "occlusion/work"))
MASK_DIR = WORK_DIR / "masks"          # masks (produced by step1)
OUT_DIR = WORK_DIR / "results"         # numeric artifacts (produced by step0/2/2b/3*/4*/5)
FIG_DIR = WORK_DIR / "figures"         # figures (produced by step4_figures)
OUT = OUT_DIR                          # for compatibility with the original script naming
FIG = FIG_DIR                          # for compatibility with the original script naming

# authoritative artifacts shipped with the repository (baseline for offline verification)
REF_DIR = OCC_DIR / "reference_results"
MANUSCRIPT_DIR = OCC_DIR / "manuscript_checks"


def ensure_dirs():
    """Create all work directories."""
    for d in (WORK_DIR, MASK_DIR, OUT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_stdout():
    """Unify stdout: UTF-8 + line buffering (so logs still show progress when a long task is interrupted)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass


# ==================== data and model paths ====================

DATA_ROOT = os.environ.get("AD3DCNN_OCC_DATA_ROOT") or _CFG["data"]["data_root"]
_SPLIT_SRC = Path(resolve_path(_CFG, "data.fixed_split"))
GRID_RESULTS = resolve_path(_CFG, "data.grid_results")
WEIGHTS_DIR = resolve_path(_CFG, "weights_dir")
CKPT_DIR = Path(WEIGHTS_DIR)           # 5-fold checkpoint (fold_1..5_best_geo.pth)
CSV = resolve_path(_CFG, "occlusion.gradcam_per_sample_csv")

ATLAS_SUB = Path(_CFG["atlas"]["harvard_oxford_sub"])
ATLAS_CORT = Path(_CFG["atlas"]["harvard_oxford_cort"])
TEMPLATE = Path(_CFG["atlas"]["mni_template"])

# ==================== methodological constants ====================

BEST_RUN_ID = int(_CFG["best_run_id"])          # the best configuration reported in the paper, Run 128
SEED = 42                                       # random seed for mask construction / donor sampling
GLOBAL_SEED = 42                                # consistent with training (common/utils.py)
NUM_CLASSES = 3
CLASS_NAMES = ["CN", "MCI", "AD"]
# Saturation threshold: based solely on the baseline p0, used to define the paper's "non-saturated
# subset" under the main definition (n=80)
SAT_THR = float(_OCC_CFG.get("saturation_threshold", 0.999999))

MAX_SAMPLES = int(os.environ.get("OCC_MAX_SAMPLES", "0")) or None


# ==================== fixed test-set split (absolutised) ====================

def _materialize_abs_split():
    """Expand the fixed split JSON into a copy holding absolute ``file_path`` values, written to the work directory.

    Background (GPU smoke test): the ``data/fixed_data_split.json`` shipped with the repository
    **stores only relative file names** (relative to ``data.data_root`` in ``config.yaml``), whereas
    the scripts of this sub-experiment hand it directly to MONAI ``LoadImaged`` — MONAI resolves it
    relative to the **process's current working directory**, and therefore raises
    ``FileNotFoundError``. The original working scripts relied on the **absolute-path version** of
    the split provided by an external training module
    (``cnn_model_v3_7_d169_group_comparison.FIXED_DATA_SPLIT_JSON``).

    Here the absolute-path copy is generated in the environment bootstrap layer: **the sample
    order, ``label`` and ``file_id`` (basename) are exactly identical to the prototype**, and only
    ``file_path`` becomes an absolute path, so no script body needs any change.
    """
    with open(_SPLIT_SRC, encoding="utf-8") as fh:
        split = json.load(fh)
    items = []
    for it in split["test_split"]:
        p = str(it["file_path"])
        if not os.path.isabs(p):
            p = os.path.join(DATA_ROOT, p)
        items.append(dict(it, file_path=p))
    out = dict(split)
    out["test_split"] = items
    dst = WORK_DIR / "fixed_split_abs.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    n_missing = sum(1 for x in items if not os.path.exists(x["file_path"]))
    if n_missing:
        print("[occlusion_config] warning: %d / %d image(s) do not exist under data_root"
              " (data_root=%s) — only the CPU statistics chain is unaffected."
              % (n_missing, len(items), DATA_ROOT))
    return dst


def _abs_split_path():
    try:
        return str(_materialize_abs_split())
    except Exception as exc:        # an environment problem must not make the module import fail
        print("[occlusion_config] warning: failed to generate an absolute-path copy of the split (%r);"
              " falling back to the split file pointed to by config.yaml (its file_path is then required to be absolute itself)."
              % (exc,))
        return str(_SPLIT_SRC)


FIXED_DATA_SPLIT_JSON = _abs_split_path()

# Runtime environment note (the external training code directory is recorded if it exists, for
# traceability only; it takes no part in imports)
BASE_CODE_DIR = os.environ.get("AD3DCNN_MODEL_CODE_DIR", "")

# ==================== torch-related (lazy import) ====================

try:                                            # available in the GPU script runtime environment
    import torch as _torch

    DEVICE = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
except Exception:                               # unavailable in the pure-CPU statistics environment
    _torch = None
    DEVICE = None

device = DEVICE                                  # for compatibility with the original script naming


def apply_determinism():
    """Uniformly enable cuDNN deterministic algorithms.

    Diagnostics confirmed that repeated inference on the same tensor with the
    same batch size is bitwise identical (difference = 0), whereas batch=1 and batch=2 differ by
    2.4×10⁻⁴ — this is the result of cuDNN automatically selecting a convolution algorithm
    according to batch size. The occlusion experiment requires the two forward passes
    ("unoccluded / occluded") to be strictly comparable, hence deterministic algorithms are enforced.
    """
    if _torch is None:
        return
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False
    _torch.use_deterministic_algorithms(True, warn_only=True)


def load_best_config_from_csv(run_id=None):
    """Read the hyperparameter configuration of the specified Run from data/all_training_results.csv.

    ⚠ The ``config`` column of the CSV is a **string** (the repr of a Python dict) and must first be
    parsed into a dict; calling ``dict(...)`` on it directly raises
    ``ValueError: dictionary update sequence element``. Parsing uniformly goes through the
    repository's ``common.utils.parse_config_from_string`` (compatible with repr / JSON).
    """
    import pandas as pd

    run_id = BEST_RUN_ID if run_id is None else run_id
    df = pd.read_csv(GRID_RESULTS)
    row = df[df["run_id"] == run_id]
    if row.empty:
        raise ValueError("run_id = %s not found" % run_id)
    cfg = row.iloc[0]["config"]
    if not isinstance(cfg, dict):
        from common.utils import parse_config_from_string

        cfg = parse_config_from_string(cfg)
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError("the config column of run_id = %s cannot be parsed into a configuration dict: %r" % (run_id, cfg))
    return cfg


def load_checkpoint(path, map_location=None):
    """Load a checkpoint (consistent with the definition used by this repository's evaluate/ scripts)."""
    return _torch.load(str(path), map_location=map_location, weights_only=False)


def get_model(*args, **kwargs):
    """Lazy-import wrapper around ``model.densenet169_attention.get_model``."""
    from model.densenet169_attention import get_model as _f

    return _f(*args, **kwargs)


def prepare_model_for_gradcam(*args, **kwargs):
    """Lazy-import wrapper around ``common.utils.prepare_model_for_gradcam``."""
    from common.utils import prepare_model_for_gradcam as _f

    return _f(*args, **kwargs)


def get_data_transforms(*args, **kwargs):
    """Lazy-import wrapper around ``common.utils.get_data_transforms``."""
    from common.utils import get_data_transforms as _f

    return _f(*args, **kwargs)


def load_fixed_split():
    """Read the fixed test-set split; returns [{'file_path','label','abs_path','file_id'}, ...]."""
    split = json.load(open(FIXED_DATA_SPLIT_JSON, encoding="utf-8"))
    items = []
    for it in split["test_split"]:
        p = it["file_path"]
        abs_p = p if os.path.isabs(p) else os.path.join(DATA_ROOT, p)
        items.append(dict(file_path=p, abs_path=abs_p, label=int(it["label"]),
                          file_id=Path(p).name.replace(".nii.gz", "")))
    return items


def load_gradcam_csv():
    """Read the per-sample Grad-CAM ROI table of the paper's §3.3; returns {file_id: row}."""
    import csv as _csv

    out = {}
    with open(CSV, encoding="utf-8-sig") as fh:
        for r in _csv.DictReader(fh):
            out[r["file_id"]] = r
    return out


def describe():
    """Print the resolved paths and environment fingerprint (for the log record)."""
    print("[occlusion_config] repository root         = %s" % REPO_ROOT)
    print("[occlusion_config] work directory          = %s" % WORK_DIR)
    print("[occlusion_config] authoritative artifacts = %s" % REF_DIR)
    print("[occlusion_config] ADNI root               = %s" % DATA_ROOT)
    print("[occlusion_config] weights directory       = %s" % CKPT_DIR)
    print("[occlusion_config] Grad-CAM table          = %s" % CSV)
    print("[occlusion_config] Run=%d SEED=%d SAT_THR=%s device=%s"
          % (BEST_RUN_ID, SEED, SAT_THR, DEVICE))


if __name__ == "__main__":
    setup_stdout()
    describe()
