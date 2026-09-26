# -*- coding: utf-8 -*-
"""Unified configuration loading: reads config.yaml to inject paths and hyperparameters into each script."""

import os
import sys

import yaml

# ---- Console robustness fallback (Windows Chinese-locale environment) -----
# The default console encoding on Chinese Windows is GBK; when a script prints
# characters such as η², δ, ρ, ± or × it raises UnicodeEncodeError and aborts the
# run. Here the stdout/stderr encoding error policy is changed to "replace", which
# keeps the console's original encoding (Chinese still displays as usual) while
# degrading the few unmappable characters to '?' instead of crashing.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass
# ---------------------------------------------------------------------------


def load_config(path=None):
    """Load config.yaml and return a dict. path defaults to config.yaml alongside this file."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(cfg, key_path, base_dir=None):
    """Fetch a value by dotted path of the form 'data.fixed_split' and resolve it to an absolute path relative to base_dir."""
    node = cfg
    for key in key_path.split("."):
        node = node[key]
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    if node and not os.path.isabs(node):
        return os.path.join(base_dir, node)
    return node



def first_existing(*candidates, default=None):
    """Return the first candidate path that exists (used by the read-side archive fallback)."""
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return default if default is not None else (candidates[0] if candidates else None)


def resolve_with_archive(cfg, name, subdir=None):
    """Read an archived artifact shipped inside the package.

    Look first under output_dir relative to the current working directory (matching where the
    scripts write); if that is absent, fall back to <package>/<output_dir>/[subdir/]name so the
    archived input is still found when the script is launched from outside the package.
    """
    parts = [cfg["output_dir"]] + ([subdir] if subdir else []) + [name]
    rel = os.path.join(*parts)
    pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)
    return first_existing(rel, pkg)
