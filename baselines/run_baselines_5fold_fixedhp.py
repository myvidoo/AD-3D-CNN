# -*- coding: utf-8 -*-
"""
S9 supplementary recomputation: 5-fold cross-validation reproduction values (protocol
identical to the main model — the hyperparameters are selected once on the 810 subjects
and then held fixed, and each fold only retrains; the per-fold values cited in paper
3.5/S9/Table S6)
=====================================================================
Protocol (identical to the archived workspace script `cv5_tuned_fixedhp.py` in <WORKSPACE>/16_..._S9/):
  - Data: data/structural_panel.csv (ADNI subset) + data/data_lists/subject_summary_20260531.xlsx (age/sex)
  - First isolate the 144-subject fixed test set (data/fixed_data_split.json); the remaining
    810 subjects undergo 5-fold stratified CV
    (StratifiedKFold, shuffle=True, random_state=42, identical to the main model)
  - Hyperparameters as selected for the main results and held fixed (Table S6): B1 lr C=1;
    B2 lr C=0.1; B3 SVM C=1/scale; B4 lr C=0.1; B5 SVM C=10/γ=0.01
  - Metric: three-class macro-average AUC (one-vs-rest, arithmetic mean of the binarised per-class AUCs)

Assertion anchor: the per-fold values in section [A] of
reference_results/s9_baselines/_cv5_fixedhp.log (4 decimal places). Everything matching
digit-for-digit = the numbers in paper Table S6 are independently reproducible from this script.
For the nested version (per-fold independent parameter selection, B4 0.7115±0.0099) see
run_baselines_tuned.py.

Environment: ants_env (sklearn).
Usage:
  python baselines/run_baselines_5fold_fixedhp.py            # assertion mode
  python baselines/run_baselines_5fold_fixedhp.py --skip-assert   # recompute only, no assertion
  Output: results/baselines/baseline_5fold_cv_fixedhp.csv
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOL = ["gmr_hippocampus_700", "gmr_amygdala_700", "gmr_parahippocampal_700",
       "gmr_mtl_700", "vmr_latventricle", "brain_gm_ml"]
DEM = ["age", "sex_m"]

# assertion anchor = the per-fold values in section [A] of reference_results/s9_baselines/_cv5_fixedhp.log (4 decimals)
ANCHOR = {
    "B1": [0.4821, 0.4719, 0.5691, 0.5012, 0.4978],
    "B2": [0.7032, 0.6940, 0.7316, 0.6812, 0.6851],
    "B3": [0.6939, 0.6826, 0.7056, 0.7140, 0.7119],
    "B4": [0.7168, 0.7213, 0.7231, 0.7010, 0.7007],
    "B5": [0.7135, 0.7250, 0.7098, 0.7039, 0.7027],
}


def macro_auc(y, P):
    return float(np.mean([roc_auc_score((y == k).astype(int), P[:, k]) for k in range(3)]))


def mk(kind, hp):
    if kind == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, C=hp["C"], random_state=42))
    return make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", C=hp["C"], gamma=hp["gamma"],
                             probability=True, random_state=42))


def main():
    ap = argparse.ArgumentParser(description="S9 fixed-hyperparameter 5-fold reproduction-value recomputation")
    ap.add_argument("--skip-assert", action="store_true",
                    help="skip the per-fold assertion against the archived log (use when floating-point differences arise across sklearn versions)")
    args = ap.parse_args()

    # ---------- data (all paths relative to the package) ----------
    panel = pd.read_csv(os.path.join(BASE, "data", "structural_panel.csv"))
    demo = pd.read_excel(os.path.join(BASE, "data", "data_lists", "subject_summary_20260531.xlsx"))
    demo["Image Data ID"] = demo["Image Data ID"].astype(str).str.strip()

    a = panel[panel.cohort == "ADNI"].copy()
    a["Image Data ID"] = a.subject.astype(str).str.split("_").str[0]
    a = a.merge(demo[["Image Data ID", "Age", "Sex"]], on="Image Data ID", how="left")
    assert a["Age"].notna().all(), "missing age after merging the ADNI panel with the demographics list"
    a["age"] = a.Age.astype(float)
    a["sex_m"] = (a.Sex == "M").astype(int)

    with open(os.path.join(BASE, "data", "fixed_data_split.json"), encoding="utf-8") as f:
        fs = json.load(f)
    stems = {os.path.basename(x["file_path"]).replace("_ws.nii.gz", "")
             for x in fs["test_split"]}
    a["is_test"] = a.subject.isin(stems)
    tr = a[~a.is_test].reset_index(drop=True)
    assert len(tr) == 810, f"the training pool should hold 810 subjects, got {len(tr)}"
    y = tr.label.map({0.0: 0, 0.5: 1, 1.0: 2}).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    B = {
        "B1": ("B1 age+sex LR", DEM, "lr", {"C": 1.0}),
        "B2": ("B2 structural panel LR", VOL, "lr", {"C": 0.1}),
        "B3": ("B3 structural panel SVM", VOL, "svm", {"C": 1.0, "gamma": "scale"}),
        "B4": ("B4 panel+demo LR", VOL + DEM, "lr", {"C": 0.1}),
        "B5": ("B5 panel+demo SVM", VOL + DEM, "svm", {"C": 10.0, "gamma": 0.01}),
    }

    print("=" * 78)
    print("[S9] 5-fold reproduction values: hyperparameters selected once on the 810 subjects and then held fixed (identical protocol to the main model)")
    print("=" * 78)
    rows, all_ok = [], True
    for key, (name, feats, kind, hp) in B.items():
        aucs = []
        for tri, vai in skf.split(np.zeros(len(y)), y):
            m = mk(kind, hp)
            m.fit(tr[feats].values[tri], y[tri])
            P = m.predict_proba(tr[feats].values[vai])
            aucs.append(macro_auc(y[vai], P))
        mean, sd = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
        print("  %-26s AUC=%.4f±%.4f  folds %s  hp=%s"
              % (name, mean, sd, ";".join("%.4f" % x for x in aucs), hp))
        # assertion: matches the archived log's per-fold values (4 decimals)
        ok = all(abs(x - x0) < 5.1e-5 for x, x0 in zip(aucs, ANCHOR[key]))
        all_ok &= ok
        print("       anchor check [%s]: %s" % (key, "OK" if ok else "MISMATCH vs _cv5_fixedhp.log"))
        rows.append({"baseline": name, "key": key, "AUC_mean": mean, "AUC_sd": sd,
                     "folds": ";".join("%.6f" % x for x in aucs),
                     "hp": json.dumps(hp, ensure_ascii=False),
                     "anchor_ok": ok})

    out_csv = os.path.join(BASE, "results", "baselines", "baseline_5fold_cv_fixedhp.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\nSaved -> %s" % out_csv)
    if not args.skip_assert:
        print("\n" + ("[all match] the fixed-hyperparameter 5-fold reproduction values of paper Table S6 are independently reproducible from this script."
                      if all_ok else "[mismatch] the per-fold values do not match the archived log; please check the data version."))
        return 0 if all_ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
