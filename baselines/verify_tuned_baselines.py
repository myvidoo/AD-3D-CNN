# -*- coding: utf-8 -*-
"""Verification script: ① the exact macro-average AUC of the ensemble; ② whether the Table 4 head-to-head values can be reproduced under "untuned B4"."""
import os, io, sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
# [AD_CNN_code adaptation] paths made relative to this repository layout
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S9 = os.path.join(BASE, "results", "baselines")
VOL_FEATS = ["gmr_hippocampus_700", "gmr_amygdala_700", "gmr_parahippocampal_700",
             "gmr_mtl_700", "vmr_latventricle", "brain_gm_ml"]
DEMO_FEATS = ["age", "sex_m"]


def macro_auc(y, P):
    return float(np.mean([roc_auc_score((y == k).astype(int), P[:, k]) for k in range(3)]))


panel = pd.read_csv(os.path.join(BASE, "data", "structural_panel.csv"))
d = pd.read_excel(os.path.join(BASE, "data", "data_lists", "subject_summary_20260531.xlsx"))
d["Image Data ID"] = d["Image Data ID"].astype(str).str.strip()
a = panel[panel.cohort == "ADNI"].copy()
a["Image Data ID"] = a.subject.astype(str).str.split("_").str[0]
a = a.merge(d[["Image Data ID", "Age", "Sex"]], on="Image Data ID", how="left")
a["age"] = a.Age.astype(float)
a["sex_m"] = (a.Sex == "M").astype(int)
fs = json.load(open(os.path.join(BASE, "data", "fixed_data_split.json"), encoding="utf-8"))
test_stems = {os.path.basename(x["file_path"]).replace("_ws.nii.gz", "") for x in fs["test_split"]}
a["is_test"] = a.subject.isin(test_stems)
tr = a[~a.is_test].reset_index(drop=True)
te = a[a.is_test].reset_index(drop=True)
y_tr = tr.label.map({0.0: 0, 0.5: 1, 1.0: 2}).values
y_te = te.label.map({0.0: 0, 0.5: 1, 1.0: 2}).values

# ---------- ① exact ensemble AUC ----------
e = pd.read_csv(os.path.join(BASE, "data", "ensemble_test_predictions.csv"))
cmap = {os.path.basename(p).replace("_ws.nii.gz", ""): i for i, p in enumerate(e.file_path)}
order = [cmap[s] for s in te.subject]
Pe = e[["prob_cn", "prob_mci", "prob_ad"]].values[order]
print("=" * 70)
print("[1] exact macro-average AUC of the ensemble model")
print("=" * 70)
ma = macro_auc(y_te, Pe)
print("  macro-AUC to 6 decimals = %.8f" % ma)
print("  %.4f formatted -> %s" % (ma, ("%.4f" % ma)))
print("  per-class AUC: CN %.6f | MCI %.6f | AD %.6f"
      % tuple(roc_auc_score((y_te == k).astype(int), Pe[:, k]) for k in range(3)))
print("  → the paper uses 0.9630 throughout (4 decimals). The values above are authoritative.")

# ---------- ② Table 4 head-to-head ----------
print()
print("=" * 70)
print("[2] Table 4 head-to-head (structural panel + demographics = 8 features) reproduction check")
print("=" * 70)


def headtohead(C_lr, C_svm=None, gamma=None, svm=False):
    out = []
    for tag, lo, hi in [("CN vs AD", 0, 2), ("CN vs MCI", 0, 1), ("MCI vs AD", 1, 2)]:
        sel = np.isin(y_te, [lo, hi])
        yb = (y_te[sel] == hi).astype(int)
        s_cnn = Pe[sel, hi] / (Pe[sel, lo] + Pe[sel, hi])
        auc_cnn = roc_auc_score(yb, s_cnn)
        trsel = np.isin(y_tr, [lo, hi])
        X = tr[VOL_FEATS + DEMO_FEATS].values
        if svm:
            m = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C_svm, gamma=gamma,
                                                    probability=True, random_state=42))
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C_lr,
                                                                  random_state=42))
        m.fit(X[trsel], (y_tr[trsel] == hi).astype(int))
        s_b = m.predict_proba(te[VOL_FEATS + DEMO_FEATS].values[sel])[:, 1]
        auc_b = roc_auc_score(yb, s_b)
        out.append((tag, auc_cnn, auc_b, (auc_cnn - auc_b) * 100))
    return out


print("\n(a) untuned B4 (LR C=1.0) — expected to reproduce paper Table 4: 0.9006 / 0.6819 / 0.6918")
for tag, ac, ab, dl in headtohead(1.0):
    print("    %-10s CNN=%.6f  panel=%.6f  Δ=%+.1f pp" % (tag, ac, ab, dl))

print("\n(b) tuned B4 (LR C=0.1) — new values")
for tag, ac, ab, dl in headtohead(0.1):
    print("    %-10s CNN=%.6f  panel=%.6f  Δ=%+.1f pp" % (tag, ac, ab, dl))

print("\n(c) tuned B5 (SVM C=10, gamma=0.01) — for reference")
for tag, ac, ab, dl in headtohead(1.0, C_svm=10.0, gamma=0.01, svm=True):
    print("    %-10s CNN=%.6f  panel=%.6f  Δ=%+.1f pp" % (tag, ac, ab, dl))

# ---------- ③ tuned main results to 6 decimals ----------
print()
print("=" * 70)
print("[3] tuned baseline main results (6 decimals)")
print("=" * 70)
tuned = pd.read_csv(os.path.join(S9, "baseline_test_set_performance_tuned.csv"))
unt = pd.read_csv(os.path.join(S9, "baseline_test_set_performance_untuned_recheck.csv"))
for (_, r), (_, u) in zip(tuned.iterrows(), unt.iterrows()):
    print("  %-30s untuned %.6f → tuned %.6f  | Acc %.4f  F1 %.4f  CI [%.4f, %.4f]  Δ%+.1f p=%.4f  hp=%s"
          % (r["baseline"], u["macro_auc"], r["macro_auc"], r["acc"], r["macro_f1"],
             r["macro_auc_ci_lo"], r["macro_auc_ci_hi"], r["delta_vs_cnn"], r["p_paired"], r["hp"]))
