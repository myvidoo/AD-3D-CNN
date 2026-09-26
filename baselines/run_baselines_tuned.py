# -*- coding: utf-8 -*-
"""Tuned-hyperparameter version of the controlled baselines (reviewer comment M2: add "standard grid training").

[the only difference from run_baselines.py]
  Original: logistic regression with C=1.0 fixed, RBF-SVM with C=1.0/gamma="scale" fixed, no tuning at all.
  This version: adds a standard grid for each baseline family and uses **inner cross-validation**
        to select parameters on the 810 non-test subjects, then trains/evaluates with the chosen
        values. The test set (144 subjects) takes no part in parameter selection at any stage.

[grids (standard)]
  Logistic regression: C ∈ {0.1, 1, 10}                                   (3 settings)
  RBF-SVM: C ∈ {0.1, 1, 10} × gamma ∈ {scale, 1e-3, 1e-2}      (9 settings)

[parameter-selection protocol (leakage-proof)]
  Inner: StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
        — mutually independent of the main model's outer split (random_state=42).
  Criterion: highest mean inner 5-fold macro-average AUC; ties broken towards the smaller C (more conservative).
  Scope:
    Main results — inner CV parameter selection on all 810 subjects → retrain on all 810 subjects → evaluate on the 144-subject test set
    5-fold reproduction — within each outer fold, inner CV parameter selection uses only that fold's 648 training subjects (nested CV)
    External cohorts — reuse the hyperparameters selected for the main results on the 810 subjects (zero-fine-tuning extrapolation)

[output] everything is written to *_tuned.csv / baseline_tuned_grid.json; the original result files are not overwritten.
"""
import os, io, sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
N_BOOT = 2000
INNER_SEED = 2026
OUTER_SEED = 42

# [AD_CNN_code adaptation] paths made relative to this repository layout; output written to results/baselines/
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # AD_CNN_code root
OUTDIR = os.path.join(BASE, "results", "baselines")
os.makedirs(OUTDIR, exist_ok=True)

PANEL = os.path.join(BASE, "data", "structural_panel.csv")
DEMO = os.path.join(BASE, "data", "data_lists", "subject_summary_20260531.xlsx")
FIXED = os.path.join(BASE, "data", "fixed_data_split.json")
CVSTATE = os.path.join(BASE, "reference_results", "cv_ensemble", "cv_state_run_128.json")
ENS = os.path.join(BASE, "data", "ensemble_test_predictions.csv")

VOL_FEATS = ["gmr_hippocampus_700", "gmr_amygdala_700", "gmr_parahippocampal_700",
             "gmr_mtl_700", "vmr_latventricle", "brain_gm_ml"]
DEMO_FEATS = ["age", "sex_m"]

LR_GRID = [{"C": c} for c in (0.1, 1.0, 10.0)]
SVM_GRID = [{"C": c, "gamma": g} for c in (0.1, 1.0, 10.0)
            for g in ("scale", 1e-3, 1e-2)]


def _assert_inputs():
    for p in (PANEL, DEMO, FIXED, CVSTATE, ENS):
        assert os.path.isfile(p), "missing input file: %s" % p


def make_model(kind, hp):
    if kind == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, C=hp["C"], random_state=42))
    if kind == "svm":
        return make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", C=hp["C"], gamma=hp["gamma"],
                                 probability=True, random_state=42))
    raise ValueError(kind)


def grid_of(kind):
    return LR_GRID if kind == "lr" else SVM_GRID


def macro_auc(y, P):
    return float(np.mean([roc_auc_score((y == k).astype(int), P[:, k]) for k in range(3)]))


def metrics(y, P):
    pred = P.argmax(1)
    out = {"acc": float(accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro")),
           "macro_auc": macro_auc(y, P)}
    for k in range(3):
        out["auc_c%d" % k] = float(roc_auc_score((y == k).astype(int), P[:, k]))
    return out


def select_hp(kind, X, y):
    """Inner 5-fold CV parameter selection. Returns (best_hp, inner scores of all candidates)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=INNER_SEED)
    scores = []
    for hp in grid_of(kind):
        fold_auc = []
        for tr_i, va_i in skf.split(X, y):
            m = make_model(kind, hp)
            m.fit(X[tr_i], y[tr_i])
            fold_auc.append(macro_auc(y[va_i], m.predict_proba(X[va_i])))
        scores.append((float(np.mean(fold_auc)), hp))
    scores.sort(key=lambda t: (-t[0], t[1].get("C", 0)))
    return scores[0][1], [{"hp": s[1], "inner_auc": round(s[0], 6)} for s in scores]


def fit_predict(kind, hp, Xtr, ytr, Xte):
    m = make_model(kind, hp)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)


def load_all():
    p = pd.read_csv(PANEL)
    print("[data] structural_panel.csv %s  cohort %s" % (p.shape, p.cohort.value_counts().to_dict()))
    d = pd.read_excel(DEMO)
    d["Image Data ID"] = d["Image Data ID"].astype(str).str.strip()
    a = p[p.cohort == "ADNI"].copy()
    a["Image Data ID"] = a.subject.astype(str).str.split("_").str[0]
    a = a.merge(d[["Image Data ID", "Age", "Sex"]], on="Image Data ID", how="left")
    miss = int(a.Age.isna().sum())
    print("[demographics] missing age after the ADNI merge %d/%d" % (miss, len(a)))
    assert miss == 0, "missing age; the ID concatenation needs checking"
    a["age"] = a.Age.astype(float)
    a["sex_m"] = (a.Sex == "M").astype(int)
    fs = json.load(open(FIXED, encoding="utf-8"))
    test_stems = {os.path.basename(x["file_path"]).replace("_ws.nii.gz", "")
                  for x in fs["test_split"]}
    print("[split] fixed test set %d subjects" % len(test_stems))
    return p, a, test_stems


def main():
    global N_BOOT
    import argparse
    ap = argparse.ArgumentParser(description="S9 controlled baselines (v58 tuned version, the main definition for paper Tables 3/4 and S9)")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="number of bootstrap resamples (2000 per the paper's definition; --n-boot 200 is fine for a smoke test)")
    args = ap.parse_args()
    N_BOOT = args.n_boot
    _assert_inputs()
    panel, adni, test_stems = load_all()
    adni["is_test"] = adni.subject.isin(test_stems)

    cv = json.load(open(CVSTATE, encoding="utf-8"))
    tr_all = adni[~adni.is_test].reset_index(drop=True)
    te = adni[adni.is_test].reset_index(drop=True)
    y_all = tr_all.label.map({0.0: 0, 0.5: 1, 1.0: 2}).values
    y_te = te.label.map({0.0: 0, 0.5: 1, 1.0: 2}).values
    print("[samples] non-test %d subjects  test %d subjects  classes %s" %
          (len(tr_all), len(te), pd.Series(y_te).value_counts().sort_index().to_dict()))
    assert len(te) == 144 and len(tr_all) == 810

    skf_out = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    ref_folds = [va for _, va in skf_out.split(np.zeros(len(y_all)), y_all)]
    print("[5-fold] per-fold outer validation %s (main model fold_splits=%d)"
          % ([len(f) for f in ref_folds], len(cv["fold_splits"])))

    BASELINES = {
        "B1 age+sex LR": (DEMO_FEATS, "lr"),
        "B2 structural panel LR": (VOL_FEATS, "lr"),
        "B3 structural panel SVM": (VOL_FEATS, "svm"),
        "B4 panel+demo LR": (VOL_FEATS + DEMO_FEATS, "lr"),
        "B5 panel+demo SVM": (VOL_FEATS + DEMO_FEATS, "svm"),
    }

    # ---------- parameter selection (uses the 810 subjects only) ----------
    print("\n[inner parameter selection] uses only the 810 non-test subjects, inner 5 folds (seed=%d)" % INNER_SEED)
    chosen, sel_log = {}, {}
    for name, (feats, kind) in BASELINES.items():
        hp, log = select_hp(kind, tr_all[feats].values, y_all)
        chosen[name] = hp
        sel_log[name] = {"kind": kind, "n_grid": len(grid_of(kind)),
                         "chosen": hp, "candidates": log}
        print("  %-30s grid of %d settings → selected %s" % (name, len(grid_of(kind)), hp))

    # ---------- main results ----------
    # baseline reproduction (untuned) as the control
    print("\n[control] untuned (original protocol C=1.0 / gamma=scale)")
    rows_raw = []
    for name, (feats, kind) in BASELINES.items():
        hp0 = {"C": 1.0} if kind == "lr" else {"C": 1.0, "gamma": "scale"}
        P0 = fit_predict(kind, hp0, tr_all[feats].values, y_all, te[feats].values)
        m = metrics(y_te, P0)
        rows_raw.append({"baseline": name, **m, "n_features": len(feats)})
        print("  %-30s Acc=%.4f  macroF1=%.4f  macroAUC=%.4f"
              % (name, m["acc"], m["macro_f1"], m["macro_auc"]))

    print("\n[main results] after tuning (inner parameter selection → retrain on 810 subjects → evaluation on the 144-subject test set)")
    rows, probs = [], {}
    for name, (feats, kind) in BASELINES.items():
        P = fit_predict(kind, chosen[name], tr_all[feats].values, y_all, te[feats].values)
        m = metrics(y_te, P)
        probs[name] = P
        rows.append({"baseline": name, **m, "n_features": len(feats),
                     "hp": json.dumps(chosen[name], ensure_ascii=False)})
        print("  %-30s Acc=%.4f  macroF1=%.4f  macroAUC=%.4f"
              % (name, m["acc"], m["macro_f1"], m["macro_auc"]))

    # ---------- Bootstrap CI ----------
    rng = np.random.default_rng(42)

    def strat_boot_idx(y):
        idx = []
        for k in np.unique(y):
            kk = np.where(y == k)[0]
            idx.append(rng.choice(kk, size=len(kk), replace=True))
        return np.concatenate(idx)

    for r in rows:
        P = probs[r["baseline"]]
        bs = np.array([metrics(y_te[i], P[i])["macro_auc"]
                       for i in (strat_boot_idx(y_te) for _ in range(N_BOOT))])
        r["macro_auc_ci_lo"] = float(np.percentile(bs, 2.5))
        r["macro_auc_ci_hi"] = float(np.percentile(bs, 97.5))

    # ---------- paired bootstrap against the ensemble model ----------
    e = pd.read_csv(ENS)
    cmap = {os.path.basename(p).replace("_ws.nii.gz", ""): i for i, p in enumerate(e.file_path)}
    order = [cmap[s] for s in te.subject]
    Pe = e[["prob_cn", "prob_mci", "prob_ad"]].values[order]
    assert (e.true_label.values[order] == y_te).all(), "the ensemble predictions disagree on sample order/labels"
    m_e = metrics(y_te, Pe)
    print("\n[ensemble] Acc=%.4f  macroF1=%.4f  macroAUC=%.4f"
          % (m_e["acc"], m_e["macro_f1"], m_e["macro_auc"]))

    d_obs = m_e["macro_auc"] - np.array([r["macro_auc"] for r in rows])
    boot_d = []
    for _ in range(N_BOOT):
        i = strat_boot_idx(y_te)
        ae = macro_auc(y_te[i], Pe[i])
        boot_d.append([ae - macro_auc(y_te[i], probs[r["baseline"]][i]) for r in rows])
    boot_d = np.array(boot_d)
    for j, r in enumerate(rows):
        r["delta_vs_cnn"] = float(d_obs[j] * 100)
        r["delta_ci_lo"] = float(np.percentile(boot_d[:, j], 2.5) * 100)
        r["delta_ci_hi"] = float(np.percentile(boot_d[:, j], 97.5) * 100)
        r["p_paired"] = max(float(2 * min((boot_d[:, j] <= 0).mean(),
                                          (boot_d[:, j] >= 0).mean())), 1.0 / N_BOOT)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTDIR, "baseline_test_set_performance_tuned.csv"),
               index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_raw).to_csv(os.path.join(OUTDIR, "baseline_test_set_performance_untuned_recheck.csv"),
                                  index=False, encoding="utf-8-sig")
    pd.DataFrame([{"baseline": "CNN 5-fold ensemble", **m_e}]).to_csv(
        os.path.join(OUTDIR, "cnn_ensemble_test_set_performance_tuned.csv"),
        index=False, encoding="utf-8-sig")

    print("\n%-30s%8s%8s%8s%20s%14s%11s" % ("baseline (tuned)", "Acc", "F1", "AUC", "95% CI", "ΔAUC vs CNN", "p (paired)"))
    print("-" * 100)
    print("%-30s%8.4f%8.4f%8.4f" % ("CNN 5-fold ensemble", m_e["acc"], m_e["macro_f1"], m_e["macro_auc"]))
    for r in rows:
        print("%-30s%8.4f%8.4f%8.4f  [%.4f,%.4f]%+13.1f%11.4f"
              % (r["baseline"], r["acc"], r["macro_f1"], r["macro_auc"],
                 r["macro_auc_ci_lo"], r["macro_auc_ci_hi"], r["delta_vs_cnn"], r["p_paired"]))

    # ---------- head-to-head two-class subsets (using the tuned model) ----------
    print("\n[head-to-head two-class subsets n=96]")
    hh = []
    PAIR = [("CN vs AD", 0, 2), ("CN vs MCI", 0, 1), ("MCI vs AD", 1, 2)]
    best_hi = max(rows, key=lambda r: r["macro_auc"])["baseline"]
    feats_hi, kind_hi = BASELINES[best_hi]
    for tag, a, b in PAIR:
        sel = np.isin(y_te, [a, b])
        yb = (y_te[sel] == b).astype(int)
        # CNN: two-class probability normalisation
        s_cnn = Pe[sel, b] / (Pe[sel, a] + Pe[sel, b])
        auc_cnn = roc_auc_score(yb, s_cnn)
        # Baseline: refitted on the corresponding two classes of the training set only (same tuned hyperparameters)
        trsel = np.isin(y_all, [a, b])
        mb = make_model(kind_hi, chosen[best_hi])
        mb.fit(tr_all[feats_hi].values[trsel], (y_all[trsel] == b).astype(int))
        s_b = mb.predict_proba(te[feats_hi].values[sel])[:, 1]
        auc_b = roc_auc_score(yb, s_b)
        hh.append({"comparison": tag, "n": int(sel.sum()), "cnn_auc": float(auc_cnn),
                   "panel_auc": float(auc_b), "delta_pp": float((auc_cnn - auc_b) * 100),
                   "panel_source": best_hi})
        print("  %-10s n=%d  CNN=%.4f  panel=%.4f  Δ=%+.1f pp"
              % (tag, int(sel.sum()), auc_cnn, auc_b, (auc_cnn - auc_b) * 100))
    pd.DataFrame(hh).to_csv(os.path.join(OUTDIR, "baseline_headtohead_tuned.csv"),
                            index=False, encoding="utf-8-sig")

    # ---------- 5-fold nested CV ----------
    print("\n[5-fold nested CV] independent parameter selection inside each fold (using only that fold's 648 training subjects)")
    cv_rows = []
    for name, (feats, kind) in BASELINES.items():
        aucs, hps = [], []
        for tr_i, va_i in skf_out.split(np.zeros(len(y_all)), y_all):
            hp_f, _ = select_hp(kind, tr_all[feats].values[tr_i], y_all[tr_i])
            P = fit_predict(kind, hp_f, tr_all[feats].values[tr_i], y_all[tr_i],
                            tr_all[feats].values[va_i])
            aucs.append(macro_auc(y_all[va_i], P))
            hps.append(hp_f)
        cv_rows.append({"baseline": name, "AUC_mean": float(np.mean(aucs)),
                        "AUC_sd": float(np.std(aucs, ddof=1)),
                        "AUC_min": float(np.min(aucs)), "AUC_max": float(np.max(aucs)),
                        "folds": ";".join("%.4f" % a for a in aucs),
                        "chosen_per_fold": json.dumps(hps, ensure_ascii=False)})
        print("  %-30s AUC=%.4f±%.4f" % (name, np.mean(aucs), np.std(aucs, ddof=1)))
    pd.DataFrame(cv_rows).to_csv(os.path.join(OUTDIR, "baseline_5fold_cv_tuned.csv"),
                                 index=False, encoding="utf-8-sig")

    # ---------- external cohorts (zero fine-tuning, reusing the hyperparameters selected for the main results) ----------
    print("\n[external cohorts: zero-fine-tuning extrapolation of the structural panel (tuned B2)]")
    ext_rows = []
    b2_feats, b2_kind = BASELINES["B2 structural panel LR"]
    for cohort, tag in [("MIRIAD", "MIRIAD"), ("OASIS2", "OASIS-2")]:
        sub = panel[panel.cohort == cohort].copy()
        P = fit_predict(b2_kind, chosen["B2 structural panel LR"],
                        tr_all[b2_feats].values, y_all, sub[b2_feats].values)
        lab = sub.label.values
        if cohort == "MIRIAD":
            auc = roc_auc_score((lab == 1.0).astype(int), P[:, 2])
            ext_rows.append({"cohort": cohort, "n": len(sub), "view": "AD vs HC (P(AD))",
                             "baseline": "B2 structural panel LR (tuned)", "AUC": float(auc)})
            print("  %s: AD vs HC  AUC=%.4f  n=%d" % (tag, auc, len(sub)))
        else:
            keep = lab != 0.5
            s = P[keep, 2] / (P[keep, 2] + P[keep, 0])
            auc = roc_auc_score((lab[keep] == 1.0).astype(int), s)
            ext_rows.append({"cohort": cohort, "n": int(keep.sum()), "view": "CN vs AD",
                             "baseline": "B2 structural panel LR (tuned)", "AUC": float(auc)})
            print("  %s: CN vs AD   AUC=%.4f  n=%d" % (tag, auc, int(keep.sum())))
    pd.DataFrame(ext_rows).to_csv(os.path.join(OUTDIR, "baseline_external_cohorts_tuned.csv"),
                                  index=False, encoding="utf-8-sig")

    json.dump({"inner_seed": INNER_SEED, "outer_seed": OUTER_SEED,
               "lr_grid": LR_GRID, "svm_grid": SVM_GRID,
               "selected": chosen, "selection_detail": sel_log,
               "cnn": m_e,
               "test_set_tuned": {r["baseline"]: {k: v for k, v in r.items()} for r in rows},
               "headtohead_tuned": hh},
              open(os.path.join(OUTDIR, "baseline_tuned_grid.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nDone -> %s" % OUTDIR)


if __name__ == "__main__":
    main()
