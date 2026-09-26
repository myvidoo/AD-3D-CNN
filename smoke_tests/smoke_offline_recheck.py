# -*- coding: utf-8 -*-
# ============================================================================
# smoke_tests/smoke_offline_recheck.py — offline reproduction smoke test (no images/GPU/weights)
#
# Purpose: from the distributed 144 per-sample prediction probabilities
#          (data/ensemble_test_predictions.csv) alone, recompute all core metrics of the
#          paper's Table 2 / Table S5 and assert them against the manuscript's numeric anchors;
#          it also verifies the RUN128 configuration and validation AUC in the 256
#          grid-search results (data/all_training_results.csv).
#
# Implementation: pure Python standard library (csv/math), no numpy/pandas dependency,
#          runnable on any Python 3.8+.
# Run: python smoke_tests/smoke_offline_recheck.py
# Exit code: 0 = all anchors pass; 1 = an inconsistency exists (details printed).
#
# Expected output:
#   Acc 0.8681 | macro F1 0.8675 | macro AUC 0.9630 | class AUC 0.9529/0.9542/0.9818
#   confusion matrix [[41,5,2],[5,39,4],[0,3,45]] | macro Brier 0.0748 | macro ECE 0.0774
#   argmax ECE 0.1163 | mean confidence 0.9717 | correct/incorrect confidence 0.9838/0.8923
#   RUN128 = axial + lr 1e-4, best AUC 0.9665
# ============================================================================
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENS = os.path.join(ROOT, "data", "ensemble_test_predictions.csv")
GRID = os.path.join(ROOT, "data", "all_training_results.csv")

# Numeric anchors (Table 2 / Table S5 / section 3.1 / S2)
TOL = 5e-4
ANCHORS = {
    "acc":            (125 / 144, TOL),        # 0.8681
    "macro_f1":       (0.8675, 5e-4),
    "macro_auc":      (0.9630, 5e-4),
    "auc_cn":         (0.9529, 5e-4),
    "auc_mci":        (0.9542, 5e-4),
    "auc_ad":         (0.9818, 5e-4),
    "macro_brier":    (0.0748, 5e-4),
    "macro_ece_ovr":  (0.0774, 5e-4),          # used from paper v66 onward: 0.0774 (the earlier 0.0772 was a bin-misalignment bug; see BUG-01 in the audit report)
    "argmax_ece":     (0.1163, 2e-3),
    "mean_conf":      (0.9717, 1e-3),
    "conf_correct":   (0.9838, 1e-3),
    "conf_wrong":     (0.8923, 1e-3),
}
# (true, pred) -> n; rows = true. Paper Table 2/S5: CN [41,5,2] / MCI [5,39,4] / AD [0,3,45]
CONFUSION_EXPECTED = {(2, 2): 45, (2, 1): 3, (2, 0): 0,
                      (1, 1): 39, (1, 0): 5, (1, 2): 4,
                      (0, 0): 41, (0, 1): 5, (0, 2): 2}


def auc_score(y_true_bin, scores):
    """Mann-Whitney rank AUC (with ties handling)."""
    pairs = sorted(zip(scores, y_true_bin))
    n = len(pairs)
    # ranks (1-based, ties handled by averaging)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    r_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    n_pos = sum(y for _, y in pairs)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ece_10bin(conf, acc, n):
    """10 equal-width bin ECE (weighted by bin sample count; the last bin [0.9, 1.0] is closed)."""
    total = 0.0
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        idx = [i for i in range(n)
               if (conf[i] > lo or (b == 0 and conf[i] >= lo)) and conf[i] <= hi]
        if not idx:
            continue
        a = sum(acc[i] for i in idx) / len(idx)
        c = sum(conf[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(a - c)
    return total


def main():
    if not os.path.exists(ENS):
        print(f"[FAIL] missing data file: {ENS}")
        return 1
    with open(ENS, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"[load] ensemble_test_predictions.csv: {len(rows)} rows")
    assert len(rows) == 144, "prediction probabilities should cover 144 subjects"

    y = [int(r["true_label"]) for r in rows]
    P = [[float(r["prob_cn"]), float(r["prob_mci"]), float(r["prob_ad"])] for r in rows]
    pred = [max(range(3), key=lambda k: p[k]) for p in P]
    n = len(y)
    CLS = {0: "CN", 1: "MCI", 2: "AD"}

    # ---- confusion matrix / Acc / macro P / R / F1 ----
    cm = {(a, b): 0 for a in range(3) for b in range(3)}
    for t, p in zip(y, pred):
        cm[(t, p)] += 1
    acc = sum(1 for t, p in zip(y, pred) if t == p) / n
    f1s, aucs = [], []
    for k in range(3):
        tp = cm[(k, k)]
        fp = sum(cm[(j, k)] for j in range(3)) - tp
        fn = sum(cm[(k, j)] for j in range(3)) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        yb = [1 if t == k else 0 for t in y]
        aucs.append(auc_score(yb, [p[k] for p in P]))
    macro_f1 = sum(f1s) / 3
    macro_auc = sum(aucs) / 3

    # ---- Brier / ECE (one-vs-rest macro-average) ----
    briers = []
    eces = []
    for k in range(3):
        yb = [1.0 if t == k else 0.0 for t in y]
        pk = [p[k] for p in P]
        briers.append(sum((p - t) ** 2 for p, t in zip(pk, yb)) / n)
        eces.append(ece_10bin(pk, yb, n))
    macro_brier = sum(briers) / 3
    macro_ece = sum(eces) / 3

    # ---- top-label (argmax) confidence and ECE ----
    conf = [max(p) for p in P]
    acc_flag = [1.0 if t == p else 0.0 for t, p in zip(y, pred)]
    argmax_ece = ece_10bin(conf, acc_flag, n)
    conf_correct = [c for c, a in zip(conf, acc_flag) if a == 1.0]
    conf_wrong = [c for c, a in zip(conf, acc_flag) if a == 0.0]
    mean_conf = sum(conf) / n
    mean_cc = sum(conf_correct) / len(conf_correct)
    mean_cw = sum(conf_wrong) / len(conf_wrong)

    # ---- print ----
    print(f"\n[recomputed results] (Table 2 / Table S5)")
    print(f"  Acc={acc:.4f} ({sum(acc_flag):.0f}/{n})  macro F1={macro_f1:.4f}  macro AUC={macro_auc:.4f}")
    print(f"  class AUC: CN={aucs[0]:.4f}  MCI={aucs[1]:.4f}  AD={aucs[2]:.4f}")
    print(f"  macro Brier={macro_brier:.4f}  macro ECE(ovr)={macro_ece:.4f}  argmaxECE={argmax_ece:.4f}")
    print(f"  mean confidence={mean_conf:.4f}  correct={mean_cc:.4f}  incorrect={mean_cw:.4f}  gap={mean_cc - mean_cw:.4f}")
    print("  confusion matrix (rows=true, cols=predicted CN/MCI/AD):")
    for t in range(3):
        print(f"    {CLS[t]:>3}: [{cm[(t,0)]}, {cm[(t,1)]}, {cm[(t,2)]}]")

    # ---- assertions ----
    got = {
        "acc": acc, "macro_f1": macro_f1, "macro_auc": macro_auc,
        "auc_cn": aucs[0], "auc_mci": aucs[1], "auc_ad": aucs[2],
        "macro_brier": macro_brier, "macro_ece_ovr": macro_ece,
        "argmax_ece": argmax_ece, "mean_conf": mean_conf,
        "conf_correct": mean_cc, "conf_wrong": mean_cw,
    }
    fails = []
    print(f"\n[anchor comparison] tolerance {TOL}")
    for name, (exp, tol) in ANCHORS.items():
        g = got[name]
        ok = abs(g - exp) <= tol
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<14} got={g:.4f}  expect={exp:.4f}")
        if not ok:
            fails.append(name)
    # cell-by-cell confusion-matrix assertions
    for (t, p), exp_n in CONFUSION_EXPECTED.items():
        if cm[(t, p)] != exp_n:
            fails.append(f"cm[{CLS[t]}->{CLS[p]}]={cm[(t,p)]} expect {exp_n}")
            print(f"  [FAIL] confusion matrix {CLS[t]}->{CLS[p]}: got {cm[(t,p)]}, expect {exp_n}")
    if len(fails) == 0:
        print("  [PASS] all 9 confusion-matrix cells agree")

    # ---- grid-search result verification ----
    print(f"\n[grid search] all_training_results.csv")
    if not os.path.exists(GRID):
        print(f"  [WARN] missing {GRID} (skipping grid verification)")
    else:
        with open(GRID, "r", encoding="utf-8-sig") as f:
            grows = list(csv.DictReader(f))
        n_completed = sum(1 for g in grows if g.get("status") == "completed")
        print(f"  rows={len(grows)}  completed={n_completed}")
        if len(grows) != 256 or n_completed != 256:
            fails.append(f"grid rows/completed = {len(grows)}/{n_completed} expect 256/256")
        run128 = [g for g in grows if g["run_id"].strip() == "128"]
        if not run128:
            fails.append("RUN128 not found")
        else:
            r = run128[0]
            cfg = r["config"]
            best_auc = float(r["best_auc"])
            ok_cfg = ("'attention_type': 'axial'" in cfg) and ("'learning_rate': 0.0001" in cfg)
            ok_auc = abs(best_auc - 0.9665) <= 5e-4
            print(f"  [{'PASS' if ok_cfg else 'FAIL'}] RUN128 config = axial + lr=1e-4")
            print(f"  [{'PASS' if ok_auc else 'FAIL'}] RUN128 best AUC = {best_auc:.4f} (expect 0.9665)")
            if not ok_cfg:
                fails.append("RUN128 config")
            if not ok_auc:
                fails.append("RUN128 auc")

    # ---- conclusion ----
    print("\n" + "=" * 62)
    if fails:
        print(f"[SMOKE FAIL] {len(fails)} inconsistencies:")
        for f_ in fails:
            print("   -", f_)
        return 1
    print("[SMOKE PASS] all paper numeric anchors reproduced consistently.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
