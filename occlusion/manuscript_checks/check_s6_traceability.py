# -*- coding: utf-8 -*-
"""S6 numerical traceability table ⇄ frozen artifact consistency check (CPU, seconds; read-only, zero GPU).

This script is the **executable acceptance evidence** for ``manuscript_checks/S6_numeric_traceability.md``:

* Default: reads the traceability table row by row, re-fetches and re-formats each value from
  ``reference_results/`` using the JSON path given in the table, and asserts **character-by-character
  agreement** with the "value" column of the table;
  it also checks that the table's metric set is **exactly identical** to the in-script ``SPEC``
  (no missing row, no extra row, no duplicate).
* ``--write``: regenerates the traceability table from the frozen artifacts according to ``SPEC``
  (which validates the two against each other).

It also contains a set of **cross-artifact consistency invariants** (the drift of cross-process /
cross-session reruns and the stability of the decisions);
these invariants do not depend on any manuscript text.

⚠ This script only involves **numbers and value paths**; it contains no manuscript body sentence.
   Tools for checking the manuscript text are author-side and are not released with this repository.

Run::

    python occlusion/manuscript_checks/check_s6_traceability.py
    python occlusion/manuscript_checks/check_s6_traceability.py --write
"""

# Spec source: the CASES table of 27_occlusion_experiment/step4g_draft_assert.py (value paths, display modes and decimal
# places are carried over item by item); the adaptation is limited to the "environment bootstrap layer": the dependency
# on author-side manuscript text is dropped and replaced by a two-way check against the public traceability table.

import argparse
import json
import math
import sys
from pathlib import Path

OCC_DIR = Path(__file__).resolve().parent.parent      # …/occlusion
if str(OCC_DIR) not in sys.path:
    sys.path.insert(0, str(OCC_DIR))

from occlusion_config import MANUSCRIPT_DIR, REF_DIR, setup_stdout  # noqa: E402

setup_stdout()

TABLE = MANUSCRIPT_DIR / "S6_numeric_traceability.md"

J = {
    "minimal_pkg_stats.json": json.loads(
        (REF_DIR / "minimal_pkg_stats.json").read_text(encoding="utf-8")),
    "patch_stats.json": json.loads(
        (REF_DIR / "patch_stats.json").read_text(encoding="utf-8")),
    "perclass_stats.json": json.loads(
        (REF_DIR / "perclass_stats.json").read_text(encoding="utf-8")),
    "rerun_diag.json": json.loads(
        (REF_DIR / "rerun_diag.json").read_text(encoding="utf-8")),
    "rerun_verify.json": json.loads(
        (REF_DIR / "rerun_verify.json").read_text(encoding="utf-8")),
    "repeat_check.json": json.loads(
        (REF_DIR / "repeat_check.json").read_text(encoding="utf-8")),
}

MINUS = "\u2212"
SUP = {"0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3", "4": "\u2074",
       "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078", "9": "\u2079",
       "-": "\u207b", "\u2212": "\u207b"}


def get(fname, path):
    """Fetch a value from an artifact by ``a/b@key=val/c``; ``@`` is a list selector."""
    cur = J[fname]
    for seg in path.split("/"):
        if "@" in seg:
            name, kv = seg.split("@", 1)
            k, v = kv.split("=", 1)
            hit = [e for e in cur[name] if str(e.get(k)) == v]
            if not hit:
                raise KeyError("list selector matched nothing: %s" % seg)
            cur = hit[0]
        else:
            cur = cur[seg]
    return cur


def fmt(v, mode, d):
    """Render a number as the display string used in the paper (minus sign U+2212, Unicode superscripts, no thousands separators)."""
    if v is None:
        return "None"
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    if mode == "int":
        return "%d" % round(v)
    if mode == "intc":
        return "{:,}".format(round(v))
    if mode == "f":
        return ("%.*f" % (d, v)).replace("-", MINUS)
    if mode == "fs":
        s = "%.*f" % (d, v)
        if not s.startswith("-"):
            s = "+" + s
        return s.replace("-", MINUS)
    if mode == "pct":
        return ("%.*f" % (d, v * 100)).replace("-", MINUS) + "%"
    if mode == "sci":
        if v == 0:
            return "0"
        e = int(math.floor(math.log10(abs(v))))
        m = v / (10.0 ** e)
        ms = "%.*f" % (d, m)
        if abs(float(ms)) >= 10.0:          # carry protection
            e += 1
            m = v / (10.0 ** e)
            ms = "%.*f" % (d, m)
        return "%s\u00d710%s" % (ms.replace("-", MINUS),
                                 "".join(SUP[c] for c in str(e)))
    raise ValueError(mode)


# ==================== traceability table spec ====================
# (metric, source artifact file, JSON value path, display mode, decimal places)
# Modes: int integer | intc thousands-separated integer (only in raw-count contexts) | f fixed point | fs signed fixed point
#        pct percentage | sci scientific notation (×10^n, Unicode superscript)

SPEC = [
    ("1. Minimal necessary package (E1 entropy / E2 transplant / E3 coupling / E3b ceiling)", [
        ("Non-saturated sample count", "minimal_pkg_stats.json", "scope/n_nonsat", "int", 0),
        ("Saturated sample count", "minimal_pkg_stats.json", "scope/n_sat", "int", 0),
        ("Axis-coordinate baseline CN", "minimal_pkg_stats.json", "E2c_axis_baseline/CN", "f", 3),
        ("Axis-coordinate baseline MCI", "minimal_pkg_stats.json", "E2c_axis_baseline/MCI", "f", 3),
        ("Axis-coordinate baseline AD", "minimal_pkg_stats.json", "E2c_axis_baseline/AD", "f", 3),
        ("Joint statistic F3", "minimal_pkg_stats.json", "E2c_joint/stat_f3", "fs", 3),
        ("Joint statistic F3 p", "minimal_pkg_stats.json", "E2c_joint/p_f3", "sci", 1),
        ("F1 control statistic", "minimal_pkg_stats.json", "E2c_joint/stat_f1", "fs", 3),
        ("F1 control p", "minimal_pkg_stats.json", "E2c_joint/p_f1", "sci", 1),
        ("Permutation test p", "minimal_pkg_stats.json", "E2c_joint/perm_p", "f", 4),
        ("Permutation null mean", "minimal_pkg_stats.json", "E2c_joint/perm_mean", "fs", 3),
        ("Permutation null SD", "minimal_pkg_stats.json", "E2c_joint/perm_sd", "f", 3),
        ("Number of permutations", "minimal_pkg_stats.json", "E2c_joint/n_perm", "int", 0),
        ("MCI recipient between-group p", "minimal_pkg_stats.json", "E2b_donor_contrast/MCI/p", "f", 3),
        ("Cell CN←AD n", "minimal_pkg_stats.json", "E2c_cells/rCN_dAD/n", "int", 0),
        ("Cell MCI←CN n", "minimal_pkg_stats.json", "E2c_cells/rMCI_dCN/n", "int", 0),
        ("Entropy non-saturated all ΔH", "minimal_pkg_stats.json", "E1_entropy/MTL_d100_F1const/all", "fs", 4),
        ("Entropy flipped subset ΔH", "minimal_pkg_stats.json", "E1_entropy/MTL_d100_F1const/flip", "f", 3),
        ("Entropy correct→incorrect ΔH", "minimal_pkg_stats.json", "E1_entropy/MTL_d100_F1const/brk", "f", 3),
        ("Number of flipped samples", "minimal_pkg_stats.json", "E1_entropy/MTL_d100_F1const/flip_n", "int", 0),
        ("Entropy Wilcoxon p", "minimal_pkg_stats.json", "E1_entropy/_MAIN_nonsat/wilcoxon_p", "f", 3),
        ("Entropy n decreased", "minimal_pkg_stats.json", "E1_entropy/_MAIN_nonsat/n_down", "int", 0),
        ("Entropy n increased", "minimal_pkg_stats.json", "E1_entropy/_MAIN_nonsat/n_up", "int", 0),
        ("Coupling CN ρ", "minimal_pkg_stats.json", "E3_coupling/CN/rho", "fs", 3),
        ("Coupling CN p", "minimal_pkg_stats.json", "E3_coupling/CN/p", "f", 3),
        ("Coupling MCI ρ", "minimal_pkg_stats.json", "E3_coupling/MCI/rho", "fs", 3),
        ("Coupling MCI p", "minimal_pkg_stats.json", "E3_coupling/MCI/p", "f", 3),
        ("Coupling AD ρ", "minimal_pkg_stats.json", "E3_coupling/AD/rho", "fs", 3),
        ("Coupling AD p", "minimal_pkg_stats.json", "E3_coupling/AD/p", "f", 3),
        ("MCI power threshold", "minimal_pkg_stats.json", "E3_coupling/MCI/min_detectable_r", "f", 2),
        ("MCI redistribution ΔCN", "minimal_pkg_stats.json", "E3_coupling/MCI_redistribution/dCN", "fs", 3),
        ("MCI redistribution p(ΔCN)", "minimal_pkg_stats.json", "E3_coupling/MCI_redistribution/p_dCN", "f", 4),
        ("MCI redistribution ΔAD", "minimal_pkg_stats.json", "E3_coupling/MCI_redistribution/dAD", "fs", 3),
        ("MCI redistribution p(ΔAD)", "minimal_pkg_stats.json", "E3_coupling/MCI_redistribution/p_dAD", "sci", 2),
        ("AD ceiling ρ(Δp,p0)", "minimal_pkg_stats.json", "E3b_ceiling/AD/rho_dp_vs_p0", "f", 3),
        ("AD partial correlation", "minimal_pkg_stats.json", "E3b_ceiling/AD/partial_rho", "f", 3),
        ("AD partial correlation p", "minimal_pkg_stats.json", "E3b_ceiling/AD/partial_p", "f", 3),
        ("CN recipient F3−F1 paired p", "minimal_pkg_stats.json", "E2_f3/receptor_CN/F3_minus_F1_p", "f", 4),
    ]),
    ("2. Main effect panel and multiple comparisons (P1–P6)", [
        ("Main definition n", "patch_stats.json", "P1_effect_panel/nonsat all MTL_d100/n", "int", 0),
        ("Main definition r_rb", "patch_stats.json", "P1_effect_panel/nonsat all MTL_d100/r_rank_biserial", "fs", 3),
        ("Main definition p", "patch_stats.json", "P1_effect_panel/nonsat all MTL_d100/p_value", "f", 4),
        ("Main definition median Δp", "patch_stats.json", "P1_effect_panel/nonsat all MTL_d100/median_delta", "sci", 1),
        ("Main definition mean Δp", "patch_stats.json", "P1_effect_panel/nonsat all MTL_d100/mean_delta", "f", 4),
        ("Main definition n positive", "patch_stats.json", "P1_effect_panel/nonsat all MTL_d100/n_pos", "int", 0),
        ("Main definition n negative", "patch_stats.json", "P1_effect_panel/nonsat all MTL_d100/n_neg", "int", 0),
        ("Main definition Cohen d_z", "patch_stats.json", "P1_effect_panel/nonsat all MTL_d100/cohen_dz", "fs", 3),
        ("Main definition Holm p (10-condition family)", "patch_stats.json", "P2_holm_vs0/family_all@label=MTL_d100_F1const/p_holm", "f", 3),
        ("F2 r_rb", "patch_stats.json", "P2_holm_vs0/per_condition/MTL_d100_F2noise/r_rank_biserial", "fs", 3),
        ("F2 p", "patch_stats.json", "P2_holm_vs0/per_condition/MTL_d100_F2noise/p_value", "f", 3),
        ("F3 r_rb", "patch_stats.json", "P2_holm_vs0/per_condition/MTL_d100_F3transplant/r_rank_biserial", "fs", 3),
        ("F3 p", "patch_stats.json", "P2_holm_vs0/per_condition/MTL_d100_F3transplant/p_value", "f", 4),
        ("d80 Holm p", "patch_stats.json", "P2_holm_vs0/family_all@label=MTL_d80_F1const/p_holm", "f", 3),
        ("C3 (hippocampus) Holm p", "patch_stats.json", "P2_holm_vs0/family_all@label=C3_hipp_F1const/p_holm", "f", 3),
        ("F3 Holm p", "patch_stats.json", "P2_holm_vs0/family_all@label=MTL_d100_F3transplant/p_holm", "f", 3),
        ("MTL vs C1 p", "patch_stats.json", "P3_direct_pairs/per_pair@name=MTL vs C1 non-MTL cortex/p_value", "f", 3),
        ("MTL vs C1 r_rb", "patch_stats.json", "P3_direct_pairs/per_pair@name=MTL vs C1 non-MTL cortex/r_rank_biserial", "fs", 3),
        ("AD probability flow to CN share", "patch_stats.json", "P4_prob_destination/AD_nonsat_share_CN_vs_MCI", "pct", 1),
        ("Dose-response ρ", "patch_stats.json", "P6_dose_exact/medianΔ/rho", "f", 2),
        ("Dose-response exact permutation p", "patch_stats.json", "P6_dose_exact/medianΔ/p_exact_two_sided", "f", 3),
        ("Dose-response asymptotic p", "patch_stats.json", "P6_dose_exact/medianΔ/p_asymptotic_t", "f", 3),
        ("Adjacent-level first-step p", "patch_stats.json", "P6_dose_exact/adjacent_pairs@name=d40_+12706 - d25_7941/p_value", "f", 3),
        ("Saturation threshold", "patch_stats.json", "scope/sat_thr", "f", 6),
    ]),
    ("3. Per-class and AD specificity", [
        ("KW H", "perclass_stats.json", "per_class_delta/_KW/H", "f", 2),
        ("KW p", "perclass_stats.json", "per_class_delta/_KW/p", "sci", 1),
        ("AD recall baseline", "perclass_stats.json", "per_class_flip/AD/recall_base", "f", 4),
        ("AD recall after occlusion", "perclass_stats.json", "per_class_flip/AD/recall_occ", "f", 4),
        ("AD correct→incorrect", "perclass_stats.json", "per_class_flip/AD/n_break", "int", 0),
        ("AD incorrect→correct", "perclass_stats.json", "per_class_flip/AD/n_fix", "int", 0),
        ("AD McNemar p", "perclass_stats.json", "ad_specificity/mcnemar_p", "f", 2),
        ("All incorrect→correct", "perclass_stats.json", "ad_specificity/all_fix", "int", 0),
        ("All correct→incorrect", "perclass_stats.json", "ad_specificity/all_break", "int", 0),
        ("CN stratum p", "perclass_stats.json", "per_class_delta/CN/p", "f", 2),
        ("MCI stratum p", "perclass_stats.json", "per_class_delta/MCI/p", "f", 2),
        ("AD stratum p", "perclass_stats.json", "per_class_delta/AD/p", "sci", 2),
    ]),
    ("4. Cross-session, cross-process reproducibility and recipient×donor distribution", [
        ("Cross-session p0 drift", "rerun_diag.json", "p0_vs_baseline_new", "sci", 3),
        ("Cross-session p_after drift", "rerun_verify.json", "p_after/max_abs_diff", "sci", 2),
        ("Cross-session mask agreement", "rerun_verify.json", "mask_vox/max_abs_diff", "f", 1),
        ("Cell-level argmax mismatch", "rerun_diag.json", "argmax/cell_p0", "int", 0),
        ("Δp sign agreement rate", "rerun_diag.json", "d_true/sign_agreement", "f", 1),
        ("Spearman(Δold,Δnew)", "rerun_diag.json", "d_true/spearman", "f", 6),
        ("Probability count of two full rescans", "repeat_check.json", "p0/n_total", "intc", 0),
        ("Recipient CN←MCI donor count", "rerun_verify.json", "donor_distribution/CN<-MCI", "int", 0),
        ("Recipient CN←AD donor count", "rerun_verify.json", "donor_distribution/CN<-AD", "int", 0),
        ("Recipient MCI←CN donor count", "rerun_verify.json", "donor_distribution/MCI<-CN", "int", 0),
        ("Recipient MCI←AD donor count", "rerun_verify.json", "donor_distribution/MCI<-AD", "int", 0),
        ("Recipient AD←CN donor count", "rerun_verify.json", "donor_distribution/AD<-CN", "int", 0),
        ("Recipient AD←MCI donor count", "rerun_verify.json", "donor_distribution/AD<-MCI", "int", 0),
    ]),
]

FLAT = [(sec, r) for sec, rows in SPEC for r in rows]


# ==================== traceability table generation ====================

def write_table():
    lines = [
        "# Occlusion causal analysis · numerical traceability table (paper §2.5 / §3.3 / §4.3, supplementary S6)",
        "",
        "> This table maps every number reported in the occlusion causal analysis to the exact value path",
        "> **within the frozen artifacts** in `occlusion/reference_results/` (`@key=val` is a list selector).",
        "> It is asserted row by row by `check_s6_traceability.py` and called by stage 4 of `occlusion/verify_occlusion.py`;",
        "> the table is generated by `python occlusion/manuscript_checks/check_s6_traceability.py --write`.",
        ">",
        "> ⚠ This table contains **only numbers and provenance**; it contains no manuscript body sentence. Tools for checking the manuscript text are author-side and are not released with this repository.",
        "> ⚠ The minus sign uses U+2212, superscripts use Unicode, and counts never use a thousands separator (matching the manuscript typesetting).",
        "",
    ]
    n = 0
    for sec, rows in SPEC:
        lines.append("## %s" % sec)
        lines.append("")
        lines.append("| # | metric | source artifact | JSON path | value |")
        lines.append("|---:|---|---|---|---:|")
        for desc, fname, path, mode, d in rows:
            n += 1
            lines.append("| %d | %s | %s | `%s` | %s |"
                         % (n, desc, fname, path, fmt(get(fname, path), mode, d)))
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**%d** items in total." % n)
    lines.append("")
    TABLE.write_text("\n".join(lines), encoding="utf-8")
    print("generated %s (%d items)" % (TABLE, n))


# ==================== traceability table check ====================

def parse_table():
    """Parse the md table into [(metric, source artifact, value path, value), ...]."""
    rows = []
    for ln in TABLE.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != 5:
            continue
        if cells[0] in ("#",) or set(cells[0]) <= set("-: "):   # header and separator rows
            continue
        rows.append((cells[1], cells[2], cells[3].strip("`"), cells[4]))
    return rows


def check_table():
    print("=" * 92)
    print("1. Traceability table ⇄ frozen artifacts, row-by-row agreement")
    print("=" * 92)

    fails = []
    if not TABLE.exists():
        return ["missing %s (it can be generated with --write)" % TABLE.name]
    rows = parse_table()

    spec_keys = [r[0] for _s, r in FLAT]
    got_keys = [r[0] for r in rows]

    dup = sorted({k for k in got_keys if got_keys.count(k) > 1})
    if dup:
        fails.append("duplicate metrics in the table: %s" % ", ".join(dup))
    missing = [k for k in spec_keys if k not in got_keys]
    extra = [k for k in got_keys if k not in spec_keys]
    if missing:
        fails.append("the table is missing %d SPEC metrics: %s" % (len(missing), ", ".join(missing[:6])))
    if extra:
        fails.append("the table has %d metrics outside SPEC: %s" % (len(extra), ", ".join(extra[:6])))
    if got_keys != spec_keys:
        order = "the row order differs from SPEC" if (not missing and not extra) else "the sets differ"
        print("[WARN] table rows: %s (does not affect the numerical check)" % order)

    spec_by_key = {r[0]: r for _s, r in FLAT}
    bad = []
    for desc, fname, path, shown in rows:
        if desc not in spec_by_key:
            continue
        _d, s_fname, _p, mode, d = spec_by_key[desc]
        if fname != s_fname:
            bad.append((desc, "source artifact column %s ≠ spec %s" % (fname, s_fname)))
            continue
        try:
            exp = fmt(get(fname, path), mode, d)
        except Exception as exc:
            bad.append((desc, "fetch failed %s:%s (%r)" % (fname, path, exc)))
            continue
        if exp != shown:
            bad.append((desc, "value %s ≠ computed from spec %s" % (shown, exp)))
    for b in bad:
        print("       x %s | %s" % b)
    n_ok = len(rows) - len(bad)
    print("[%s] row-by-row agreement %d / %d" % ("OK" if not bad else "FAIL", n_ok, len(rows)))
    if bad:
        fails.append("traceability table: %d rows disagree with the artifacts" % len(bad))
    return fails


# ==================== cross-artifact consistency invariants ====================

def check_invariants():
    print()
    print("=" * 92)
    print("2. Cross-artifact consistency invariants (cross-process / cross-session; independent of any manuscript text)")
    print("=" * 92)

    d = J["rerun_diag.json"]
    v = J["rerun_verify.json"]
    r = J["repeat_check.json"]
    p = J["patch_stats.json"]

    checks = [
        ("Two full rescans bitwise identical (p0 / p_after / mask)",
         (r["p0"]["max_abs_diff"], r["pa"]["max_abs_diff"], r["mv"]["max_abs_diff"]),
         (0.0, 0.0, 0.0)),
        ("Each rescan saved 4320 probabilities",
         (r["p0"]["n_total"], r["pa"]["n_total"]), (4320, 4320)),
        ("Donor identity per sample identical across the two saves",
         r["donor"]["n_mismatch"], 0),
        ("Cross-session p_after max absolute drift",
         v["p_after"]["max_abs_diff"], 0.0013266801834106445),
        ("Cross-session vs baseline_p0 max absolute drift",
         v["p0"]["max_abs_diff"], 0.0009449124336242676),
        ("Cross-session argmax mismatch count (p_after / p0)",
         (v["p_after"]["n_argmax_mismatch"], v["p0"]["n_argmax_mismatch"]), (0, 0)),
        ("Cell-level argmax mismatch count (p0 / p_after)",
         (d["argmax"]["cell_p0"], d["argmax"]["cell_p_after"]), (0, 0)),
        ("Sample-level argmax mismatch count (p0 / p_after)",
         (d["argmax"]["sample_p0"], d["argmax"]["sample_p_after"]), (0, 0)),
        ("Total number of cells", d["argmax"]["total_cells"], 1440),
        ("Δp sign agreement rate", d["d_true"]["sign_agreement"], 1.0),
        ("Saturated stratification (old / new)",
         (d["saturation"]["n_sat_old"], d["saturation"]["n_sat_new"]), (64, 64)),
        ("Non-saturated stratification (old / new)",
         (d["saturation"]["n_nonsat_old"], d["saturation"]["n_nonsat_new"]), (80, 80)),
        ("Number of stratification flips", d["saturation"]["n_flip"], 0),
    ]

    bad = []
    for name, got, exp in checks:
        ok = got == exp
        print("    %-44s measured=%-32s %s" % (name, got, "OK" if ok else "x expected " + str(exp)))
        if not ok:
            bad.append((name, got, exp))

    # C1 (volume-matched non-MTL cortex): no probability flows into CN after occlusion —— a directionality
    # landmark, which must be NaN
    c1 = p["P4_prob_destination"]["flow_by_condition"]["C1_cort_23_F1const"]["share_to_CN"]
    c1_ok = isinstance(c1, float) and math.isnan(c1)
    print("    %-44s measured=%-32s %s" % ("C1 non-MTL cortex share flowing to CN", c1, "OK" if c1_ok else "x expected NaN"))
    if not c1_ok:
        bad.append(("C1 non-MTL cortex share flowing to CN", c1, "NaN"))

    print("[%s] invariants %d / %d" % ("OK" if not bad else "FAIL",
                                       len(checks) + 1 - len(bad), len(checks) + 1))
    return ["invariants not satisfied: %s" % (bad,)] if bad else []


# ==================== main ====================

def main():
    ap = argparse.ArgumentParser(description="S6 numerical traceability table ⇄ frozen artifacts consistency check")
    ap.add_argument("--write", action="store_true", help="regenerate the traceability table from the frozen artifacts according to SPEC")
    args = ap.parse_args()

    if args.write:
        write_table()
        return 0

    fails = check_table()
    fails += check_invariants()

    print()
    print("=" * 92)
    if fails:
        print("overall: FAIL —— traceability table row-by-row agreement %s | invariants %s"
              % ("passed" if len(fails) < 2 else "see above",
                 "passed" if not fails else "see above"))
        for f in fails:
            print("  x %s" % f)
        print("=" * 92)
        return 1
    print("overall: PASS —— traceability table %d / %d rows agree, all cross-artifact invariants pass" % (len(FLAT), len(FLAT)))
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
