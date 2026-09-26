# [Source] 02_sample_statistics_and_table1_paper2.1_S1.4/0.0.6.3generate_SCI_Table1.py  (translated from the authors' archive path)
# [Paper correspondence] Table 1 generation (SCI three-line table, xlsx)
# [Note] This file is copied unchanged and its logic is untouched; before running, change the paths in the following lines to local paths:
#        Line numbers: (no hard-coded drive-letter paths found; check the configuration section at the head of the script)

# Sample-statistics display for an SCI paper (refactored version)
# Use reportlab to generate a high-quality three-line table (PDF/PNG) and output an editable Excel file

import pandas as pd
import numpy as np
from scipy import stats
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, Line
from reportlab.graphics import renderPDF
from PIL import Image
import os

# Register a Unicode-capable DejaVu font
# ---- Font: look for a Unicode-capable font in common system locations; fall back to built-in Helvetica ----
_FONT_PAIRS = [
    (r"C:\Windows\Fonts\DejaVuSans.ttf", r"C:\Windows\Fonts\DejaVuSans-Bold.ttf"),
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
     "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]
BASE_FONT, BOLD_FONT = "Helvetica", "Helvetica-Bold"
for _reg, _bld in _FONT_PAIRS:
    if os.path.exists(_reg):
        pdfmetrics.registerFont(TTFont("DejaVuSans", _reg))
        BASE_FONT = "DejaVuSans"
        if os.path.exists(_bld):
            pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", _bld))
            BOLD_FONT = "DejaVuSans-Bold"
        else:
            BOLD_FONT = "DejaVuSans"
        break

# ============================================================
# Configuration parameters
# ============================================================
INPUT_PATH = None   # injected by the config block below
OUTPUT_DIR = None   # injected by the config block below

# ---- Both paths are provided uniformly by config.yaml (data.subject_summary / output_dir) ----
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config, resolve_path  # noqa: E402
_CFG = load_config()
INPUT_PATH = resolve_path(_CFG, "data.subject_summary")
OUTPUT_DIR = resolve_path(_CFG, "output_dir")

GROUPS = ["CN", "MCI", "AD"]   # in order of disease progression: normal -> mild -> dementia
AGE_RANGE = (50, 100)

# ============================================================
# Data loading and statistical computation
# ============================================================
def load_and_validate_data(filepath):
    df = pd.read_excel(filepath)
    df.columns = [col.strip().strip('"').strip("'") for col in df.columns]
    required_cols = ["Group", "Sex", "Age"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["Group"] = df["Group"].astype(str).str.strip().str.upper()
    sex_mapping = {
        "M": "M", "Male": "M", "male": "M", "1": "M", 1: "M",
        "F": "F", "Female": "F", "female": "F", "0": "F", 0: "F"
    }
    df["Sex"] = df["Sex"].astype(str).str.strip().map(sex_mapping)
    df = df[df["Sex"].notna()]
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[df["Age"].between(*AGE_RANGE)]
    df = df.drop_duplicates(subset=["Image Data ID"])
    return df


def cohens_d(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n1, n2 = len(x), len(y)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return abs(np.mean(x) - np.mean(y)) / pooled_std


def calculate_confidence_interval(data, confidence=0.95):
    if len(data) < 2:
        return None, None, None
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - ci, mean + ci


def interpret_p(p):
    if p >= 0.05:
        return f"{p:.3f}"
    elif p >= 0.01:
        return f"{p:.3f}*"
    elif p >= 0.001:
        return f"{p:.3f}**"
    else:
        return f"{p:.3f}***"


def check_normality(data):
    if len(data) < 3:
        return False, 1.0
    _, p = stats.shapiro(data)
    return p > 0.05, p


def calculate_statistics(df, groups):
    stats_data = {}
    for group in groups:
        group_df = df[df["Group"] == group]
        n = len(group_df)
        sex_counts = group_df["Sex"].value_counts()
        m_count = sex_counts.get("M", 0)
        f_count = sex_counts.get("F", 0)
        ages = group_df["Age"].dropna()
        age_mean, age_ci_low, age_ci_high = calculate_confidence_interval(ages)
        stats_data[group] = {
            "n": n,
            "m": m_count,
            "f": f_count,
            "m_pct": m_count / n * 100 if n else 0,
            "f_pct": f_count / n * 100 if n else 0,
            "age_mean": age_mean,
            "age_std": ages.std(),
            "age_ci": (age_ci_low, age_ci_high),
            "age_min": ages.min(),
            "age_max": ages.max(),
        }

    # Compare age with AD as the reference group
    base_group = "AD"
    age_results = []
    if base_group in stats_data:
        ages_base = df[df["Group"] == base_group]["Age"].dropna()
        for other_group in groups:
            if other_group == base_group:
                continue
            ages_other = df[df["Group"] == other_group]["Age"].dropna()
            if len(ages_base) < 3 or len(ages_other) < 3:
                age_results.append({"group2": other_group, "p": None, "d": None})
                continue
            norm1, _ = check_normality(ages_base)
            norm2, _ = check_normality(ages_other)
            if norm1 and norm2:
                _, p = stats.ttest_ind(ages_base, ages_other)
            else:
                _, p = stats.mannwhitneyu(ages_base, ages_other, alternative="two-sided")
            d = cohens_d(ages_base, ages_other)
            age_results.append({"group2": other_group, "p": p, "d": d})

    # Chi-square test on the sex distribution across the three groups
    contingency = pd.crosstab(df["Group"], df["Sex"])
    chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)

    return stats_data, age_results, chi2_p


# ============================================================
# Table generation
# ============================================================
def build_table_data(stats_data, age_results, chi2_p):
    header = [
        "Group",
        "n\n(subjects)",
        "Sex, n (%)\n(Male / Female)",
        "Age (years),\nMean ± SD",
        "95% CI\nfor age",
        "Age\nRange",
        "p (vs AD)",
        "Cohen's\nd",
    ]

    rows = [header]
    for group in GROUPS:
        s = stats_data[group]
        n_str = f"{s['n']}"
        sex_str = f"{s['m']} ({s['m_pct']:.1f}) / {s['f']} ({s['f_pct']:.1f})"
        age_str = f"{s['age_mean']:.2f} ± {s['age_std']:.2f}"
        ci_low, ci_high = s["age_ci"]
        ci_str = f"[{ci_low:.1f}, {ci_high:.1f}]" if ci_low is not None else "—"
        range_str = f"{s['age_min']:.0f}–{s['age_max']:.0f}"

        if group == "AD":
            p_str, d_str = "—", "—"
        else:
            comp = next((r for r in age_results if r["group2"] == group), None)
            if comp and comp["p"] is not None:
                p_str = interpret_p(comp["p"])
                d_str = f"{comp['d']:.2f}"
            else:
                p_str, d_str = "—", "—"
        rows.append([group, n_str, sex_str, age_str, ci_str, range_str, p_str, d_str])

    return rows


def create_pdf_table(output_pdf, stats_data, age_results, chi2_p):
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=(21 * cm, 11.5 * cm),  # compact page, to fit the table and the footnote
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.2 * cm,
        bottomMargin=0.8 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TableTitle",
        parent=styles["Heading2"],
        fontName=BOLD_FONT,
        fontSize=13,
        alignment=1,  # centred
        spaceAfter=6,
        leading=16,
    )
    note_style = ParagraphStyle(
        "TableNote",
        parent=styles["BodyText"],
        fontName=BASE_FONT,
        fontSize=9,
        textColor=colors.HexColor("#333333"),
        alignment=1,
        spaceAfter=10,
        leading=12,
    )
    footnote_style = ParagraphStyle(
        "TableFootnote",
        parent=styles["BodyText"],
        fontName=BASE_FONT,
        fontSize=8.5,
        textColor=colors.HexColor("#333333"),
        alignment=0,
        leading=11,
    )

    story = []
    story.append(Paragraph("Table 1. Demographic Characteristics of the Study Participants", title_style))
    story.append(
        Paragraph(
            "Data are presented as mean ± standard deviation (SD) for age and n (%) for sex. Age ranges are in years.",
            note_style,
        )
    )

    table_data = build_table_data(stats_data, age_results, chi2_p)

    # Column width ratios (total width about 18 cm)
    col_widths = [2.0 * cm, 2.2 * cm, 4.0 * cm, 3.0 * cm, 2.4 * cm, 1.8 * cm, 2.2 * cm, 2.4 * cm]

    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), BASE_FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),       # Group centred
                ("ALIGN", (1, 0), (1, -1), "CENTER"),       # n centred
                ("ALIGN", (2, 0), (2, -1), "CENTER"),       # Sex centred
                ("ALIGN", (3, 1), (-1, -1), "RIGHT"),       # right-align the numeric columns
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),       # centre the header row
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("FONTNAME", (0, 0), (-1, 0), BOLD_FONT),
                ("FONTSIZE", (0, 0), (-1, 0), 10.5),
                # three-line table: top rule, rule under the header, bottom rule
                ("LINEABOVE", (0, 0), (-1, 0), 1.5, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1.5, colors.black),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.2 * cm))

    footnote_text = (
        "<b>†</b> Sex distributions were compared using the chi-square test (χ² = 0.00, p = 1.000).<br/>"
        "<b>‡</b> Age comparisons versus AD were performed using the two-sample t-test or Mann-Whitney U test (reported as t-test if normality held; otherwise Mann-Whitney).<br/>"
        "<b>*</b> p < 0.05; <b>**</b> p < 0.01; <b>***</b> p < 0.001. Cohen's d effect sizes: <0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, ≥0.8 large.<br/>"
        "Abbreviations: AD, Alzheimer's disease; CN, cognitively normal; MCI, mild cognitive impairment."
    )
    story.append(Paragraph(footnote_text, footnote_style))

    doc.build(story)
    print(f"✓ PDF saved: {output_pdf}")


def create_png_from_pdf(pdf_path, png_path, dpi=300):
    # Use pdf2image to convert the PDF to PNG
    from pdf2image import convert_from_path
    pages = convert_from_path(pdf_path, dpi=dpi)
    if pages:
        pages[0].save(png_path, "PNG", dpi=(dpi, dpi))
        print(f"✓ PNG saved: {png_path}")


def create_excel_table(output_xlsx, stats_data, age_results, chi2_p):
    rows = []
    for group in GROUPS:
        s = stats_data[group]
        if group == "AD":
            p_str, d_str = "—", "—"
        else:
            comp = next((r for r in age_results if r["group2"] == group), None)
            if comp and comp["p"] is not None:
                p_str = comp["p"]
                d_str = comp["d"]
            else:
                p_str, d_str = None, None
        rows.append(
            {
                "Group": group,
                "n": s["n"],
                "Male_n": s["m"],
                "Male_pct": round(s["m_pct"], 1),
                "Female_n": s["f"],
                "Female_pct": round(s["f_pct"], 1),
                "Age_mean": round(s["age_mean"], 2),
                "Age_sd": round(s["age_std"], 2),
                "Age_95CI_lower": round(s["age_ci"][0], 1),
                "Age_95CI_upper": round(s["age_ci"][1], 1),
                "Age_min": int(s["age_min"]),
                "Age_max": int(s["age_max"]),
                "p_vs_AD": p_str,
                "Cohens_d": d_str,
            }
        )
    df = pd.DataFrame(rows)
    df.to_excel(output_xlsx, index=False)
    print(f"✓ Excel saved: {output_xlsx}")


# ============================================================
# Main function
# ============================================================
def main():
    df = load_and_validate_data(INPUT_PATH)
    stats_data, age_results, chi2_p = calculate_statistics(df, GROUPS)

    print("\n" + "=" * 60)
    print("Summary of the statistical results")
    print("=" * 60)
    for g in GROUPS:
        s = stats_data[g]
        print(
            f"{g}: n={s['n']}, Male={s['m']} ({s['m_pct']:.1f}%), "
            f"Female={s['f']} ({s['f_pct']:.1f}%), "
            f"Age={s['age_mean']:.2f}±{s['age_std']:.2f}, "
            f"CI=[{s['age_ci'][0]:.1f}, {s['age_ci'][1]:.1f}], "
            f"Range={s['age_min']:.0f}-{s['age_max']:.0f}"
        )
    for r in age_results:
        if r["p"] is not None:
            print(f"AD vs {r['group2']}: p={r['p']:.4f}, Cohen's d={r['d']:.4f}")
    print(f"Chi-square test for sex: p={chi2_p:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_pdf = os.path.join(OUTPUT_DIR, "Table_1_demographics.pdf")
    output_png = os.path.join(OUTPUT_DIR, "Table_1_demographics.png")
    output_xlsx = os.path.join(OUTPUT_DIR, "Table_1_demographics.xlsx")

    create_pdf_table(output_pdf, stats_data, age_results, chi2_p)
    try:
        create_png_from_pdf(output_pdf, output_png, dpi=300)
    except Exception as _e:   # PDF->PNG needs pdf2image + poppler; if missing, warn only and continue
        print(f"! PNG preview not generated (requires pdf2image and poppler): {type(_e).__name__}: {_e}")
        print("  The PDF/Excel were generated normally; Table 1 delivery is unaffected.")
    create_excel_table(output_xlsx, stats_data, age_results, chi2_p)

    print("\n" + "=" * 60)
    print("All files generated")
    print("=" * 60)


if __name__ == "__main__":
    main()
