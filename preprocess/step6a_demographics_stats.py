# [Source] 02_sample_statistics_and_table1_paper2.1_S1.4/0.0.6.1sample_info_statistics_added_effect_size_calculation.py  (translated from the authors' archive path)
# [Paper correspondence] 2.1/Table 1 demographics statistics + effect sizes (Kruskal-Wallis/Mann-Whitney/Cohen's d)
# [Note] This file is copied unchanged and its logic is untouched; before running, change the paths in the following lines to local paths:
#        Line numbers: [84, 430, 438]  (line numbers refer to the original Chinese version)

# Version 0.0.6.1 adds effect-size computation and result output, providing more comprehensive statistical-analysis support.
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')   # headless: plt.show() becomes a no-op and cannot block
import matplotlib.pyplot as plt
import seaborn as sns


# Paths come from config.yaml (data.subject_summary / output_dir); no machine-specific paths here
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config, resolve_path  # noqa: E402
_CFG = load_config()
_SUMMARY_XLSX = resolve_path(_CFG, "data.subject_summary")
_OUT_DIR = resolve_path(_CFG, "output_dir")
os.makedirs(_OUT_DIR, exist_ok=True)

# Set a Chinese font (prefer a system Chinese font)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# If problems persist after the settings above, you can force the default matplotlib font
plt.rcParams['font.family'] = 'sans-serif'

# ============================================
# Effect-size computation functions
# ============================================
def cohens_d(x, y):
    """
    Compute the Cohen's d effect size (difference of the two group means divided by the pooled SD)
    Interpretation criteria:
      < 0.2: negligible
      0.2 - 0.5: small effect
      0.5 - 0.8: medium effect
      ≥ 0.8: large effect
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    n1, n2 = len(x), len(y)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    
    # Pooled standard deviation (with degrees-of-freedom correction)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = abs(np.mean(x) - np.mean(y)) / pooled_std
    return d


def rank_biserial_r(x, y):
    """
    Compute the rank-biserial correlation coefficient (effect size of the Mann-Whitney U test)
    Formula: r = 1 - (2 * U) / (n1 * n2)
    Interpretation criteria:
      < 0.1: negligible
      0.1 - 0.3: small effect
      0.3 - 0.5: medium effect
      ≥ 0.5: large effect
    """
    u_stat, _ = stats.mannwhitneyu(x, y, alternative='two-sided')
    n1, n2 = len(x), len(y)
    r = 1 - (2 * u_stat) / (n1 * n2)
    return abs(r)


def interpret_cohens_d(d):
    """Interpret the magnitude of Cohen's d"""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small effect"
    elif d < 0.8:
        return "medium effect"
    else:
        return "large effect"


def interpret_rank_biserial(r):
    """Interpret the rank-biserial correlation coefficient"""
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small effect"
    elif r < 0.5:
        return "medium effect"
    else:
        return "large effect"


# Read the "summary 2" file
df = pd.read_excel(_SUMMARY_XLSX)

print("="*80)
print("                    Summary 2 — sex and age distribution statistics report")
print("="*80)
print(f"Data generated at: {pd.Timestamp.now()}")
print(f"Total samples: {len(df)}")
print()

# ============================================
# I. Basic descriptive statistics
# ============================================
print("I. Basic descriptive statistics")
print("-"*80)

# 1.1 Sample size per group
print("\n1.1 Sample-size distribution:")
print(f"{'Group':<10} {'N':<10} {'Share':<10}")
for group in ['AD', 'CN', 'MCI']:
    n = len(df[df['Group'] == group])
    pct = n / len(df) * 100
    print(f"{group:<10} {n:<10} {pct:.1f}%")

# 1.2 Sex distribution
print("\n1.2 Sex distribution:")
print(f"\n{'Group':<10} {'Male':<10} {'Female':<10} {'M:F ratio':<15} {'chi-square p (vs AD)':<20}")

ad_sex = df[df['Group'] == 'AD']['Sex'].value_counts()

for group in ['AD', 'CN', 'MCI']:
    subset = df[df['Group'] == group]
    sex_counts = subset['Sex'].value_counts()
    
    male = sex_counts.get('M', 0)
    female = sex_counts.get('F', 0)
    ratio = f"{male}:{female}"
    
    # Chi-square test comparing with the AD group
    if group != 'AD':
        contingency = pd.DataFrame({
            'AD': [ad_sex.get('M', 0), ad_sex.get('F', 0)],
            group: [male, female]
        }, index=['M', 'F'])
        
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        p_str = f"p={p_value:.4f}"
        sig = " ✓" if p_value > 0.05 else " ⚠ significant difference!"
    else:
        p_str = "- (baseline)"
    
    print(f"{group:<10} {male:<10} {female:<10} {ratio:<15} {p_str:<20}")

# 1.3 Overall between-group sex test
print("\n1.3 Overall chi-square test on sex across the three groups:")
contingency_all = pd.DataFrame({
    'AD': [ad_sex.get('M', 0), ad_sex.get('F', 0)],
    'CN': [df[df['Group']=='CN']['Sex'].value_counts().get('M', 0), 
           df[df['Group']=='CN']['Sex'].value_counts().get('F', 0)],
    'MCI': [df[df['Group']=='MCI']['Sex'].value_counts().get('M', 0), 
            df[df['Group']=='MCI']['Sex'].value_counts().get('F', 0)]
}, index=['M', 'F'])
chi2_all, p_all, dof, expected = stats.chi2_contingency(contingency_all)
print(f"χ² = {chi2_all:.4f}, df = {dof}, p = {p_all:.4f}")
print(f"{'→ no significant difference in the sex distribution across the three groups ✓' if p_all > 0.05 else '→ significant difference in the sex distribution across the three groups ⚠'}")

# ============================================
# II. Age distribution statistics
# ============================================
print("\n" + "="*80)
print("II. Age distribution statistics")
print("-"*80)

# Ensure that Age is numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Extract the age data for each group
ad_ages = df[df['Group']=='AD']['Age'].dropna()
cn_ages = df[df['Group']=='CN']['Age'].dropna()
mci_ages = df[df['Group']=='MCI']['Age'].dropna()

print(f"\n{'Group':<10} {'Mean':<10} {'SD':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'Range':<15}")
print("-"*80)

age_stats = {}
for group in ['AD', 'CN', 'MCI']:
    ages = df[df['Group'] == group]['Age'].dropna()
    age_stats[group] = {
        'n': len(ages),
        'mean': ages.mean(),
        'std': ages.std(),
        'median': ages.median(),
        'min': ages.min(),
        'max': ages.max()
    }
    print(f"{group:<10} {ages.mean():<10.2f} {ages.std():<10.2f} {ages.median():<10.0f} "
          f"{ages.min():<10.0f} {ages.max():<10.0f} {f'{ages.min():.0f}-{ages.max():.0f}':<15}")

# 2.1 Normality test for the age distribution
print("\n2.1 Normality test for age (Shapiro-Wilk):")
for group in ['AD', 'CN', 'MCI']:
    ages = df[df['Group'] == group]['Age'].dropna()
    stat, p = stats.shapiro(ages)
    print(f"  {group}: W={stat:.4f}, p={p:.4f}", end="")
    print(" → normal distribution ✓" if p > 0.05 else " → non-normal distribution ⚠")

# 2.2 Homogeneity-of-variance test
print("\n2.2 Homogeneity-of-variance test (Levene):")
stat, p = stats.levene(ad_ages, cn_ages, mci_ages)
print(f"  Levene statistic={stat:.4f}, p={p:.4f}", end="")
print(" → variances are homogeneous ✓" if p > 0.05 else " → variances are not homogeneous ⚠")

# 2.3 ANOVA / Kruskal-Wallis
print("\n2.3 Overall test of age differences across the three groups:")
# First test normality and homogeneity of variance
normal_all = all(stats.shapiro(df[df['Group']==g]['Age'].dropna())[1] > 0.05 for g in ['AD', 'CN', 'MCI'])
homogeneous = stats.levene(ad_ages, cn_ages, mci_ages)[1] > 0.05

if normal_all and homogeneous:
    # Use ANOVA
    f_stat, p_anova = stats.f_oneway(ad_ages, cn_ages, mci_ages)
    print(f"  Used: one-way analysis of variance (ANOVA)")
    print(f"  F = {f_stat:.4f}, p = {p_anova:.4f}")
    overall_p = p_anova
else:
    # Use Kruskal-Wallis
    h_stat, p_kw = stats.kruskal(ad_ages, cn_ages, mci_ages)
    print(f"  Used: Kruskal-Wallis H test (non-parametric)")
    print(f"  H = {h_stat:.4f}, p = {p_kw:.4f}")
    overall_p = p_kw

if overall_p > 0.05:
    print("  → no significant age difference across the three groups ✓")
else:
    print("  → significant age difference across the three groups ⚠")

# 2.4 Pairwise comparisons (with effect sizes)
print("\n2.4 Pairwise comparisons (versus the AD group):")
ad_mean = age_stats['AD']['mean']

for group in ['CN', 'MCI']:
    group_ages = df[df['Group'] == group]['Age'].dropna()
    group_mean = age_stats[group]['mean']
    diff = group_mean - ad_mean  # keep the sign
    
    # Effect-size computation
    d = cohens_d(ad_ages, group_ages)
    r = rank_biserial_r(ad_ages, group_ages)
    
    # Choose the test method
    if normal_all and homogeneous:
        t_stat, p_val = stats.ttest_ind(ad_ages, group_ages)
        test_name = "t test"
        stat_str = f"t={t_stat:.4f}"
    else:
        u_stat, p_val = stats.mannwhitneyu(ad_ages, group_ages, alternative='two-sided')
        test_name = "Mann-Whitney U"
        stat_str = f"U={u_stat:.4f}"
    
    print(f"\n  AD vs {group}:")
    print(f"    Age difference: {diff:+.2f} years (AD: {ad_mean:.1f}, {group}: {group_mean:.1f})")
    print(f"    {test_name}: {stat_str}, p={p_val:.4f}", end="")
    
    if p_val > 0.05:
        print(" → no significant difference ✓")
    else:
        print(f" → significant difference ⚠ (p<0.05)")
    
    # Output the effect sizes
    print(f"    Cohen's d = {d:.4f} → {interpret_cohens_d(d)}")
    print(f"    Rank-Biserial r = {r:.4f} → {interpret_rank_biserial(r)}")
    
    # Overall assessment of the effect sizes
    if p_val > 0.05:
        print(f"    Assessment: no statistically significant difference, good matching ✓")
    else:
        if d < 0.2:
            print(f"    Assessment: statistically significant but the effect size is very small ({interpret_cohens_d(d)}), of limited practical relevance ✓")
        elif d < 0.5:
            print(f"    Assessment: statistically significant with a small effect; discuss in the paper ⚠")
        else:
            print(f"    Assessment: statistically significant with a non-negligible effect size; sensitivity analysis recommended ⚠⚠")

# 2.5 Age-matching quality assessment (with effect sizes)
print("\n2.5 Age-matching quality assessment:")
print("{:<12} {:<10} {:<12} {:<12} {:<10} {:<15}".format('Comparison', 'Mean diff', "Cohen's d", 'Effect interpretation', 'R-B r', 'Quality rating'))
print("-"*75)

for group in ['CN', 'MCI']:
    group_ages = df[df['Group'] == group]['Age'].dropna()
    diff = abs(age_stats[group]['mean'] - ad_mean)
    d = cohens_d(ad_ages, group_ages)
    r = rank_biserial_r(ad_ages, group_ages)
    
    # Overall quality rating
    if d < 0.2:
        quality = "Excellent ✓"
    elif d < 0.3:
        quality = "Good △"
    elif d < 0.5:
        quality = "Acceptable ⚠"
    else:
        quality = "Needs improvement ✗"
    
    print(f"AD vs {group:<5} {diff:<10.2f} {d:<12.4f} {interpret_cohens_d(d):<12} {r:<10.4f} {quality:<15}")

print("\n  Effect-size interpretation criteria:")
print("    Cohen's d: <0.2 negligible | 0.2-0.5 small effect | 0.5-0.8 medium effect | ≥0.8 large effect")
print("    Rank-Biserial r: <0.1 negligible | 0.1-0.3 small effect | 0.3-0.5 medium effect | ≥0.5 large effect")

# ============================================
# III. Overall assessment
# ============================================
print("\n" + "="*80)
print("III. Overall assessment")
print("-"*80)

# Sex-balance assessment
sex_balanced = p_all > 0.05

# Age-balance assessment
age_balanced = overall_p > 0.05

# Effect-size-based age-matching assessment
cn_d = cohens_d(ad_ages, cn_ages)
mci_d = cohens_d(ad_ages, mci_ages)
cn_r = rank_biserial_r(ad_ages, cn_ages)
mci_r = rank_biserial_r(ad_ages, mci_ages)
cn_age_ok = cn_d < 0.2  # Cohen's d < 0.2 = negligible
mci_age_ok = mci_d < 0.2

print(f"\nSex balance: {'✓ pass' if sex_balanced else '⚠ fail'} (three-group chi-square test p={p_all:.4f})")
print(f"Age balance: {'✓ pass' if age_balanced else '⚠ fail'} (Kruskal-Wallis p={overall_p:.4f})")
print(f"CN age matching: {'✓ pass' if cn_age_ok else '⚠ fail'} (Cohen's d={cn_d:.4f}, {interpret_cohens_d(cn_d)})")
print(f"MCI age matching: {'✓ pass' if mci_age_ok else '⚠ fail'} (Cohen's d={mci_d:.4f}, {interpret_cohens_d(mci_d)})")

# Paper-quality score
score = 0
score += 25 if sex_balanced else 0
score += 25 if age_balanced else (15 if overall_p > 0.01 else 0)
score += 25 if cn_age_ok else (15 if cn_d < 0.5 else 0)
score += 25 if mci_age_ok else (15 if mci_d < 0.5 else 0)

print(f"\nPaper-quality score: {score}/100")
if score >= 90:
    print("Assessment: excellent — fully meets the standard for a high-quality paper")
elif score >= 75:
    print("Assessment: good — suitable for publication; report the matching details in the Methods")
elif score >= 60:
    print("Assessment: acceptable — sensitivity analysis recommended")
else:
    print("Assessment: needs improvement — consider optimising the matching strategy or enlarging the sample pool")

# ============================================
# IV. Visualisation
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4.1 Bar chart of the sex distribution
ax1 = axes[0, 0]
sex_counts = df.groupby(['Group', 'Sex']).size().unstack(fill_value=0)
colors_sex = ['#66B2FF', '#FF9999']  # M blue, F pink
sex_counts.plot(kind='bar', ax=ax1, color=colors_sex, edgecolor='black', linewidth=0.5)
ax1.set_title('Gender Distribution by Group', fontsize=13, fontweight='bold')
ax1.set_xlabel('Group', fontsize=11)
ax1.set_ylabel('Number of Subjects', fontsize=11)
ax1.legend(['Male', 'Female'], fontsize=10)
ax1.tick_params(axis='x', rotation=0)
# Add value labels
for container in ax1.containers:
    ax1.bar_label(container, fmt='%d', fontsize=9)

# 4.2 Box plot of age
ax2 = axes[0, 1]
box_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bp = ax2.boxplot([ad_ages, cn_ages, mci_ages], 
                  labels=['AD', 'CN', 'MCI'],
                  patch_artist=True,
                  widths=0.5)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_title('Age Distribution by Group', fontsize=13, fontweight='bold')
ax2.set_xlabel('Group', fontsize=11)
ax2.set_ylabel('Age (years)', fontsize=11)
# Add mean markers
for i, ages in enumerate([ad_ages, cn_ages, mci_ages]):
    ax2.scatter(i+1, ages.mean(), color='red', s=80, zorder=5, marker='D')
ax2.legend([bp['boxes'][0], plt.Line2D([0], [0], marker='D', color='w', 
                                        markerfacecolor='red', markersize=8)],
           ['IQR Range', 'Mean'], fontsize=9)

# 4.3 Overlaid age histograms
ax3 = axes[1, 0]
for i, (group, ages, color) in enumerate(zip(['AD', 'CN', 'MCI'], 
                                               [ad_ages, cn_ages, mci_ages],
                                               box_colors)):
    ax3.hist(ages, bins=20, alpha=0.5, label=group, color=color, edgecolor='black', linewidth=0.5)
ax3.set_title('Age Distribution Histogram', fontsize=13, fontweight='bold')
ax3.set_xlabel('Age (years)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.legend(fontsize=10)

# 4.4 Summary table
ax4 = axes[1, 1]
ax4.axis('off')

# Build the summary text (with effect sizes)
summary_text = f"""
══════════════════════════════
     Statistical analysis summary
══════════════════════════════

Sample size:
  AD:  318 subjects
  CN:  318 subjects
  MCI: 318 subjects

Sex distribution (M/F):
  AD:  {ad_sex.get('M', 0)}/{ad_sex.get('F', 0)}
  CN:  {df[df['Group']=='CN']['Sex'].value_counts().get('M', 0)}/{df[df['Group']=='CN']['Sex'].value_counts().get('F', 0)}
  MCI: {df[df['Group']=='MCI']['Sex'].value_counts().get('M', 0)}/{df[df['Group']=='MCI']['Sex'].value_counts().get('F', 0)}

Age (mean±SD):
  AD:  {ad_ages.mean():.1f} ± {ad_ages.std():.1f}
  CN:  {cn_ages.mean():.1f} ± {cn_ages.std():.1f}
  MCI: {mci_ages.mean():.1f} ± {mci_ages.std():.1f}

Effect sizes (Cohen's d, vs AD):
  CN:  d={cn_d:.4f} ({interpret_cohens_d(cn_d)})
  MCI: d={mci_d:.4f} ({interpret_cohens_d(mci_d)})

Between-group differences:
  sex chi-square: p={p_all:.4f}
  age test: p={overall_p:.4f}

Quality score: {score}/100
══════════════════════════════
"""
ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='center', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Demographic Distribution Analysis - Summary 2', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()

# Save the figure
output_path = os.path.join(_OUT_DIR, "demographic_analysis.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved: {output_path}")
plt.show()

# ============================================
# V. Save the statistical report
# ============================================
report_path = os.path.join(_OUT_DIR, "statistics_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("                    Summary 2 — sex and age distribution statistics report\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Report generated at: {pd.Timestamp.now()}\n")
    f.write(f"Total samples: {len(df)}\n\n")
    
    f.write("I. Sample size\n")
    f.write("-"*40 + "\n")
    for group in ['AD', 'CN', 'MCI']:
        n = len(df[df['Group'] == group])
        f.write(f"  {group}: {n} subjects\n")
    
    f.write("\nII. Sex distribution\n")
    f.write("-"*40 + "\n")
    for group in ['AD', 'CN', 'MCI']:
        subset = df[df['Group'] == group]
        sex_counts = subset['Sex'].value_counts()
        f.write(f"  {group}: M={sex_counts.get('M', 0)}, F={sex_counts.get('F', 0)}\n")
    f.write(f"\n  three-group chi-square test: χ²={chi2_all:.2f}, p={p_all:.4f}\n")
    
    f.write("\nIII. Age distribution\n")
    f.write("-"*40 + "\n")
    for group in ['AD', 'CN', 'MCI']:
        stats_dict = age_stats[group]
        f.write(f"  {group}: {stats_dict['mean']:.2f} ± {stats_dict['std']:.2f} years")
        f.write(f" (range: {stats_dict['min']:.0f}-{stats_dict['max']:.0f})\n")
    
    f.write(f"\n  between-group age-difference test: p={overall_p:.4f}\n")
    
    # Effect-size information
    f.write(f"\n  Effect sizes (vs AD):\n")
    f.write(f"    Cohen's d:\n")
    f.write(f"      AD vs CN:  d={cohens_d(ad_ages, cn_ages):.4f} ({interpret_cohens_d(cohens_d(ad_ages, cn_ages))})\n")
    f.write(f"      AD vs MCI: d={cohens_d(ad_ages, mci_ages):.4f} ({interpret_cohens_d(cohens_d(ad_ages, mci_ages))})\n")
    f.write(f"    Rank-Biserial r:\n")
    f.write(f"      AD vs CN:  r={rank_biserial_r(ad_ages, cn_ages):.4f} ({interpret_rank_biserial(rank_biserial_r(ad_ages, cn_ages))})\n")
    f.write(f"      AD vs MCI: r={rank_biserial_r(ad_ages, mci_ages):.4f} ({interpret_rank_biserial(rank_biserial_r(ad_ages, mci_ages))})\n")
    
    f.write("\nIV. Paper-quality assessment\n")
    f.write("-"*40 + "\n")
    f.write(f"  Overall score: {score}/100\n")
    
    f.write("\nV. Conclusion\n")
    f.write("-"*40 + "\n")
    if score >= 75:
        f.write("  The dataset quality is good and it is suitable for statistical analysis.\n")
        f.write("  Report the matching procedure above and the statistical results in the Methods section of the paper.\n")
    else:
        f.write("  Further optimisation of the matching strategy is recommended.\n")

print(f"Statistical report saved: {report_path}")
print("\n" + "="*80)
print("Analysis complete!")
print("="*80)