#!/usr/bin/env python3
"""
Section 3.2 hyperparameter grid-search result analysis — complete plotting and statistics code
========================================================================
This script generates all the figures and tables required for Section 3.2.3 of the paper:

  (a) Figure_S3_rf_importance     — random-forest permutation feature importance (horizontal bar chart)
  (b) Figure_S2_lr_violin         — box plot + violin plot + statistical significance for the different learning rates
  (c) Figure_S1_auc_robustness        — three-group box plot: comparison of the AUC distribution before and after excluding LR=1e-5
  (d) table_robustness_comparison   — summary statistics CSV comparing the mean/SD before and after exclusion
                                      (data only; the rendered table/plot is not part of the released figure set)

Dependencies: pip install pandas numpy matplotlib scipy scikit-learn
Input: all_training_results.csv (results of the 256 grid-search runs)
Output: all figures (PNG 300dpi + PDF vector), CSV statistics tables, console statistics summary
========================================================================
"""

import os, sys, ast, warnings
import numpy as np
import pandas as pd
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import kruskal, mannwhitneyu, gaussian_kde
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

# Deterministic jitter for the scatter overlays (the two RNG.normal calls in the violin and
# robustness panels). These previously used the unseeded global numpy state, which made those
# two figures differ at pixel level from run to run. Drawing from a seeded generator keeps the
# plots reproducible without perturbing the global random state used elsewhere.
RNG = np.random.default_rng(42)

# =========================================================================
# 0. Paths and global configuration
# =========================================================================
# Read the data and output paths from config.yaml (replacing the original hard-coded relative paths)
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config as _load_config, resolve_path as _resolve_path
_cfg = _load_config()
DATA_DIR = os.path.dirname(_resolve_path(_cfg, "data.grid_results"))
OUTPUT_DIR = _cfg["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- SCI plotting style ----------
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'DejaVu Sans'],
    'font.size':        10,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.dpi':      300,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
    'savefig.pad_inches': 0.05,
})

# ---------- colour scheme ----------
COLOR_LR = {
    1e-5:   '#D81B60',   # red — suboptimal learning rate
    2e-5:   '#4393C3',   # light blue
    5e-5:   '#2166AC',   # dark blue
    1e-4:   '#1B7837',   # green — optimal learning rate
}

COLOR_ROBUST = {
    'all':    '#607D8B',   # blue-grey — all-runs baseline
    'low':    '#D81B60',   # red — LR=1e-5
    'high':   '#1B7837',   # green — after exclusion
}

# =========================================================================
# 1. Data loading and parsing
# =========================================================================
print('=' * 60)
print('Loading data…')
df_raw = pd.read_csv(os.path.join(DATA_DIR, 'all_training_results.csv'))

def parse_config(s):
    return ast.literal_eval(s)

cfgs = df_raw['config'].apply(parse_config)

df = df_raw.copy()
df['attention_type']        = cfgs.apply(lambda x: x.get('attention_type', 'none'))
df['learning_rate']         = cfgs.apply(lambda x: float(x.get('learning_rate', 1e-5)))
df['batch_size']            = cfgs.apply(lambda x: int(x.get('batch_size', 3)))
df['dropout_rate']          = cfgs.apply(lambda x: float(x.get('dropout_rate', 0.0)))
df['augmentation_intensity']= cfgs.apply(lambda x: int(x.get('augmentation_intensity', 0)))
df['grad_accum_steps']      = cfgs.apply(lambda x: int(x.get('grad_accum_steps', 1)))
df['best_auc'] = df_raw['best_auc'].astype(float)
df['best_f1']  = df_raw['best_f1'].astype(float)
df['best_acc'] = df_raw['best_acc'].astype(float)

print(f'  {len(df)} runs in total')
print(f'  Learning rates: {sorted(df["learning_rate"].unique())}')
print(f'  Attention types: {list(df["attention_type"].unique())}')

# Split: suboptimal learning rate vs the rest
df_low  = df[df['learning_rate'] == 1e-5]      # n=64
df_high = df[df['learning_rate'] != 1e-5]      # n=192

# =========================================================================
# 2. Figure (a) — random-forest permutation feature importance
# =========================================================================
print('\n' + '=' * 60)
print('Figure (a) random-forest permutation feature importance')

attn_map = {'none': 0, 'cbam': 1, 'eca': 2, 'axial': 3}
X = pd.DataFrame({
    'learning_rate':      df['learning_rate'],
    'attention_type':     df['attention_type'].map(attn_map),
    'batch_size':         df['batch_size'],
    'dropout_rate':       df['dropout_rate'],
    'augmentation':       df['augmentation_intensity'],
    'grad_accum_steps':   df['grad_accum_steps'],
})
y = df['best_auc'].values

display_names = [
    'Learning Rate', 'Attention Type', 'Batch Size',
    'Dropout Rate', 'Augmentation', 'Grad Accum Steps',
]

rf = RandomForestRegressor(n_estimators=500, min_samples_split=5,
                           min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X, y)
cv_r2 = cross_val_score(rf, X, y, cv=5, scoring='r2')
print(f'  RF 5-fold CV R² = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}')
print(f'  RF Training R²  = {rf.score(X, y):.4f}')

perm = permutation_importance(rf, X, y, n_repeats=30, random_state=42,
                              n_jobs=-1, scoring='r2')
imps     = perm.importances_mean
imps_std = perm.importances_std
pcts     = imps / imps.sum() * 100

order    = np.argsort(imps)
names_sorted = [display_names[i] for i in order]

print('  Feature importance (ascending):')
for i in range(len(order)):
    print(f'    {names_sorted[i]:20s}  {imps[order[i]]:.6f}  ({pcts[order[i]]:.1f}%)  ±{imps_std[order[i]]:.6f}')
print(f'  >>> learning-rate share = {pcts[display_names.index("Learning Rate")]:.1f}%')

# --- plotting ---
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = ['#D81B60' if n == 'Learning Rate' else '#4393C3' for n in names_sorted]
bars = ax.barh(range(len(names_sorted)), imps[order], xerr=imps_std[order],
               color=bar_colors, edgecolor='black', linewidth=0.5,
               capsize=3, height=0.6)
for i, (b, p) in enumerate(zip(bars, pcts[order])):
    ax.text(b.get_width() + imps_std[order][i] + 0.0005,
            b.get_y() + b.get_height() / 2, f'{p:.1f}%',
            va='center', fontsize=9, fontweight='bold')
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels(names_sorted, fontsize=10)
ax.set_xlabel('Permutation Importance (R² Decrease)', fontsize=11, fontweight='bold')
ax.set_title('Random Forest Feature Importance for Validation AUC\n'
             '(Permutation Importance, n_repeats = 30)',
             fontsize=12, fontweight='bold')
ax.set_xlim(0, imps[order[-1]] * 1.25)
ax.grid(axis='x', alpha=0.3, linestyle='--')

lr_pct = pcts[display_names.index('Learning Rate')]
ax.annotate(f'Learning Rate dominates\nwith {lr_pct:.1f}% of total importance',
            xy=(imps[order[-1]], len(names_sorted) - 1),
            xytext=(imps[order[-1]] * 0.7, len(names_sorted) - 2.2),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
            fontsize=9, fontweight='bold', color='#D81B60',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'Figure_S3_rf_importance.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'Figure_S3_rf_importance.pdf'))
plt.close()
print('  → Figure_S3_rf_importance.png / .pdf')

# =========================================================================
# 3. Figure (b) — box plot + violin plot + statistical significance for the different learning rates
# =========================================================================
print('\n' + '=' * 60)
print('Figure (b) violin plot + box plot for the different learning rates + Kruskal-Wallis')

lr_vals  = sorted(df['learning_rate'].unique())
lr_labels= ['1e-5', '2e-5', '5e-5', '1e-4']
lr_data  = [df[df['learning_rate'] == lr]['best_auc'].values for lr in lr_vals]

print('  Per-group statistics:')
for lr, lbl, d in zip(lr_vals, lr_labels, lr_data):
    print(f'    LR={lbl}: mean={d.mean():.4f}, std={d.std():.4f}, '
          f'min={d.min():.4f}, max={d.max():.4f}, n={len(d)}')

# Kruskal-Wallis
H, p_kw = kruskal(*lr_data)
print(f'  Kruskal-Wallis: H = {H:.2f}, p = {p_kw:.2e}')

# Dunn's post-hoc (Mann-Whitney U + Bonferroni)
print('  Dunn\'s post-hoc (Bonferroni corrected):')
n_pairs = 6
for i, j in combinations(range(4), 2):
    U, p = mannwhitneyu(lr_data[i], lr_data[j], alternative='two-sided')
    p_adj = min(p * n_pairs, 1.0)
    sig = '***' if p_adj < 0.001 else ('**' if p_adj < 0.01 else
           ('*' if p_adj < 0.05 else 'ns'))
    print(f'    {lr_labels[i]} vs {lr_labels[j]}: U={U:.0f}, p_adj={p_adj:.4f} {sig}')

# significance letters
letters = {lr_vals[0]: 'a'}
for lr in lr_vals[1:]:
    letters[lr] = 'b'

# --- plotting ---
fig, ax = plt.subplots(figsize=(8, 6))
pos = np.arange(4)
vp = ax.violinplot(lr_data, positions=pos, showmeans=False,
                   showmedians=False, showextrema=False, widths=0.7)
for body, lr in zip(vp['bodies'], lr_vals):
    body.set_facecolor(COLOR_LR[lr])
    body.set_alpha(0.3)
    body.set_edgecolor('black')
    body.set_linewidth(0.5)

ax.boxplot(lr_data, positions=pos, widths=0.2, patch_artist=True,
           showfliers=True,
           boxprops=dict(facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.9),
           whiskerprops=dict(color='black', linewidth=0.8),
           capprops=dict(color='black', linewidth=0.8),
           medianprops=dict(color='red', linewidth=1.5),
           flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5))

for i, d in enumerate(lr_data):
    jit = RNG.normal(0, 0.04, len(d))
    ax.scatter(np.full(len(d), pos[i]) + jit, d, alpha=0.15, s=8,
               color='black', zorder=1)

# letter labels
for i, lr in enumerate(lr_vals):
    ax.text(pos[i], lr_data[i].max() + 0.005, f'({letters[lr]})',
            ha='center', fontsize=12, fontweight='bold')

# significance bracket
ym = max(d.max() for d in lr_data)
ax.plot([pos[0], pos[0], pos[-1], pos[-1]],
        [ym + 0.015, ym + 0.018, ym + 0.018, ym + 0.015],
        'k-', linewidth=0.8)
ax.text((pos[0] + pos[-1]) / 2, ym + 0.020, '***',
        ha='center', fontsize=14, fontweight='bold')

ax.plot([pos[0], pos[0], pos[1], pos[1]],
        [ym + 0.027, ym + 0.030, ym + 0.030, ym + 0.027],
        'k-', linewidth=0.8)
ax.text((pos[0] + pos[1]) / 2, ym + 0.032, '***',
        ha='center', fontsize=14, fontweight='bold')

ax.set_xticks(pos)
ax.set_xticklabels([f'{lr:.0e}' for lr in lr_vals], fontsize=11)
ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation AUC (Macro-Averaged)', fontsize=12, fontweight='bold')
ax.set_title('Effect of Learning Rate on Model Performance\n'
             '(Violin Plot + Box Plot with Statistical Annotations)',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

ax.text(0.98, 0.02,
        f'Kruskal-Wallis: H = {H:.2f}, p = {p_kw:.2e}\n'
        f'Groups: a = 1e-5 (significantly lower)\n'
        f'        b = 2e-5, 5e-5, 1e-4 (no significant difference)\n'
        f'*** p < 0.001 (Dunn\'s post-hoc, Bonferroni corrected)',
        transform=ax.transAxes, fontsize=8, va='bottom', ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'Figure_S2_lr_violin.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'Figure_S2_lr_violin.pdf'))
plt.close()
print('  → Figure_S2_lr_violin.png / .pdf')

# =========================================================================
# 4. Figure (c) — three-group box plot: comparison of the AUC distribution before and after excluding LR=1e-5
# =========================================================================
print('\n' + '=' * 60)
print('Figure (c) three-group box plot: AUC distribution before and after excluding the suboptimal learning rate')

groups = {
    'All 256 Runs\n(n=256)':        df['best_auc'].values,
    'LR=1e-5 Only\n(n=64)':         df_low['best_auc'].values,
    'Excluding 1e-5\n(n=192)':      df_high['best_auc'].values,
}
group_colors = [
    COLOR_ROBUST['all'],
    COLOR_ROBUST['low'],
    COLOR_ROBUST['high'],
]

fig, ax = plt.subplots(figsize=(7, 6))
pos = [0, 1, 2]
names = list(groups.keys())
data  = list(groups.values())

bp = ax.boxplot(data, positions=pos, widths=0.35, patch_artist=True,
                showfliers=False,
                boxprops=dict(edgecolor='black', linewidth=0.8, alpha=0.7),
                whiskerprops=dict(color='black', linewidth=0.8),
                capprops=dict(color='black', linewidth=0.8),
                medianprops=dict(color='black', linewidth=2))
for patch, c in zip(bp['boxes'], group_colors):
    patch.set_facecolor(c)

for i, (d, c) in enumerate(zip(data, group_colors)):
    jit = RNG.normal(0, 0.06, len(d))
    ax.scatter(np.full(len(d), pos[i]) + jit, d, alpha=0.25, s=12,
               color=c, edgecolors='none', zorder=2)

for i, d in enumerate(data):
    m = d.mean()
    ax.plot(i, m, 'D', color='black', markersize=8,
            markeredgecolor='white', markeredgewidth=0.8, zorder=5)
    ax.annotate(f'{m:.4f}', xy=(i, m), xytext=(i + 0.3, m + 0.002),
                fontsize=9, fontweight='bold', ha='left',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

ym = max(d.max() for d in data)
ax.plot([0, 0, 2, 2],
        [ym + 0.005, ym + 0.007, ym + 0.007, ym + 0.005],
        'k-', linewidth=0.8)
ax.text(1, ym + 0.008, 'p < 0.001', ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(pos)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel('Validation AUC (Macro-Averaged)', fontsize=11, fontweight='bold')
ax.set_title('Robustness Analysis: Removing Suboptimal Learning Rate\n',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

legend_elements = [
    Patch(facecolor=COLOR_ROBUST['all'],  alpha=0.7, label='All 256 Runs'),
    Patch(facecolor=COLOR_ROBUST['low'],  alpha=0.7, label='LR=1e-5 Only (Suboptimal)'),
    Patch(facecolor=COLOR_ROBUST['high'], alpha=0.7, label='Excluding 1e-5 (Optimal)'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'Figure_S1_auc_robustness.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'Figure_S1_auc_robustness.pdf'))
plt.close()
print('  → Figure_S1_auc_robustness.png / .pdf')

# =========================================================================
# 5. Table (d) — summary statistics table comparing the mean/SD before and after exclusion
# =========================================================================
print('\n' + '=' * 60)
print('Table (d) comparison of performance statistics before and after excluding LR=1e-5')

metrics      = ['best_auc', 'best_f1', 'best_acc']
metric_names = {'best_auc': 'Validation AUC', 'best_f1': 'Validation F1',
                'best_acc': 'Validation Accuracy'}

stats_all  = {}
stats_high = {}
for m in metrics:
    stats_all[m]  = {'mean': df[m].mean(), 'std': df[m].std(),
                     'min': df[m].min(), 'max': df[m].max()}
    stats_high[m] = {'mean': df_high[m].mean(), 'std': df_high[m].std(),
                     'min': df_high[m].min(), 'max': df_high[m].max()}

print(f'  {"Metric":<20s} {"Stat":<10s} {"All 256":>12s} {"Excl 1e-5":>16s} {"Change":>10s}')
print('  ' + '-' * 70)
for m in metrics:
    name = metric_names[m]
    for s in ['mean', 'std']:
        before = stats_all[m][s]
        after  = stats_high[m][s]
        delta  = after - before
        print(f'  {name:<20s} {s:<10s} {before:>12.4f} {after:>16.4f} {delta:>+10.4f}')

# output CSV
csv_rows = []
for m in metrics:
    name = metric_names[m]
    csv_rows.append({
        'Metric': name,
        'Mean (All 256)':     f'{stats_all[m]["mean"]:.4f}',
        'Std (All 256)':      f'{stats_all[m]["std"]:.4f}',
        'Mean (Excl 1e-5)':   f'{stats_high[m]["mean"]:.4f}',
        'Std (Excl 1e-5)':    f'{stats_high[m]["std"]:.4f}',
        'Mean Change':        f'{stats_high[m]["mean"] - stats_all[m]["mean"]:+.4f}',
        'Std Change':         f'{stats_high[m]["std"]  - stats_all[m]["std"]:+.4f}',
    })
pd.DataFrame(csv_rows).to_csv(os.path.join(OUTPUT_DIR, 'table_robustness_comparison.csv'),
                               index=False)
print('  → table_robustness_comparison.csv')

# =========================================================================
# 6. Console statistics summary (for direct citation in paper writing)
# =========================================================================
print('\n' + '=' * 60)
print('Statistics summary for paper writing')
print('=' * 60)

print(f'\n[Random Forest] learning-rate share of importance = {lr_pct:.1f}%')
print(f'[Kruskal-Wallis] H = {H:.2f}, p = {p_kw:.2e}')
print(f'[LR=1e-5] AUC = {df_low["best_auc"].mean():.4f} ± {df_low["best_auc"].std():.4f}')
print(f'[All 256]  AUC = {df["best_auc"].mean():.4f} ± {df["best_auc"].std():.4f}')
print(f'[Excl LR]  AUC = {df_high["best_auc"].mean():.4f} ± {df_high["best_auc"].std():.4f}')
print(f'[AUC gain]  Δmean = {stats_high["best_auc"]["mean"] - stats_all["best_auc"]["mean"]:+.4f}, '
      f'Δstd = {stats_high["best_auc"]["std"] - stats_all["best_auc"]["std"]:+.4f}')

print(f'\nAUC per learning rate:')
for lr, lbl, d in zip(lr_vals, lr_labels, lr_data):
    print(f'  LR={lbl}: {d.mean():.4f} ± {d.std():.4f}')

print(f'\nAll outputs saved to: {OUTPUT_DIR}')
print('Done.')
