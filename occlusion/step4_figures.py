# -*- coding: utf-8 -*-
"""Occlusion causality analysis Step 4: paper-ready figures (CPU, seconds).

Outputs (300 dpi, English labels, no titles, conforming to the paper's figure conventions):
  fig_occlusion_dose_response.png  dose-response curve (voxel count vs median Δp_true)
  fig_occlusion_flip.png           prediction flip matrix (baseline vs after occlusion)
  fig_occlusion_controls.png       comparison of the Δp_true distributions of MTL and the three controls
  fig_occlusion_paired.png         paired scatter (p0[true] vs p_after[true])
  fig_occlusion_by_class.png       Δp_true stratified by true class
  fig_occlusion_recall.png         change in recall for each class

Input: <work>/results/occlusion_raw.npz + occlusion_meta.json
Output: <work>/figures/*.png

Run: python occlusion/step4_figures.py"""

# Source: 27_occlusion_experiment/step4_figures.py (the computation logic is unchanged line by line; only the
# "environment bootstrap layer" was changed to be provided uniformly by occlusion_config; see
# occlusion/README.md for the refactoring notes).

import json
import sys
import numpy as np
from scipy import stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from occlusion_config import (  # noqa: E402
    FIG, OUT, SAT_THR, ensure_dirs,
    setup_stdout,
)

setup_stdout()
ensure_dirs()

raw = np.load(OUT / 'occlusion_raw.npz', allow_pickle=True)
meta = json.load(open(OUT / 'occlusion_meta.json', encoding='utf-8'))
cond_names = [str(x) for x in raw['cond_names']]
file_ids = [str(x) for x in raw['file_ids']]
true_lab = raw['true_labels']
cond_idx = raw['cond_idx']
P0 = raw['p0'].astype(np.float64)
P1 = raw['p_after'].astype(np.float64)

uniq_ids = sorted(set(file_ids))
id2row = {f: i for i, f in enumerate(uniq_ids)}
N = len(uniq_ids)


def mat_of(c):
    rows = {i: None for i in range(N)}
    for k, fid in enumerate(file_ids):
        if cond_names[cond_idx[k]] == c:
            rows[id2row[fid]] = P1[k]
    return np.array([rows[i] for i in range(N)])


P0m = np.array([P0[[k for k in range(len(file_ids))
                   if cond_names[cond_idx[k]] == cond_names[0] and file_ids[k] == fid][0]]
                for fid in uniq_ids])
TL = np.array([true_lab[[k for k in range(len(file_ids)) if file_ids[k] == fid][0]]
               for fid in uniq_ids])
base_pred = P0m.argmax(1)
IDX = np.arange(N)

# ★ saturation stratification (consistent with step3): the Δp_true of saturated samples is swallowed by the float64 softmax clamp
sat = P0m.max(1) > SAT_THR
NS = ~sat
print('saturated samples %d / %d, non-saturated %d (figures 1/4 use the non-saturated subset)' % (sat.sum(), N, NS.sum()))

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
C_MAIN, C_CTRL = '#c0392b', '#2c7fb8'

# ==================== figure 1: dose-response ====================
doses = ['MTL_d25_F1const', 'MTL_d40_F1const', 'MTL_d60_F1const',
         'MTL_d80_F1const', 'MTL_d100_F1const']
dvox = [meta['dose_vox'][k.replace('MTL_', '').replace('_F1const', '')] for k in doses]
dmed, dmean, dsem = [], [], []
for c in doses:
    Pa = mat_of(c)
    d = (P0m[IDX, TL] - Pa[IDX, TL])[NS]      # ★ non-saturated subset
    dmed.append(np.median(d)); dmean.append(d.mean())
    dsem.append(d.std(ddof=1) / np.sqrt(len(d)))
dmed, dmean, dsem = map(np.array, (dmed, dmean, dsem))
pct = np.array(dvox) / meta['mask_vox']['mtl'] * 100

fig, ax = plt.subplots(figsize=(3.6, 2.9))
ax.errorbar(pct, dmean, yerr=1.96 * dsem, fmt='o-', color=C_MAIN,
            capsize=3, lw=1.4, ms=4.5, mfc='white', mew=1.3, label='Mean $\\Delta p_{true}$')
ax.plot(pct, dmed, 's--', color='#7f8c8d', lw=1.2, ms=4, mfc='white',
        mew=1.2, label='Median $\\Delta p_{true}$')
ax.axhline(0, color='k', lw=0.7, ls=':')
ax.set_xlabel('MTL volume occluded (% of full MTL)')
ax.set_ylabel('$\\Delta p_{true}$ ($p_0 - p_{occ}$)')
ax.set_xticks(pct)
ax.set_xticklabels(['%.0f' % p for p in pct])
rho, prho = st.spearmanr(dvox, dmed)
ax.text(0.04, 0.94, '$\\rho$ = %.3f, $p$ = %.2g\n($n$ = %d non-saturated)' % (rho, prho, int(NS.sum())),
        transform=ax.transAxes, fontsize=7.5, va='top')
ax.legend(frameon=False, fontsize=7.5, loc='lower right')
fig.savefig(FIG / 'fig_occlusion_dose_response.png')
plt.close(fig)
print('saved fig_occlusion_dose_response.png  ρ=%.4f p=%.3g' % (rho, prho))

# ==================== figure 2: flip matrix ====================
Pa = mat_of('MTL_d100_F1const')
pred1 = Pa.argmax(1)
cm0 = np.zeros((3, 3), int)
cm1 = np.zeros((3, 3), int)
for i in range(N):
    cm0[TL[i], base_pred[i]] += 1
    cm1[TL[i], pred1[i]] += 1

fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.6))
names = ['CN', 'MCI', 'AD']
for ax, cm, ttl in zip(axes, (cm0, cm1), ('Baseline', 'MTL occluded (100%)')):
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm0.max())
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=9,
                    color='white' if cm[i, j] > 0.55 * cm0.max() else 'black')
    ax.set_xticks(range(3)); ax.set_xticklabels(names)
    ax.set_yticks(range(3)); ax.set_yticklabels(names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(ttl, fontsize=8.5)
    ax.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / 'fig_occlusion_flip.png')
plt.close(fig)
acc1 = np.mean(pred1 == TL)
print('saved fig_occlusion_flip.png   acc_after=%.4f' % acc1)

# ==================== figure 3: main mask vs controls ====================
ctrl_c1 = 'C1_%s_F1const' % meta['c1_name']
# ★ short labels + voxel counts on a separate row (to avoid a crowded x axis; the three-line labels of v1 overlapped)
show = [('MTL_d100_F1const', 'MTL', meta['mask_vox']['mtl']),
        ('C1_%s_F1const' % meta['c1_name'], 'Non-MTL\nvolume-matched', meta['mask_vox']['c1']),
        ('C2_ball_F1const', 'Random\nspheres', meta['mask_vox']['ball']),
        ('C3_hipp_F1const', 'Hippo-\ncampus', meta['mask_vox']['hipp'])]
data, labels, vox = [], [], []
for c, lb, vx in show:
    Pc = mat_of(c)
    data.append((P0m[IDX, TL] - Pc[IDX, TL])[NS])   # ★ non-saturated subset
    labels.append(lb); vox.append(vx)

fig, ax = plt.subplots(figsize=(3.8, 3.0))
parts = ax.violinplot(data, showextrema=False, widths=0.78)
for i, b in enumerate(parts['bodies']):
    b.set_facecolor(C_MAIN if i == 0 else C_CTRL)
    b.set_alpha(0.30 if i == 0 else 0.20)
    b.set_edgecolor(C_MAIN if i == 0 else C_CTRL)
    b.set_linewidth(1.0)
ax.boxplot(data, widths=0.20, showfliers=False, patch_artist=True,
           medianprops=dict(color='k', lw=1.1),
           boxprops=dict(facecolor='white', lw=0.9),
           whiskerprops=dict(lw=0.9), capprops=dict(lw=0.9))
ax.axhline(0, color='k', lw=0.7, ls=':')
ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel('$\\Delta p_{true}$ ($p_0 - p_{occ}$)')
ax.set_xlim(0.35, len(labels) + 0.65)
ax.margins(y=0.20)
ylim = ax.get_ylim()
# voxel counts on a secondary row (rotated), significance markers at the top of the axes
for i, vx in enumerate(vox):
    ax.text(i + 1, ylim[0] - 0.055 * (ylim[1] - ylim[0]), '{:,} vox'.format(vx),
            ha='center', va='top', fontsize=6.3, color='#555555', rotation=0)
for i, d in enumerate(data):
    w_, p_ = st.wilcoxon(d) if np.any(d != 0) else (np.nan, np.nan)
    star = '*' if p_ < 0.05 else 'n.s.'
    ax.text(i + 1, ylim[1] - 0.05 * (ylim[1] - ylim[0]), star, ha='center', fontsize=8.5)
ax.set_ylim(ylim[0] - 0.09 * (ylim[1] - ylim[0]), ylim[1])
fig.savefig(FIG / 'fig_occlusion_controls.png')
plt.close(fig)
print('saved fig_occlusion_controls.png (n=%d non-saturated)' % int(NS.sum()))

# ==================== figure 4: paired scatter ====================
p0t = P0m[IDX, TL]
p1t = Pa[IDX, TL]
fig, ax = plt.subplots(figsize=(3.3, 3.1))
corr = base_pred == TL
ax.scatter(p0t[corr], p1t[corr], s=9, alpha=0.55, color=C_CTRL,
           edgecolors='none', label='Correct ($n$=%d)' % corr.sum())
ax.scatter(p0t[~corr], p1t[~corr], s=16, alpha=0.8, color=C_MAIN,
           marker='^', edgecolors='none', label='Incorrect ($n$=%d)' % (~corr).sum())
lim = [min(p0t.min(), p1t.min()) * 0.98, 1.01]
ax.plot(lim, lim, 'k--', lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('$p_{true}$ before occlusion')
ax.set_ylabel('$p_{true}$ after MTL occlusion')
n_below = int((p1t < p0t).sum())
ax.text(0.04, 0.94, 'below diagonal: %d/%d' % (n_below, N),
        transform=ax.transAxes, fontsize=7.5, va='top')
ax.legend(frameon=False, fontsize=7.5, loc='lower right')
fig.savefig(FIG / 'fig_occlusion_paired.png')
plt.close(fig)
print('saved fig_occlusion_paired.png   below-diagonal=%d/%d' % (n_below, N))

# ==================== figure 5: Δp_true stratified by true class ====================
CLS = ['CN', 'MCI', 'AD']
grp = [(P0m[IDX, TL] - Pa[IDX, TL])[(TL == i) & NS] for i in range(3)]
fig, ax = plt.subplots(figsize=(3.6, 3.0))
parts = ax.violinplot(grp, showextrema=False, widths=0.78)
for b, col in zip(parts['bodies'], ('#7f8c8d', '#2c7fb8', C_MAIN)):
    b.set_facecolor(col); b.set_alpha(0.25); b.set_edgecolor(col); b.set_linewidth(1.0)
ax.boxplot(grp, widths=0.20, showfliers=False, patch_artist=True,
           medianprops=dict(color='k', lw=1.1),
           boxprops=dict(facecolor='white', lw=0.9),
           whiskerprops=dict(lw=0.9), capprops=dict(lw=0.9))
ax.axhline(0, color='k', lw=0.7, ls=':')
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(CLS)
ax.set_xlabel('True class'); ax.set_ylabel('$\\Delta p_{true}$ ($p_0 - p_{occ}$)')
ax.set_xlim(0.35, 3.65); ax.margins(y=0.20)
ylim = ax.get_ylim()
for i, g in enumerate(grp):
    p_ = st.wilcoxon(g)[1] if np.any(g != 0) else np.nan
    star = '***' if p_ < 0.001 else ('*' if p_ < 0.05 else 'n.s.')
    ax.text(i + 1, ylim[1] - 0.05 * (ylim[1] - ylim[0]), star, ha='center', fontsize=8.5)
    ax.text(i + 1, ylim[0] - 0.075 * (ylim[1] - ylim[0]), '$n$=%d' % len(g),
            ha='center', va='top', fontsize=6.5, color='#555555')
ax.set_ylim(ylim[0] - 0.13 * (ylim[1] - ylim[0]), ylim[1])
fig.savefig(FIG / 'fig_occlusion_by_class.png')
plt.close(fig)
H_, pH_ = st.kruskal(*grp)
print('saved fig_occlusion_by_class.png  KW H=%.3f p=%.3g' % (H_, pH_))

# ==================== figure 6: class-level recall change ====================
rec0 = np.array([np.mean(base_pred[TL == i] == TL[TL == i]) for i in range(3)])
rec1 = np.array([np.mean(pred1[TL == i] == TL[TL == i]) for i in range(3)])
fig, ax = plt.subplots(figsize=(3.4, 2.9))
x = np.arange(3); wd = 0.34
ax.bar(x - wd / 2, rec0, wd, color='#95a5a6', label='Baseline', edgecolor='white', lw=0.7)
ax.bar(x + wd / 2, rec1, wd, color=C_MAIN, label='MTL occluded', edgecolor='white', lw=0.7)
for i in range(3):
    ax.text(i, max(rec0[i], rec1[i]) + 0.035, '%+.3f' % (rec1[i] - rec0[i]),
            ha='center', fontsize=7.5,
            color=C_MAIN if rec1[i] < rec0[i] else '#2c7fb8')
ax.set_xticks(x); ax.set_xticklabels(CLS)
ax.set_ylim(0.70, 1.06); ax.set_ylabel('Recall')
ax.legend(frameon=False, fontsize=7.5, loc='lower left', ncol=1)
fig.savefig(FIG / 'fig_occlusion_recall.png')
plt.close(fig)
print('saved fig_occlusion_recall.png  recall base=%s after=%s'
      % (np.round(rec0, 4).tolist(), np.round(rec1, 4).tolist()))
print('\nfigures written to:', FIG)
