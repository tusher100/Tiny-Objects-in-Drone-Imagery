import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# palette: categorical slots 1-3 of the validated reference palette (light mode)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42,
})

def iou_shift(s, d):
    d = np.minimum(d, s)
    return (s - d) / (s + d)

# ---------------- Figure: similarity curves + stability vs size ----------------
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.42))
off = np.linspace(0, 40, 801)
kappa = 12.8
a0 = ax[0]
a0.plot(off, np.exp(-off / kappa), color=BLUE, lw=1.6, label='Gauss.-Wass., all $s$')
for s, ls in ((4, '-'), (16, '--'), (64, ':')):
    a0.plot(off, iou_shift(s, off), color=ORANGE, lw=1.4, ls=ls, label=f'IoU, $s={s}$')
a0.legend(loc='upper right', fontsize=6.8, handlelength=2.0, borderaxespad=0.1)
a0.set_xlabel('Offset (pixels)'); a0.set_ylabel('Similarity'); a0.set_xlim(0, 40); a0.set_ylim(0, 1.02)
a0.set_title('(a) Similarity under translation', loc='left')

sizes = np.array([4, 8, 16, 32, 64])
dist_nwd = np.array([1.1786, 0.5872, 0.3091, 0.1565, 0.0868])
dist_iou = np.array([59.4597, 0.6955, 0.3113, 0.1574, 0.0878])
flip = np.array([0.8864, 0.5356, 0.2767, 0.1367, 0.0688])
a1 = ax[1]
a1.plot(sizes, dist_iou, color=ORANGE, lw=1.6, marker='s', ms=4.5, label='IoU cost')
a1.plot(sizes, dist_nwd, color=BLUE, lw=1.6, marker='o', ms=4.5, label='Gaussian-Wasserstein cost')
a1.set_xscale('log', base=2); a1.set_yscale('log'); a1.set_xticks(sizes); a1.set_xticklabels([str(s) for s in sizes])
a1.set_xlabel('Object side $s$ (pixels)'); a1.set_ylabel('Distance (object sizes)')
a1.set_title('(b) Top-3 predictions to object', loc='left')
a1.annotate('IoU', xy=(4, 59.46), xytext=(5.0, 30), color=INK2, fontsize=7.5)
a1.annotate('Gaussian-W.', xy=(4, 1.18), xytext=(4.3, 2.0), color=INK2, fontsize=7.5)
a1.minorticks_off()
a2 = ax[2]
a2.plot(sizes, flip, color=ORANGE, lw=1.6, marker='s', ms=4.5)
a2.set_xscale('log', base=2); a2.set_xticks(sizes); a2.set_xticklabels([str(s) for s in sizes]); a2.minorticks_off()
a2.set_ylim(0, 1); a2.set_xlabel('Object side $s$ (pixels)'); a2.set_ylabel('Fraction of positives changed')
a2.set_title('(c) Hard IoU$\\geq$0.5 rule, 1 px shift', loc='left')
fig.savefig('fig_similarity_stability.pdf'); fig.savefig('fig_similarity_stability.png', dpi=200); plt.close(fig)

# ---------------- Figure: VisDrone ----------------
runs = {('base', 0): dict(AP=15.92, AP50=26.90, v=0.38, t=2.57, s=8.98, r=25.16),
        ('base', 1): dict(AP=16.18, AP50=27.33, v=0.56, t=2.60, s=8.94, r=25.67),
        ('ot', 0): dict(AP=14.81, AP50=26.68, v=0.36, t=2.92, s=9.51, r=23.87),
        ('ot', 1): dict(AP=15.42, AP50=27.58, v=0.40, t=3.07, s=9.49, r=24.73)}
keys = ['v', 't', 's', 'r', 'AP', 'AP50']
labels = ['$<8$', '8–16', '16–32', '$\\geq$32', 'AP', 'AP$_{50}$']
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.32))
a0 = ax[0]
x = np.arange(len(keys))
bmean = np.array([np.mean([runs[('base', s)][k] for s in (0, 1)]) for k in keys])
omean = np.array([np.mean([runs[('ot', s)][k] for s in (0, 1)]) for k in keys])
diff = omean - bmean  # differences of the per-run means (Table 3)
a0.bar(x, diff, width=0.56, color=BLUE, alpha=0.9, zorder=2, label='Entropic OT, mean')
for sd, mk in ((0, 'o'), (1, 'D')):
    a0.scatter(x, [runs[('ot', sd)][k] - bm for k, bm in zip(keys, bmean)], s=16, marker=mk,
               facecolor='white', edgecolor=INK, linewidth=0.8, zorder=3, label=f'Entropic OT, seed {sd}')
    a0.scatter(x + 0.36, [runs[('base', sd)][k] - bm for k, bm in zip(keys, bmean)], s=14, marker=mk,
               facecolor=ORANGE, edgecolor=ORANGE, linewidth=0.6, zorder=3, label=f'Baseline, seed {sd}')
a0.axhline(0, color=INK2, lw=0.8, zorder=1)
for xi, d in zip(x, diff):
    a0.text(xi, 0.86, f"{d:+.3f}", ha='center', va='center', fontsize=7.5, color=INK)
a0.axvline(3.5, color=GRID, lw=0.8)
a0.set_xticks(x); a0.set_xticklabels(labels)
a0.set_ylabel('AP change vs. baseline mean (points)'); a0.set_ylim(-1.75, 1.0)
a0.set_title('(a) Test AP by object size (pixels)', loc='left')
a0.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6.8, ncol=3, handletextpad=0.3, columnspacing=0.9)
a0.grid(axis='x', visible=False)

curves = json.load(open('epoch_curves.json'))
a1 = ax[1]
for name, col, lab in (('default', ORANGE, 'Baseline'), ('ot_nwd', BLUE, 'Entropic OT')):
    for sd, ls in ((0, '-'), (1, '--')):
        c = curves[f'{name}_s{sd}']
        ep = sorted(int(e) for e in c)
        a1.plot(ep, [c[str(e)]['map'] for e in ep], color=col, lw=1.5, ls=ls, label=f'{lab}, seed {sd}')
a1.set_xlabel('Epoch'); a1.set_ylabel('Validation mAP$_{50:95}$'); a1.set_xlim(1, 15)
a1.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
a1.set_title('(b) Validation accuracy during training', loc='left')
a1.legend(loc='lower right', fontsize=7)
fig.savefig('fig_visdrone.pdf'); fig.savefig('fig_visdrone.png', dpi=200); plt.close(fig)

# ---------------- Figure: Sinkhorn residual ----------------
if os.path.exists('theory_checks.json'):
    T = json.load(open('theory_checks.json'))['sinkhorn_residual_curve']
    fig, ax = plt.subplots(figsize=(4.3, 2.5))
    for (e, col, mk) in (('0.05', ORANGE, 's'), ('0.1', BLUE, 'o'), ('0.5', AQUA, '^')):
        r = np.maximum(np.array(T['curves'][e]), 1e-17)
        it = np.arange(1, len(r) + 1)
        ax.plot(it, r, color=col, lw=1.5, marker=mk, ms=3.5, markevery=20, label=f'$\\varepsilon={e}$')
    ax.axvline(50, color=INK2, lw=0.8, ls=':')
    ax.annotate('training: 50 iterations', xy=(50, 1e-1), xytext=(55, 1e-1), fontsize=7.5, color=INK2)
    ax.set_yscale('log'); ax.set_xlabel('Sinkhorn iteration'); ax.set_ylabel('Row-marginal error ($\\ell_1$)')
    ax.set_xlim(1, len(r)); ax.legend(loc='upper right')
    fig.savefig('fig_sinkhorn.pdf'); fig.savefig('fig_sinkhorn.png', dpi=200); plt.close(fig)
print('figures done')
