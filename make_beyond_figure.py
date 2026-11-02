import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.linewidth': 0.6, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
d = json.load(open('compare_v4.json'))
A = d['annotation_noise']; sizes = [4, 8, 16, 32, 64, 128]
rules = [('ot_nwd', 'Algorithm 1', BLUE, 'o', '-'), ('ot_kadapt', r'OT, adaptive $\kappa_j$', VIOLET, 'v', '--'),
         ('ot_iou', 'OT, IoU cost', ORANGE, 's', '-'), ('simota', 'SimOTA-style', AQUA, 'D', '-'),
         ('atss', 'ATSS-style', YELLOW, '^', '-'), ('tal', 'Task-aligned (CIoU)', MAGENTA, 'P', '-')]
fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.25), constrained_layout=True)
for k, (key, title) in enumerate([('shift', '(a) Ground truth translated by 1 pixel'), ('scale', '(b) Ground-truth width and height jittered (10%)')]):
    a = ax[k]
    for r, lab, c, mk, ls in rules:
        if key == 'shift' and r == 'ot_kadapt': continue
        m = np.array([A[str(s)][r][key][0] for s in sizes]); e = np.array([A[str(s)][r][key][1] for s in sizes])
        a.fill_between(sizes, m - e, m + e, color=c, alpha=0.15, lw=0)
        a.plot(sizes, m, color=c, lw=1.6, ls=ls, marker=mk, ms=4.5, label=lab)
    a.set_xscale('log', base=2); a.set_xticks(sizes); a.set_xticklabels(sizes)
    a.set_xlabel('object size $s$ (pixels)'); a.set_ylabel('change of positive pairs'); a.set_title(title, loc='left')
    a.set_ylim(0, 0.8)
ax[0].legend(frameon=False, loc='upper right')
ax[1].legend(frameon=False, loc='upper left')
K = d['kappa_sweep']; kap = [3.2, 6.4, 12.8, 25.6, 51.2]
a = ax[2]
for s, c, mk in [(4, BLUE, 'o'), (16, ORANGE, 's'), (64, AQUA, 'D')]:
    a.plot(kap, [K[str(s)][str(k)]['abs'][0] for k in kap], color=c, lw=1.6, marker=mk, ms=4.5, label=f'$s={s}$, 1-pixel shift')
    a.plot(kap, [K[str(s)][str(k)]['rel'][0] for k in kap], color=c, lw=1.4, ls='--', marker=mk, ms=4.5, mfc='white', label=f'$s={s}$, shift $s/8$')
a.axvline(12.8, color=INK2, lw=0.8, ls=':'); a.text(13.3, 0.40, r'$\kappa=12.8$', fontsize=7.5, color=INK2)
a.set_xscale('log', base=2); a.set_xticks(kap); a.set_xticklabels(kap)
a.set_xlabel(r'Gaussian-Wasserstein scale $\kappa$ (pixels)'); a.set_ylabel('change of positive pairs'); a.set_ylim(0, 0.8)
a.set_title(r'(c) Algorithm 1 for different $\kappa$', loc='left'); a.legend(frameon=False, ncol=1, loc='upper right', fontsize=7)
fig.savefig('fig_beyond.png', dpi=300)
print('ok')
