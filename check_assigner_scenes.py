"""Numerical checks of the theorems in the paper (numpy/scipy, CPU only).

A  sharpness of the stability constant on the 2x2 example
B  stability bounds (total-variation form and plan-weighted form) on random dense problems
C  epsilon -> 0 suboptimality bound against the exact LP value
D  Sinkhorn contraction in Hilbert's projective metric vs the Birkhoff rate
E  assigner-like synthetic scenes (same rules as ot_assigner.py): Sinkhorn residual after 50 iterations,
   plan change under 1 px jitter vs the bounds, positive-set changes, two-prediction example of Prop. 3
"""
import json, numpy as np
from scipy.special import logsumexp
from scipy.optimize import linprog

OUT = {}

def sinkhorn(C, a, b, eps, iters=None, tol=1e-12, max_iter=40000, return_hist=False):
    logK = -C / eps
    la, lb = np.log(a), np.log(b)
    f = np.zeros_like(a); g = np.zeros_like(b)
    hist = []
    it = 0
    while True:
        f = la - logsumexp(logK + g[None, :], axis=1)
        g = lb - logsumexp(logK + f[:, None], axis=0)
        it += 1
        if return_hist: hist.append(g.copy())
        if iters is not None and it >= iters: break
        if iters is None and (it % 25 == 0 or it >= max_iter):
            P = np.exp(logK + f[:, None] + g[None, :])
            if np.abs(P.sum(1) - a).sum() < tol or it >= max_iter: break
    P = np.exp(logK + f[:, None] + g[None, :])
    return (P, f, g, hist) if return_hist else P

def osc(x): return float(x.max() - x.min())

# ---------------------------------------------------------------- E: assigner-like scenes
def anchors(img=960, strides=(8, 16, 32)):
    pts, st = [], []
    for s in strides:
        n = img // s; xs = (np.arange(n) + 0.5) * s
        gx, gy = np.meshgrid(xs, xs); pts.append(np.stack([gx.ravel(), gy.ravel()], -1)); st.append(np.full(n * n, s))
    return np.concatenate(pts), np.concatenate(st).astype(float)

def phi(b): return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2, (b[..., 2] - b[..., 0]) / 2, (b[..., 3] - b[..., 1]) / 2], -1)
def w2(p, g): return np.linalg.norm(phi(p)[:, None] - phi(g)[None], axis=2)
def iou(p, g):
    lt = np.maximum(p[:, None, :2], g[None, :, :2]); rb = np.minimum(p[:, None, 2:], g[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    return inter / ((p[:, 2:] - p[:, :2]).prod(-1)[:, None] + (g[:, 2:] - g[:, :2]).prod(-1)[None] - inter + 1e-12)

CFG = dict(eps=0.1, kappa=12.8, cls_w=1.0, loc_w=3.0, radius=2.5, min_cand=9, max_k=10, big=1e4)

def candidates(gt, anc, strd):
    gc = (gt[:, :2] + gt[:, 2:]) / 2
    nd = np.linalg.norm(gc[:, None] - anc[None], axis=2) / strd[None]
    cand = nd <= CFG['radius']
    top = np.argsort(nd, axis=1)[:, :CFG['min_cand']]
    for j in range(len(gt)): cand[j, top[j]] = True
    cidx = np.where(cand.any(0))[0]
    return cidx, cand[:, cidx].T

def build(pred, z, gt, cand, kind, k=None):
    sim = np.exp(-w2(pred, gt) / CFG['kappa']) if kind == 'nwd' else iou(pred, gt)
    C = CFG['cls_w'] * (-z)[:, None] + CFG['loc_w'] * (1 - sim)
    C = np.where(cand, C, CFG['big'])
    if k is None:
        simc = np.where(cand, sim, 0)
        M = len(pred)
        k = np.clip(np.sort(simc, axis=0)[::-1][:min(CFG['max_k'], M)].sum(0), 1, CFG['max_k'])
        if k.sum() > 0.9 * M: k = k * (0.9 * M / k.sum())
    M = len(pred)
    a = np.full(M, 1 / M); b = np.concatenate([[M - k.sum()], k]) / M
    Cf = np.concatenate([np.zeros((M, 1)), C], 1)
    return Cf, a, b, k, sim

def positives(P, k):
    col = P[:, 1:]; M = col.shape[0]
    kk = np.maximum(np.round(k), 1).astype(int)
    order = np.argsort(-col, axis=0, kind='stable')
    Pm = np.zeros_like(col, bool)
    for j in range(col.shape[1]): Pm[order[:kk[j], j], j] = True
    jbest = np.where(Pm, col, -1).argmax(1); pos = Pm.any(1)
    return {(int(i), int(jbest[i])) for i in np.where(pos)[0]}

def jitter(pred, delta, rng):
    ang = rng.uniform(0, 2 * np.pi, len(pred)); d = delta * np.stack([np.cos(ang), np.sin(ang)], 1)
    return pred + np.concatenate([d, d], 1)

anc, strd = anchors()
rng = np.random.default_rng(3)
E = {}
for s in (4, 8, 16, 32, 64):
    acc = {k: [] for k in ('res50_nwd', 'res50_iou', 'l1_nwd', 'l1_iou', 'bound_tv', 'bound_w', 'bound_post',
                           'posflip_nwd', 'posflip_iou', 'hardflip', 'dist_nwd', 'dist_iou', 'M')}
    for trial in range(40):
        G = 8
        c = rng.uniform(64, 960 - 64, (G, 2)); gt = np.concatenate([c - s / 2, c + s / 2], 1)
        cidx, cand = candidates(gt, anc, strd)
        A0 = anc[cidx]; M = len(cidx)
        sz = s * rng.uniform(0.8, 1.2, (M, 1)); ctr = A0 + rng.normal(0, s / 4, A0.shape)
        pred = np.concatenate([ctr - sz / 2, ctr + sz / 2], 1)
        z = np.clip(rng.normal(-3, 1, M), -9.21, 9.21)
        pred2 = jitter(pred, 1.0, rng)
        acc['M'].append(M)
        for kind in ('nwd', 'iou'):
            Cf, a, b, k, sim = build(pred, z, gt, cand, kind)
            Pref = sinkhorn(Cf, a, b, CFG['eps'], iters=1500)
            P50 = sinkhorn(Cf, a, b, CFG['eps'], iters=50)
            acc['res50_' + kind].append(np.abs(P50.sum(1) - a).sum())
            # fixed marginals (theorem setting)
            Cf2, _, _, _, _ = build(pred2, z, gt, cand, kind, k=k)
            Pref2 = sinkhorn(Cf2, a, b, CFG['eps'], iters=1500)
            acc['l1_' + kind].append(np.abs(Pref - Pref2).sum())
            if kind == 'nwd':
                dloc = CFG['loc_w'] * 1.0 / CFG['kappa']
                bfg = 1 - b[0]
                acc['bound_tv'].append(min(2.0, dloc / CFG['eps']))
                acc['bound_w'].append(min(2.0, 2 * dloc * np.sqrt(bfg) / CFG['eps']))
                D = Cf2 - Cf
                acc['bound_post'].append(np.sqrt(2) / CFG['eps'] * np.sqrt((D ** 2 * np.maximum(Pref, Pref2)).sum()))
            # positive sets as implemented (k recomputed, 50 iterations)
            pos1 = positives(P50, k)
            Cf2b, a2, b2, k2, _ = build(pred2, z, gt, cand, kind)
            pos2 = positives(sinkhorn(Cf2b, a2, b2, CFG['eps'], iters=50), k2)
            acc['posflip_' + kind].append(len(pos1 ^ pos2) / max(1, len(pos1 | pos2)))
            gc = (gt[:, :2] + gt[:, 2:]) / 2; pc = (pred[:, :2] + pred[:, 2:]) / 2
            acc['dist_' + kind].append(np.mean([np.linalg.norm(pc[i] - gc[j]) / s for i, j in pos1]))
        h1 = {(i, j) for i, j in zip(*np.where((iou(pred, gt) >= 0.5) & cand))}
        h2 = {(i, j) for i, j in zip(*np.where((iou(pred2, gt) >= 0.5) & cand))}
        acc['hardflip'].append(len(h1 ^ h2) / len(h1 | h2) if (h1 | h2) else np.nan)
    E[s] = {k: float(np.nanmean(v)) for k, v in acc.items()}
    E[s]['res50_nwd_max'] = float(np.max(acc['res50_nwd'])); E[s]['res50_iou_max'] = float(np.max(acc['res50_iou']))
OUT['E_scenes'] = E

# Proposition 3(ii): two coincident predictions on one s x s object, shift one by delta
def two_pred(s, delta, kind, eps=0.1, lam=3.0, kappa=12.8):
    t = lam * (2 * delta / (s + delta)) if kind == 'iou' else lam * (1 - np.exp(-delta / kappa))
    a = b = np.array([0.5, 0.5])
    P0 = sinkhorn(np.array([[0, 1.0], [0, 1.0]]), a, b, eps)
    P1 = sinkhorn(np.array([[0, 1.0 + t], [0, 1.0]]), a, b, eps)
    return float(np.abs(P1 - P0).sum()), float(np.tanh(t / (4 * eps)))
OUT['P3_two_pred'] = {str(s): {k: two_pred(s, 1.0, k) for k in ('iou', 'nwd')} for s in (2, 4, 8, 16, 32, 64)}

# similarity curves for Figure 1
off = np.linspace(0, 40, 401)
curves = {}
for s in (4, 16, 64):
    g = np.array([[0, 0, s, s]], float)
    p = np.stack([off, np.zeros_like(off), off + s, np.full_like(off, s)], 1)
    curves[s] = dict(iou=iou(p, g)[:, 0].tolist(), nwd=np.exp(-w2(p, g)[:, 0] / 12.8).tolist())
OUT['curves'] = dict(offset=off.tolist(), **{str(k): v for k, v in curves.items()})

# Sinkhorn residual vs iterations for Figure (masked scene vs dense), eps in {0.05,0.1,0.5}
rng = np.random.default_rng(4)
res_curves = {}
s = 16; G = 8
c = rng.uniform(64, 960 - 64, (G, 2)); gt = np.concatenate([c - s / 2, c + s / 2], 1)
cidx, cand = candidates(gt, anc, strd); A0 = anc[cidx]; M = len(cidx)
sz = s * rng.uniform(0.8, 1.2, (M, 1)); ctr = A0 + rng.normal(0, s / 4, A0.shape)
pred = np.concatenate([ctr - sz / 2, ctr + sz / 2], 1); z = np.clip(rng.normal(-3, 1, M), -9.21, 9.21)
for e in (0.05, 0.1, 0.5):
    CFG['eps'] = e
    Cf, a, b, k, _ = build(pred, z, gt, cand, 'nwd')
    logK = -Cf / e; f = np.zeros(M); g = np.zeros(G + 1); r = []
    for it in range(200):
        f = np.log(a) - logsumexp(logK + g[None], axis=1); g = np.log(b) - logsumexp(logK + f[:, None], axis=0)
        r.append(float(np.abs(np.exp(logK + f[:, None] + g[None]).sum(1) - a).sum()))
    res_curves[str(e)] = r
CFG['eps'] = 0.1
OUT['sinkhorn_residual_curve'] = dict(M=int(M), G=G, s=s, curves=res_curves)

json.dump(OUT, open('tc_e_small.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in OUT.items() if k not in ('curves', 'sinkhorn_residual_curve')}, indent=1, default=float))
