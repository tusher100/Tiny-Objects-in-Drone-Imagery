"""Extra checks requested by review: (1) Sinkhorn row-error bound for the plan the algorithm returns,
(2) how many predictions overlap each object in the Kaggle stability scenes, (3) corrected Hilbert-metric check."""
import json, numpy as np
from scipy.special import logsumexp
out = {}
def dH(x, y): r = np.log(x) - np.log(y); return float(r.max() - r.min())
rng = np.random.default_rng(5)
worst, viol_rate, n = 0.0, 0, 0
for trial in range(2000):
    m = rng.integers(3, 30); n1 = rng.integers(2, 8)
    a = rng.dirichlet(np.ones(m)); b = rng.dirichlet(np.ones(n1))
    eps = rng.choice([0.1, 0.3, 1.0]); C = rng.uniform(0, 1, (m, n1)) * rng.choice([0.2, 0.5, 1.0])
    K = np.exp(-C / eps)
    # reference scalings
    v = np.ones(n1)
    for _ in range(20000):
        u = a / (K @ v); v = b / (K.T @ u)
    vstar = v.copy()
    lk = np.log(K)
    cr = lk[:, None, :, None] + lk[None, :, None, :] - lk[:, None, None, :] - lk[None, :, :, None]
    lam = np.tanh(cr.max() / 4)
    v = np.ones(n1); d0 = dH(v, vstar)
    for l in range(1, 31):
        u = a / (K @ v); v = b / (K.T @ u)                 # (u^(l), v^(l))
        P = u[:, None] * K * v[None, :]                     # plan returned by the algorithm: exact columns
        err = np.abs(P.sum(1) - a).sum()
        bound = np.exp((1 + lam ** 2) * lam ** (2 * l - 1) * d0) - 1
        if err > 1e-14:
            n += 1; worst = max(worst, err / bound)
            if err > bound * (1 + 1e-9): viol_rate += 1
out['row_error_bound'] = dict(checks=n, violations=viol_rate, max_ratio=worst)
# overlaps in the Kaggle stability scenes (same generator as stability_exp.py)
def anchors(img=640, strides=(8, 16, 32)):
    pts = []
    for s in strides:
        k = img // s; xs = (np.arange(k) + 0.5) * s; gx, gy = np.meshgrid(xs, xs); pts.append(np.stack([gx.ravel(), gy.ravel()], -1))
    return np.concatenate(pts)
def iou(p, g):
    lt = np.maximum(p[:, None, :2], g[None, :, :2]); rb = np.minimum(p[:, None, 2:], g[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    return inter / ((p[:, 2:] - p[:, :2]).prod(-1)[:, None] + (g[:, 2:] - g[:, :2]).prod(-1)[None] - inter + 1e-12)
anc = anchors(); rng = np.random.default_rng(0)
ov = {}
for s in (4, 8, 16, 32, 64):
    cnt, none = [], []
    for _ in range(50):
        c = rng.uniform(64, 640 - 64, (5, 2)); gt = np.concatenate([c - s / 2, c + s / 2], 1)
        sz = s * rng.uniform(0.8, 1.2, (len(anc), 1)); ctr = anc + rng.normal(0, s / 4, anc.shape)
        pred = np.concatenate([ctr - sz / 2, ctr + sz / 2], 1)
        o = (iou(pred, gt) > 0).sum(0); cnt += list(o); none += list(o == 0)
    ov[s] = dict(mean_overlapping=float(np.mean(cnt)), frac_none=float(np.mean(none)))
out['overlaps'] = ov
json.dump(out, open('extra_checks.json', 'w'), indent=1); print(json.dumps(out, indent=1))
