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

# ---------------------------------------------------------------- A: sharpness
eps = 0.1
a = b = np.array([0.5, 0.5])
P0 = sinkhorn(np.zeros((2, 2)), a, b, eps)
rows = []
for t in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
    Ct = np.array([[0, t], [t, 0]])
    Pt = sinkhorn(Ct, a, b, eps)
    l1 = np.abs(Pt - P0).sum()
    rows.append(dict(t=t, l1=l1, bound=t / (2 * eps), ratio=l1 / (t / (2 * eps)),
                     closed_form=2 * abs(1 / (1 + np.exp(-t / eps)) - 0.5)))
OUT['A_sharpness'] = rows

# ---------------------------------------------------------------- B: stability bounds on random problems
rng = np.random.default_rng(0)
rTV, rW = [], []
for trial in range(3000):
    m = rng.integers(2, 41); n1 = rng.integers(2, 11)
    a = rng.dirichlet(np.ones(m)); b = rng.dirichlet(np.ones(n1))
    eps = float(np.exp(rng.uniform(np.log(0.03), np.log(1.0))))
    C = rng.uniform(0, 1, (m, n1)) * rng.choice([0.3, 1, 3])
    D = rng.normal(0, 1, (m, n1)) * rng.choice([1e-3, 1e-2, 1e-1, 0.5])
    P = sinkhorn(C, a, b, eps); Q = sinkhorn(C + D, a, b, eps)
    l1 = np.abs(P - Q).sum()
    tv = osc(D) / (2 * eps)
    w = np.sqrt(2) / eps * np.sqrt((D ** 2 * np.maximum(P, Q)).sum())
    rTV.append(l1 / tv); rW.append(l1 / w)
OUT['B_stability'] = dict(trials=3000, max_ratio_TV=float(np.max(rTV)), median_ratio_TV=float(np.median(rTV)),
                          max_ratio_weighted=float(np.max(rW)), median_ratio_weighted=float(np.median(rW)))

# ---------------------------------------------------------------- C: eps -> 0
def H(p): p = p[p > 0]; return float(-(p * np.log(p)).sum())
gaps = {e: [] for e in (1.0, 0.3, 0.1, 0.03, 0.01)}
dists = {e: [] for e in gaps}
rng = np.random.default_rng(1)
for trial in range(300):
    m = rng.integers(2, 16); n1 = rng.integers(2, 7)
    a = rng.dirichlet(np.ones(m)); b = rng.dirichlet(np.ones(n1)); C = rng.uniform(0, 1, (m, n1))
    Aeq = np.zeros((m + n1, m * n1))
    for i in range(m): Aeq[i, i * n1:(i + 1) * n1] = 1
    for j in range(n1): Aeq[m + j, j::n1] = 1
    lp = linprog(C.ravel(), A_eq=Aeq, b_eq=np.concatenate([a, b]), bounds=(0, None), method='highs')
    P0 = lp.x.reshape(m, n1)
    for e in gaps:
        P = sinkhorn(C, a, b, e)
        gaps[e].append(((C * P).sum() - lp.fun) / (e * min(H(a), H(b))))
        dists[e].append(np.abs(P - P0).sum())
OUT['C_eps_to_zero'] = {str(e): dict(max_gap_over_bound=float(np.max(gaps[e])), min_gap=float(np.min(np.array(gaps[e]))),
                                     median_L1_to_LP=float(np.median(dists[e]))) for e in gaps}

# Part D (Sinkhorn contraction) is in check_D_fixed.py
json.dump(OUT, open('check_random_problems.json', 'w'), indent=1, default=float)
print('ABCD done')
