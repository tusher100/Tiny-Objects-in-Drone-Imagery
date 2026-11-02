"""Assignment-level comparison and ablation on synthetic YOLOv8-like scenes (CPU, numpy).

No detector is trained. Every rule sees the same anchors, predicted boxes and logits, and the script measures
which anchors become positives. Rules:
  ot_nwd      Algorithm 1 (entropic OT, Gaussian-Wasserstein cost, r_min candidates, dynamic k)
  ot_iou      Algorithm 1 with the IoU cost (and IoU in the dynamic k)
  ot_normin   Algorithm 1 without the r_min nearest candidates (radius rule only)
  ot_fixk     Algorithm 1 with a fixed demand k_j = 5
  greedy_nwd  same candidates, cost and dynamic k, but each object takes its r_j cheapest candidates
              (no transport plan; conflicts go to the cheaper object)
  ot_kadapt   Algorithm 1 with kappa_j = max(kappa, 0.5 sqrt(w_j h_j))
  simota      SimOTA as in YOLOX (centre-in-box or centre radius 2.5 strides, cost cls + 3(-log IoU),
              dynamic k = int(sum of top-10 IoUs))
  atss        ATSS (9 closest anchors per level, anchor boxes of side 8 x stride, IoU >= mean+std,
              centre inside the object)
  tal         task-aligned assigner of YOLOv8 (centre inside the object, metric p^0.5 IoU^6, top-10,
              conflicts to highest IoU)
Metrics per object size: fraction of objects without positive, positives per object, mean distance of
positive predictions to the object centre (object sizes), fraction of positive pairs that change after
a 1-pixel translation of every predicted box (full rule rerun), and mean soft target of positives.
Also: margin certificate of the positive-set theorem, stability in the marginals on random problems,
and Hall-condition feasibility in crowded scenes.
"""
import json, sys, time, numpy as np
from scipy.special import logsumexp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

EPS, KAPPA, LC, LL, RAD, RMIN, KMAX, BIG, ZETA = 0.1, 12.8, 1.0, 3.0, 2.5, 9, 10, 1e4, np.log(1e4 - 1)
IMG = 960

def anchors(img=IMG, strides=(8, 16, 32)):
    pts, st = [], []
    for s in strides:
        n = img // s; xs = (np.arange(n) + 0.5) * s
        gx, gy = np.meshgrid(xs, xs); pts.append(np.stack([gx.ravel(), gy.ravel()], -1)); st.append(np.full(n * n, s))
    return np.concatenate(pts), np.concatenate(st).astype(float)

ANC, STR = anchors()
LEVEL_SLICES = []
_o = 0
for s in (8, 16, 32):
    n = (IMG // s) ** 2; LEVEL_SLICES.append(slice(_o, _o + n)); _o += n

def phi(b): return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2, (b[..., 2] - b[..., 0]) / 2, (b[..., 3] - b[..., 1]) / 2], -1)
def w2(p, g): return np.linalg.norm(phi(p)[:, None] - phi(g)[None], axis=2)
def iou(p, g):
    lt = np.maximum(p[:, None, :2], g[None, :, :2]); rb = np.minimum(p[:, None, 2:], g[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    return inter / ((p[:, 2:] - p[:, :2]).prod(-1)[:, None] + (g[:, 2:] - g[:, :2]).prod(-1)[None] - inter + 1e-12)

def sinkhorn(C, a, b, eps=EPS, iters=50):
    logK = -C / eps; la, lb = np.log(a), np.log(b); f = np.zeros_like(a); g = np.zeros_like(b)
    for _ in range(iters):
        f = la - logsumexp(logK + g[None, :], axis=1)
        g = lb - logsumexp(logK + f[:, None], axis=0)
    return np.exp(logK + f[:, None] + g[None, :])

def sigmoid(z): return 1 / (1 + np.exp(-z))

# ------------------------------------------------------------------ transport-based rules
def ot_candidates(gt, rmin=RMIN, inbox=False):
    if inbox:
        cand = ((ANC[None, :, 0] > gt[:, None, 0]) & (ANC[None, :, 0] < gt[:, None, 2]) &
                (ANC[None, :, 1] > gt[:, None, 1]) & (ANC[None, :, 1] < gt[:, None, 3]))
        cidx = np.where(cand.any(0))[0]
        return cidx, cand[:, cidx].T
    gc = (gt[:, :2] + gt[:, 2:]) / 2
    nd = np.linalg.norm(gc[:, None] - ANC[None], axis=2) / STR[None]
    cand = nd <= RAD
    if rmin:
        top = np.argsort(nd, axis=1)[:, :rmin]
        for j in range(len(gt)): cand[j, top[j]] = True
    cidx = np.where(cand.any(0))[0]
    return cidx, cand[:, cidx].T

def ot_problem(pred, z, gt, cand, kind='nwd', fixk=None, kadapt=False, k=None):
    if kind == 'nwd':
        kap = np.full(len(gt), KAPPA)
        if kadapt: kap = np.maximum(KAPPA, 0.5 * np.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])))
        sim = np.exp(-w2(pred, gt) / kap[None])
    else:
        sim = iou(pred, gt)
    C = LC * (-np.clip(z, -ZETA, ZETA))[:, None] + LL * (1 - sim)
    C = np.where(cand, C, BIG)
    M = len(pred)
    if k is None:
        if fixk is not None:
            k = np.full(len(gt), float(fixk))
        else:
            simc = np.where(cand, sim, 0)
            k = np.clip(np.sort(simc, axis=0)[::-1][:min(KMAX, M)].sum(0), 1, KMAX)
        if k.sum() > 0.9 * M: k = k * (0.9 * M / k.sum())
    a = np.full(M, 1 / M); b = np.concatenate([[M - k.sum()], k]) / M
    Cf = np.concatenate([np.zeros((M, 1)), C], 1)
    return Cf, a, b, k, sim

def select_from_scores(score, r, higher=True):
    """score: M x n (higher better if higher). Each column takes its r_j best; conflicts to best score."""
    M, n = score.shape
    s = score if higher else -score
    order = np.argsort(-s, axis=0, kind='stable')
    Pm = np.zeros((M, n), bool)
    for j in range(n): Pm[order[:r[j], j], j] = True
    jbest = np.where(Pm, s, -np.inf).argmax(1); pos = Pm.any(1)
    return {int(i): int(jbest[i]) for i in np.where(pos)[0]}

def soft_targets(assign, p_cls, sim, alpha=0.5, beta=6.0):
    """eq. (4.5): t_ij = max_{P_j} sim * A_ij / max_{P_j} A, A = p^alpha sim^beta."""
    out = {}
    byobj = {}
    for i, j in assign.items(): byobj.setdefault(j, []).append(i)
    for j, idx in byobj.items():
        A = p_cls[idx] ** alpha * sim[idx, j] ** beta
        sm = sim[idx, j].max(); Am = A.max()
        for t, i in enumerate(idx): out[i] = float(sm * A[t] / Am) if Am > 0 else 0.0
    return out

def rule_ot(pred_all, z_all, gt, variant):
    rmin = 0 if variant == 'ot_normin' else RMIN
    cidx, cand = ot_candidates(gt, rmin, inbox=(variant == 'ot_inbox'))
    if len(cidx) == 0: return {}, {}
    pred, z = pred_all[cidx], z_all[cidx]
    kind = 'iou' if variant == 'ot_iou' else 'nwd'
    Cf, a, b, k, sim = ot_problem(pred, z, gt, cand, kind, fixk=5 if variant == 'ot_fixk' else None,
                                  kadapt=(variant == 'ot_kadapt'))
    r = np.maximum(np.round(k), 1).astype(int)
    if variant == 'greedy_nwd':
        loc = select_from_scores(Cf[:, 1:], r, higher=False)
    else:
        P = sinkhorn(Cf, a, b)
        loc = select_from_scores(P[:, 1:], r, higher=True)
    # objects with no candidate at all (possible only without r_min) get nothing
    assign = {int(cidx[i]): j for i, j in loc.items() if cand[i, j]}
    simfull = np.zeros((len(pred_all), len(gt))); simfull[cidx] = sim
    t = soft_targets(assign, sigmoid(z_all), simfull)
    return assign, t

def rule_simota(pred, z, gt):
    gc = (gt[:, :2] + gt[:, 2:]) / 2
    inbox = ((ANC[None, :, 0] > gt[:, None, 0]) & (ANC[None, :, 0] < gt[:, None, 2]) &
             (ANC[None, :, 1] > gt[:, None, 1]) & (ANC[None, :, 1] < gt[:, None, 3]))
    r = RAD * STR[None]
    incen = (np.abs(ANC[None, :, 0] - gc[:, 0:1]) < r) & (np.abs(ANC[None, :, 1] - gc[:, 1:2]) < r)
    fg = (inbox | incen).any(0); idx = np.where(fg)[0]
    io = iou(pred[idx], gt)
    cost = -np.log(sigmoid(z[idx]))[:, None] + 3 * (-np.log(io + 1e-8)) + 1e5 * (~(inbox[:, idx] & incen[:, idx])).T
    ktop = min(10, len(idx))
    k = np.maximum(np.sort(io, axis=0)[::-1][:ktop].sum(0).astype(int), 1)
    loc = select_from_scores(cost, k, higher=False)
    return {int(idx[i]): j for i, j in loc.items()}, {}

def rule_atss(pred, z, gt, topk=9):
    gc = (gt[:, :2] + gt[:, 2:]) / 2
    d = np.linalg.norm(gc[:, None] - ANC[None], axis=2)
    abox = np.concatenate([ANC - 4 * STR[:, None], ANC + 4 * STR[:, None]], 1)
    io_all = iou(abox, gt)                          # A x n
    assign_iou = {}
    for j in range(len(gt)):
        cands = np.concatenate([sl.start + np.argsort(d[j, sl])[:topk] for sl in LEVEL_SLICES])
        v = io_all[cands, j]; thr = v.mean() + v.std()
        inside = ((ANC[cands, 0] > gt[j, 0]) & (ANC[cands, 0] < gt[j, 2]) & (ANC[cands, 1] > gt[j, 1]) & (ANC[cands, 1] < gt[j, 3]))
        for i in cands[(v >= thr) & inside]:
            if i not in assign_iou or io_all[i, j] > assign_iou[i][1]: assign_iou[int(i)] = (j, io_all[i, j])
    return {i: v[0] for i, v in assign_iou.items()}, {}

def rule_tal(pred, z, gt, topk=10, alpha=0.5, beta=6.0):
    inbox = ((ANC[None, :, 0] - gt[:, None, 0] > 1e-9) & (gt[:, None, 2] - ANC[None, :, 0] > 1e-9) &
             (ANC[None, :, 1] - gt[:, None, 1] > 1e-9) & (gt[:, None, 3] - ANC[None, :, 1] > 1e-9)).T  # A x n
    idx = np.where(inbox.any(1))[0]
    if len(idx) == 0: return {}, {}
    io = iou(pred[idx], gt) * inbox[idx]
    p = sigmoid(z[idx])
    met = (p[:, None] ** alpha) * io ** beta * inbox[idx]
    sel = np.zeros_like(met, bool)
    for j in range(len(gt)):
        cj = np.where(inbox[idx, j])[0]
        if len(cj): sel[cj[np.argsort(-met[cj, j], kind='stable')[:topk]], j] = True
    assign, t = {}, {}
    for r_ in np.where(sel.any(1))[0]:
        js = np.where(sel[r_])[0]; j = js[np.argmax(io[r_, js])] if len(js) > 1 else js[0]
        assign[int(idx[r_])] = int(j)
    byobj = {}
    for i, j in assign.items(): byobj.setdefault(j, []).append(i)
    pos = {int(ii): r_ for r_, ii in enumerate(idx)}
    for j, ii in byobj.items():
        rows = [pos[i] for i in ii]
        mm = met[rows, j].max(); im = io[rows, j].max()
        for i, r_ in zip(ii, rows): t[i] = float(met[r_, j] / mm * im) if mm > 0 else 0.0
    return assign, t

RULES = ['ot_nwd', 'ot_iou', 'ot_normin', 'ot_inbox', 'ot_fixk', 'greedy_nwd', 'ot_kadapt', 'simota', 'atss', 'tal']

def run_rule(name, pred, z, gt):
    if name.startswith('ot_') or name == 'greedy_nwd': return rule_ot(pred, z, gt, name)
    return {'simota': rule_simota, 'atss': rule_atss, 'tal': rule_tal}[name](pred, z, gt)

def scene(s, rng, G=8):
    c = rng.uniform(max(64, s), IMG - max(64, s), (G, 2)); gt = np.concatenate([c - s / 2, c + s / 2], 1)
    A = len(ANC)
    sz = s * rng.uniform(0.8, 1.2, (A, 1)); ctr = ANC + rng.normal(0, s / 4, ANC.shape)
    pred = np.concatenate([ctr - sz / 2, ctr + sz / 2], 1)
    z = np.clip(rng.normal(-3, 1, A), -9.21, 9.21)
    ang = rng.uniform(0, 2 * np.pi, A); d = np.stack([np.cos(ang), np.sin(ang)], 1)
    pred2 = pred + np.concatenate([d, d], 1)
    return gt, pred, z, pred2

def main_compare(n_scenes=40, sizes=(4, 8, 16, 32, 64, 128)):
    res = {}
    for s in sizes:
        rng = np.random.default_rng(1000 + s)
        acc = {r: dict(nopos=[], npos=[], dist=[], flip=[], tgt=[]) for r in RULES}
        for _ in range(n_scenes):
            gt, pred, z, pred2 = scene(s, rng)
            gc = (gt[:, :2] + gt[:, 2:]) / 2; pc = (pred[:, :2] + pred[:, 2:]) / 2
            for r in RULES:
                a1, t1 = run_rule(r, pred, z, gt)
                a2, _ = run_rule(r, pred2, z, gt)
                cnt = np.bincount(list(a1.values()), minlength=len(gt)) if a1 else np.zeros(len(gt), int)
                acc[r]['nopos'].extend((cnt == 0).tolist()); acc[r]['npos'].extend(cnt.tolist())
                if a1: acc[r]['dist'].append(np.mean([np.linalg.norm(pc[i] - gc[j]) / s for i, j in a1.items()]))
                p1, p2 = set(a1.items()), set(a2.items())
                acc[r]['flip'].append(len(p1 ^ p2) / max(1, len(p1 | p2)))
                if t1: acc[r]['tgt'].append(np.mean(list(t1.values())))
        res[s] = {r: {k: (float(np.mean(v)) if len(v) else None) for k, v in d.items()} for r, d in acc.items()}
        print(s, {r: (round(res[s][r]['nopos'], 3), round(res[s][r]['npos'], 2), round(res[s][r]['flip'], 3)) for r in RULES}, flush=True)
    return res

# ------------------------------------------------------------------ margin certificate (Theorem: positive sets)
def certificate(n_scenes=40, sizes=(4, 8, 16, 32, 64)):
    out = {}
    for s in sizes:
        rng = np.random.default_rng(2000 + s)
        cert = cert_l1 = tot = viol = same_tot = 0; gaps = []; l1s = []; linf = []
        for _ in range(n_scenes):
            gt, pred_all, z_all, pred2_all = scene(s, rng)
            cidx, cand = ot_candidates(gt)
            pred, z, pred2 = pred_all[cidx], z_all[cidx], pred2_all[cidx]
            Cf, a, b, k, _ = ot_problem(pred, z, gt, cand)
            Cf2, _, _, _, _ = ot_problem(pred2, z, gt, cand, k=k)      # same marginals
            P = sinkhorn(Cf, a, b, iters=3000); P2 = sinkhorn(Cf2, a, b, iters=3000)
            l1 = np.abs(P - P2).sum(); l1s.append(l1)
            r = np.maximum(np.round(k), 1).astype(int)
            for j in range(len(gt)):
                col, col2 = P[:, j + 1], P2[:, j + 1]
                o = np.sort(col)[::-1]; g = o[r[j] - 1] - o[r[j]]
                dinf = np.abs(col - col2).max(); linf.append(dinf * len(col))
                gaps.append(g * len(col))                       # in units of 1/m
                tot += 1
                same = set(np.argsort(-col)[:r[j]]) == set(np.argsort(-col2)[:r[j]])
                same_tot += same
                if dinf < g / 2:
                    cert += 1
                    if not same: viol += 1
                if l1 < g: cert_l1 += 1
        out[s] = dict(columns=tot, certified=cert / tot, certified_l1=cert_l1 / tot, unchanged=same_tot / tot,
                      violations=viol, median_gap_times_m=float(np.median(gaps)),
                      median_colchange_times_m=float(np.median(linf)), median_l1=float(np.median(l1s)))
        print('cert', s, out[s], flush=True)
    return out

# ------------------------------------------------------------------ stability in the marginals (random problems)
def scal(K, a, b, tol=1e-14, it=20000):
    v = np.ones(K.shape[1])
    for t in range(it):
        u = a / (K @ v); v = b / (K.T @ u)
        if t % 20 == 0 and np.abs((u[:, None] * K * v[None]).sum(1) - a).sum() < tol: break
    return u[:, None] * K * v[None]

def marginal_check(n=2000, seed=7):
    rng = np.random.default_rng(seed); ratios = []; viol = 0
    for _ in range(n):
        m = rng.integers(2, 41); q = rng.integers(2, 11)
        eps = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
        C = rng.uniform(0, 1, (m, q)); a = rng.dirichlet(np.ones(m)); b = rng.dirichlet(np.ones(q))
        b2 = b * np.exp(rng.normal(0, 10 ** rng.uniform(-4, -1), q)); b2 /= b2.sum()
        K = np.exp(-C / eps)
        L = np.log(K); th = (L[:, :, None, None] + L[None, None] - L[:, None, None, :].transpose(0, 2, 1, 3) * 0)  # placeholder
        # projective diameter: max over i,i',j,j' of L_ij + L_i'j' - L_ij' - L_i'j
        Dm = L[:, None, :, None] + L[None, :, None, :] - L[:, None, None, :] - L[None, :, :, None]
        lam = np.tanh(Dm.max() / 4)
        dH = np.ptp(np.log(b) - np.log(b2))
        bound = 2 * np.tanh(dH / (4 * (1 - lam)))          # Theorem 5.6, total-variation form
        P = scal(K, a, b); P2 = scal(K, a, b2)
        obs = np.abs(P - P2).sum()
        ratios.append(obs / bound)
        if obs > bound * (1 + 1e-6) + 1e-9: viol += 1
    return dict(problems=n, violations=viol, max_ratio=float(np.max(ratios)), median_ratio=float(np.median(ratios)))

# ------------------------------------------------------------------ Hall condition in crowded scenes
def hall_check(n_scenes=200, G=50, seed=11):
    rng = np.random.default_rng(seed); fails = 0; ms = []
    for _ in range(n_scenes):
        ncl = rng.integers(2, 6); cc = rng.uniform(100, IMG - 100, (ncl, 2))
        c = cc[rng.integers(0, ncl, G)] + rng.normal(0, 40, (G, 2))
        s = np.exp(rng.uniform(np.log(4), np.log(48), (G, 1)))
        gt = np.concatenate([c - s / 2, c + s / 2], 1)
        cidx, cand = ot_candidates(gt)
        M = len(cidx); ms.append(M)
        k = np.full(G, float(KMAX))                         # worst case: every demand at its cap
        if k.sum() > 0.9 * M: k = k * (0.9 * M / k.sum())
        # bipartite flow: source -> objects (cap k_j), object -> candidate anchors (cap 1), anchor -> sink (cap 1)
        SC = 1000
        src, sink = 0, 1 + G + M
        rows, cols, caps = [], [], []
        for j in range(G):
            rows.append(src); cols.append(1 + j); caps.append(int(np.floor(k[j] * SC)))
            for i in np.where(cand[:, j])[0]: rows.append(1 + j); cols.append(1 + G + i); caps.append(SC)
        for i in range(M): rows.append(1 + G + i); cols.append(sink); caps.append(SC)
        Gm = csr_matrix((np.array(caps, np.int32), (rows, cols)), shape=(sink + 1, sink + 1))
        flow = maximum_flow(Gm, src, sink).flow_value
        if flow < sum(int(np.floor(x * SC)) for x in k): fails += 1
    return dict(scenes=n_scenes, objects=G, failures=fails, median_m=float(np.median(ms)))

if __name__ == '__main__':
    t0 = time.time(); OUT = {}
    OUT['marginal_check'] = marginal_check(); print('marg', OUT['marginal_check'], flush=True)
    OUT['hall'] = hall_check(); print('hall', OUT['hall'], flush=True)
    OUT['certificate'] = certificate();
    OUT['compare'] = main_compare()
    OUT['seconds'] = time.time() - t0
    json.dump(OUT, open('compare_assigners.json', 'w'), indent=1)
    print('done', OUT['seconds'])
