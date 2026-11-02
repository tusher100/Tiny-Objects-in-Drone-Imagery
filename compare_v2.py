"""Second round of assignment-level experiments requested by the review panel (CPU, numpy; no training).

1. Tables 4-5 rerun on all six sizes with Monte Carlo standard errors; TAL now uses CIoU as Ultralytics 8.3.0.
2. Object-aware prediction model: predictions of anchors near an object regress towards it, logits grow with IoU.
3. Crowded scenes (50 clustered objects of mixed size): OT vs the greedy rule, conflict rates, agreement.
4. Size-relative perturbations: plan change under a shift of s/8 for the NWD and IoU costs.
5. epsilon sweep in crowded scenes.
6. Margin vs flip: are columns with small selection margins the ones whose top-r set changes?
"""
import json, time, numpy as np
from scipy.stats import spearmanr
import compare_assigners as CA
from compare_assigners import (ANC, STR, iou, sigmoid, ot_candidates, ot_problem, sinkhorn, select_from_scores,
                               soft_targets, rule_simota, rule_atss, IMG, RAD, RMIN, KMAX, EPS)

# ------------------------------------------------------------------ TAL with CIoU (Ultralytics 8.3.0 iou_calculation)
def ciou(p, g):
    io = iou(p, g)
    pc = (p[:, None, :2] + p[:, None, 2:]) / 2; gc = (g[None, :, :2] + g[None, :, 2:]) / 2
    rho2 = ((pc - gc) ** 2).sum(-1)
    lt = np.minimum(p[:, None, :2], g[None, :, :2]); rb = np.maximum(p[:, None, 2:], g[None, :, 2:])
    c2 = ((rb - lt) ** 2).sum(-1) + 1e-9
    wp, hp = p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]; wg, hg = g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]
    v = (4 / np.pi ** 2) * (np.arctan(wg / hg)[None] - np.arctan(wp / hp)[:, None]) ** 2
    alpha = v / (v - io + 1 + 1e-9)
    return np.clip(io - (rho2 / c2 + v * alpha), 0, None)

def rule_tal(pred, z, gt, topk=10, alpha=0.5, beta=6.0):
    inbox = ((ANC[None, :, 0] - gt[:, None, 0] > 1e-9) & (gt[:, None, 2] - ANC[None, :, 0] > 1e-9) &
             (ANC[None, :, 1] - gt[:, None, 1] > 1e-9) & (gt[:, None, 3] - ANC[None, :, 1] > 1e-9)).T
    idx = np.where(inbox.any(1))[0]
    if len(idx) == 0: return {}, {}, 0
    io = ciou(pred[idx], gt) * inbox[idx]
    p = sigmoid(z[idx]); met = (p[:, None] ** alpha) * io ** beta * inbox[idx]
    sel = np.zeros_like(met, bool)
    for j in range(len(gt)):
        cj = np.where(inbox[idx, j])[0]
        if len(cj): sel[cj[np.argsort(-met[cj, j], kind='stable')[:topk]], j] = True
    conflicts = int((sel.sum(1) > 1).sum())
    assign, t = {}, {}
    for r_ in np.where(sel.any(1))[0]:
        js = np.where(sel[r_])[0]; j = js[np.argmax(io[r_, js])] if len(js) > 1 else js[0]
        assign[int(idx[r_])] = int(j)
    byobj = {}
    for i, j in assign.items(): byobj.setdefault(j, []).append(i)
    pos = {int(ii): r_ for r_, ii in enumerate(idx)}
    for j, ii in byobj.items():
        rows = [pos[i] for i in ii]; mm = met[rows, j].max(); im = io[rows, j].max()
        for i, r_ in zip(ii, rows): t[i] = float(met[r_, j] / mm * im) if mm > 0 else 0.0
    return assign, t, conflicts

# ------------------------------------------------------------------ transport rules with conflict counting
def select_count(score, r, higher=True):
    M, n = score.shape; s = score if higher else -score
    order = np.argsort(-s, axis=0, kind='stable'); Pm = np.zeros((M, n), bool)
    for j in range(n): Pm[order[:r[j], j], j] = True
    jbest = np.where(Pm, s, -np.inf).argmax(1); pos = Pm.any(1)
    return {int(i): int(jbest[i]) for i in np.where(pos)[0]}, int((Pm.sum(1) > 1).sum())

def rule_ot(pred_all, z_all, gt, variant, eps=EPS):
    cidx, cand = ot_candidates(gt, RMIN, inbox=(variant == 'ot_inbox'))
    if len(cidx) == 0: return {}, {}, 0
    pred, z = pred_all[cidx], z_all[cidx]
    kind = 'iou' if variant == 'ot_iou' else 'nwd'
    Cf, a, b, k, sim = ot_problem(pred, z, gt, cand, kind, fixk=5 if variant == 'ot_fixk' else None,
                                  kadapt=(variant == 'ot_kadapt'))
    r = np.maximum(np.round(k), 1).astype(int)
    if variant == 'greedy_nwd':
        loc, conf = select_count(Cf[:, 1:], r, higher=False)
    else:
        P = sinkhorn(Cf, a, b, eps=eps); loc, conf = select_count(P[:, 1:], r, higher=True)
    assign = {int(cidx[i]): j for i, j in loc.items() if cand[i, j]}
    simfull = np.zeros((len(pred_all), len(gt))); simfull[cidx] = sim
    return assign, soft_targets(assign, sigmoid(z_all), simfull), conf

def run(name, pred, z, gt, eps=EPS):
    if name in ('simota', 'atss'):
        a, t = {'simota': rule_simota, 'atss': rule_atss}[name](pred, z, gt); return a, t, 0
    if name == 'tal': return rule_tal(pred, z, gt)
    return rule_ot(pred, z, gt, name, eps)

RULES = ['ot_nwd', 'ot_iou', 'ot_normin', 'ot_inbox', 'ot_fixk', 'greedy_nwd', 'ot_kadapt', 'simota', 'atss', 'tal']

# ------------------------------------------------------------------ scene generators
def shift_all(pred, rng, delta):
    ang = rng.uniform(0, 2 * np.pi, len(pred)); d = np.stack([np.cos(ang), np.sin(ang)], 1) * np.asarray(delta).reshape(-1, 1)
    return pred + np.concatenate([d, d], 1)

def gen_anchor(gt, sizes, rng):
    """the original generator: every anchor predicts a box of the nearest object's size around itself"""
    gc = (gt[:, :2] + gt[:, 2:]) / 2
    near = np.argmin(((ANC[:, None] - gc[None]) ** 2).sum(-1), 1); s = sizes[near]
    sz = s[:, None] * rng.uniform(0.8, 1.2, (len(ANC), 1)); ctr = ANC + rng.normal(0, 1, ANC.shape) * (s[:, None] / 4)
    z = np.clip(rng.normal(-3, 1, len(ANC)), -9.21, 9.21)
    return np.concatenate([ctr - sz / 2, ctr + sz / 2], 1), z, s

def gen_object(gt, sizes, rng):
    """object-aware: anchors within 2.5 strides of an object regress towards it; logits grow with IoU"""
    pred, z, s = gen_anchor(gt, sizes, rng)
    gc = (gt[:, :2] + gt[:, 2:]) / 2
    d = np.sqrt(((ANC[:, None] - gc[None]) ** 2).sum(-1)) / STR[:, None]
    near = np.argmin(d, 1); close = d[np.arange(len(ANC)), near] <= RAD
    idx = np.where(close)[0]; j = near[idx]; sj = sizes[j]
    c = gc[j] + rng.normal(0, 1, (len(idx), 2)) * np.maximum(1.0, 0.15 * sj)[:, None]
    w = sj * np.exp(rng.normal(0, 0.1, len(idx))); h = sj * np.exp(rng.normal(0, 0.1, len(idx)))
    pred[idx] = np.stack([c[:, 0] - w / 2, c[:, 1] - h / 2, c[:, 0] + w / 2, c[:, 1] + h / 2], 1)
    io = iou(pred[idx], gt)[np.arange(len(idx)), j]
    z[idx] = np.clip(-4 + 6 * io + rng.normal(0, 0.5, len(idx)), -9.21, 9.21)
    return pred, z, s

def scene_sep(s, rng, G=8):
    c = rng.uniform(max(64, s), IMG - max(64, s), (G, 2))
    return np.concatenate([c - s / 2, c + s / 2], 1), np.full(G, float(s))

def scene_crowd(rng, G=50):
    ncl = rng.integers(2, 6); cc = rng.uniform(100, IMG - 100, (ncl, 2))
    c = np.clip(cc[rng.integers(0, ncl, G)] + rng.normal(0, 40, (G, 2)), 30, IMG - 30)
    s = np.exp(rng.uniform(np.log(4), np.log(48), G))
    return np.concatenate([c - s[:, None] / 2, c + s[:, None] / 2], 1), s

def se(v): v = np.asarray(v, float); return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

# ------------------------------------------------------------------ 1+2: separated scenes, both generators, all sizes
def separated(gen, n_scenes=40, sizes=(4, 8, 16, 32, 64, 128), seed=0):
    out = {}
    for s in sizes:
        rng = np.random.default_rng(seed + s)
        acc = {r: dict(nopos=[], npos=[], dist=[], flip=[], tgt=[]) for r in RULES}
        for _ in range(n_scenes):
            gt, sz = scene_sep(s, rng); pred, z, _ = gen(gt, sz, rng); pred2 = shift_all(pred, rng, 1.0)
            gc = (gt[:, :2] + gt[:, 2:]) / 2; pc = (pred[:, :2] + pred[:, 2:]) / 2
            for r in RULES:
                a1, t1, _ = run(r, pred, z, gt); a2, _, _ = run(r, pred2, z, gt)
                cnt = np.bincount(list(a1.values()), minlength=len(gt)) if a1 else np.zeros(len(gt), int)
                acc[r]['nopos'].append(float((cnt == 0).mean())); acc[r]['npos'].append(float(cnt.mean()))
                if a1: acc[r]['dist'].append(np.mean([np.linalg.norm(pc[i] - gc[j]) / s for i, j in a1.items()]))
                p1, p2 = set(a1.items()), set(a2.items())
                acc[r]['flip'].append(len(p1 ^ p2) / max(1, len(p1 | p2)))
                if t1: acc[r]['tgt'].append(np.mean(list(t1.values())))
        out[s] = {r: {k: ([float(np.mean(v)), se(v)] if len(v) else None) for k, v in d.items()} for r, d in acc.items()}
        print(gen.__name__, s, {r: (round(out[s][r]['nopos'][0], 3), round(out[s][r]['flip'][0], 3)) for r in ('ot_nwd', 'greedy_nwd', 'tal', 'simota')}, flush=True)
    return out

# ------------------------------------------------------------------ 3+5: crowded scenes
BUCKETS = [(0, 8), (8, 16), (16, 32), (32, 1e9)]
def crowded(n_scenes=60, seed=5, rules=('ot_nwd', 'greedy_nwd', 'ot_iou', 'ot_inbox', 'simota', 'atss', 'tal'), eps_list=(0.03, 0.1, 0.3, 1.0)):
    rng = np.random.default_rng(seed)
    acc = {r: dict(nopos={b: [] for b in range(4)}, npos=[], flip=[], conf=[], agree=[]) for r in rules}
    epsacc = {e: dict(flip=[], agree=[], nopos=[]) for e in eps_list}
    shared = []
    for _ in range(n_scenes):
        gt, sz = scene_crowd(rng); pred, z, _ = gen_object(gt, sz, rng); pred2 = shift_all(pred, rng, 1.0)
        cidx, cand = ot_candidates(gt); shared.append(float((cand.sum(1) > 1).mean()))
        res = {}
        for r in rules:
            a1, _, conf = run(r, pred, z, gt); a2, _, _ = run(r, pred2, z, gt); res[r] = a1
            cnt = np.bincount(list(a1.values()), minlength=len(gt)) if a1 else np.zeros(len(gt), int)
            for b, (lo, hi) in enumerate(BUCKETS):
                m = (sz >= lo) & (sz < hi)
                if m.any(): acc[r]['nopos'][b].append(float((cnt[m] == 0).mean()))
            acc[r]['npos'].append(float(cnt.mean()))
            p1, p2 = set(a1.items()), set(a2.items())
            acc[r]['flip'].append(len(p1 ^ p2) / max(1, len(p1 | p2)))
            acc[r]['conf'].append(conf / max(1, len(a1)))
        g = set(res['greedy_nwd'].items())
        for r in rules:
            p = set(res[r].items()); acc[r]['agree'].append(len(p & g) / max(1, len(p | g)))
        for e in eps_list:
            a1, _, _ = rule_ot(pred, z, gt, 'ot_nwd', eps=e); a2, _, _ = rule_ot(pred2, z, gt, 'ot_nwd', eps=e)
            p1, p2 = set(a1.items()), set(a2.items())
            epsacc[e]['flip'].append(len(p1 ^ p2) / max(1, len(p1 | p2)))
            epsacc[e]['agree'].append(len(p1 & g) / max(1, len(p1 | g)))
            cnt = np.bincount(list(a1.values()), minlength=len(gt)); epsacc[e]['nopos'].append(float((cnt == 0).mean()))
    out = {r: dict(nopos=[[float(np.mean(v)), se(v)] for v in d['nopos'].values()],
                   **{k: [float(np.mean(d[k])), se(d[k])] for k in ('npos', 'flip', 'conf', 'agree')}) for r, d in acc.items()}
    eps_out = {str(e): {k: [float(np.mean(v)), se(v)] for k, v in d.items()} for e, d in epsacc.items()}
    print('crowded', {r: (round(out[r]['agree'][0], 3), round(out[r]['flip'][0], 3), round(out[r]['conf'][0], 3)) for r in rules}, flush=True)
    print('eps', eps_out, flush=True)
    return dict(rules=out, eps=eps_out, shared_candidate_fraction=[float(np.mean(shared)), se(shared)], scenes=n_scenes)

# ------------------------------------------------------------------ 4: size-relative perturbations of the plan
def relative_shift(n_scenes=20, sizes=(4, 8, 16, 32, 64), seed=9):
    out = {}
    for s in sizes:
        rng = np.random.default_rng(seed + s); res = {'nwd': [], 'iou': []}
        for _ in range(n_scenes):
            gt, sz = scene_sep(s, rng); pred_all, z_all, _ = gen_anchor(gt, sz, rng)
            cidx, cand = ot_candidates(gt); pred, z = pred_all[cidx], z_all[cidx]
            pred2 = shift_all(pred, rng, s / 8)
            for kind in ('nwd', 'iou'):
                Cf, a, b, k, _ = ot_problem(pred, z, gt, cand, kind)
                Cf2, _, _, _, _ = ot_problem(pred2, z, gt, cand, kind, k=k)
                res[kind].append(np.abs(sinkhorn(Cf, a, b, iters=1500) - sinkhorn(Cf2, a, b, iters=1500)).sum())
        out[s] = {k: [float(np.mean(v)), se(v)] for k, v in res.items()}
        print('rel', s, out[s], flush=True)
    return out

# ------------------------------------------------------------------ 6: margins and flips
def margin_flip(n_scenes=40, sizes=(8, 32), seed=13):
    g_all, flip_all = [], []
    for s in sizes:
        rng = np.random.default_rng(seed + s)
        for _ in range(n_scenes):
            gt, sz = scene_sep(s, rng); pred_all, z_all, _ = gen_anchor(gt, sz, rng)
            cidx, cand = ot_candidates(gt); pred, z = pred_all[cidx], z_all[cidx]; pred2 = shift_all(pred, rng, 1.0)
            Cf, a, b, k, _ = ot_problem(pred, z, gt, cand); Cf2, _, _, _, _ = ot_problem(pred2, z, gt, cand, k=k)
            P = sinkhorn(Cf, a, b, iters=3000); P2 = sinkhorn(Cf2, a, b, iters=3000)
            r = np.maximum(np.round(k), 1).astype(int)
            for j in range(len(gt)):
                col, col2 = P[:, j + 1], P2[:, j + 1]; o = np.sort(col)[::-1]
                g_all.append((o[r[j] - 1] - o[r[j]]) * len(col))
                flip_all.append(float(set(np.argsort(-col)[:r[j]]) != set(np.argsort(-col2)[:r[j]])))
    g_all, flip_all = np.array(g_all), np.array(flip_all)
    q = np.quantile(g_all, [0.25, 0.5, 0.75])
    quart = [float(flip_all[(g_all >= lo) & (g_all < hi)].mean()) for lo, hi in zip([-1, *q], [*q, 1e9])]
    rho = spearmanr(g_all, flip_all).correlation
    print('margin', quart, rho, flush=True)
    return dict(columns=int(len(g_all)), flip_rate_by_margin_quartile=quart, spearman=float(rho), quartiles=q.tolist())

if __name__ == '__main__':
    t0 = time.time(); OUT = {}
    OUT['margin_flip'] = margin_flip()
    OUT['relative_shift'] = relative_shift()
    OUT['crowded'] = crowded()
    OUT['separated_anchor'] = separated(gen_anchor)
    OUT['separated_object'] = separated(gen_object, seed=100)
    OUT['seconds'] = time.time() - t0
    json.dump(OUT, open('compare_v2.json', 'w'), indent=1)
    print('done', OUT['seconds'])
