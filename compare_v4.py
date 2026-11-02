"""Round-4 training-free analyses (CPU, numpy).

1. Annotation noise: the ground-truth boxes, not the predictions, are perturbed (1-pixel translation, or a
   log-normal jitter of width and height). Unlike Corollary 3, the candidate sets S change as well.
2. Sensitivity to the Gaussian-Wasserstein scale kappa: positive-set change under absolute (1 px) and
   size-relative (s/8) translations of the predictions, positives per object and soft targets.
3. Perturbation directions: plan change (fixed marginals, converged plans) and positive-set change under
   random per-box directions, one common direction, and radial shifts away from / towards the nearest object.
"""
import json, time, numpy as np
import compare_assigners as CA
from compare_assigners import ANC, ot_candidates, ot_problem, sinkhorn, IMG
from compare_v2 import run, gen_object, scene_sep, se, shift_all

def pairs_change(a1, a2):
    p1, p2 = set(a1.items()), set(a2.items())
    return len(p1 ^ p2) / max(1, len(p1 | p2))

# ------------------------------------------------------------------ 1. annotation noise
def annotation_noise(n_scenes=30, sizes=(4, 8, 16, 32, 64, 128), seed=200,
                     rules=('ot_nwd', 'ot_iou', 'ot_kadapt', 'simota', 'atss', 'tal')):
    out = {}
    for s in sizes:
        rng = np.random.default_rng(seed + s)
        acc = {r: dict(shift=[], scale=[]) for r in rules}
        for _ in range(n_scenes):
            gt, sz = scene_sep(s, rng); pred, z, _ = gen_object(gt, sz, rng)
            gt_t = shift_all(gt, rng, 1.0)
            c = (gt[:, :2] + gt[:, 2:]) / 2; wh = (gt[:, 2:] - gt[:, :2]) * np.exp(rng.normal(0, 0.1, (len(gt), 2)))
            gt_s = np.concatenate([c - wh / 2, c + wh / 2], 1)
            for r in rules:
                a1, _, _ = run(r, pred, z, gt)
                acc[r]['shift'].append(pairs_change(a1, run(r, pred, z, gt_t)[0]))
                acc[r]['scale'].append(pairs_change(a1, run(r, pred, z, gt_s)[0]))
        out[s] = {r: {k: [float(np.mean(v)), se(v)] for k, v in d.items()} for r, d in acc.items()}
        print('annot', s, {r: (round(out[s][r]['shift'][0], 3), round(out[s][r]['scale'][0], 3)) for r in rules}, flush=True)
    return out

# ------------------------------------------------------------------ 2. kappa sweep
def kappa_sweep(n_scenes=30, sizes=(4, 16, 64), kappas=(3.2, 6.4, 12.8, 25.6, 51.2), seed=300):
    out = {}
    for s in sizes:
        rng = np.random.default_rng(seed + s)
        acc = {kp: dict(abs=[], rel=[], npos=[], tgt=[]) for kp in kappas}
        for _ in range(n_scenes):
            gt, sz = scene_sep(s, rng); pred, z, _ = gen_object(gt, sz, rng)
            p_abs = shift_all(pred, rng, 1.0); p_rel = shift_all(pred, rng, s / 8)
            for kp in kappas:
                CA.KAPPA = kp
                a1, t1, _ = run('ot_nwd', pred, z, gt)
                acc[kp]['abs'].append(pairs_change(a1, run('ot_nwd', p_abs, z, gt)[0]))
                acc[kp]['rel'].append(pairs_change(a1, run('ot_nwd', p_rel, z, gt)[0]))
                cnt = np.bincount(list(a1.values()), minlength=len(gt))
                acc[kp]['npos'].append(float(cnt.mean()))
                if t1: acc[kp]['tgt'].append(float(np.mean(list(t1.values()))))
            CA.KAPPA = 12.8
        out[s] = {str(kp): {k: [float(np.mean(v)), se(v)] for k, v in d.items()} for kp, d in acc.items()}
        print('kappa', s, {kp: (round(v['abs'][0], 3), round(v['rel'][0], 3), round(v['npos'][0], 2)) for kp, v in out[s].items()}, flush=True)
    return out

# ------------------------------------------------------------------ 3. perturbation directions
def directions(n_scenes=20, sizes=(4, 16, 64), seed=400):
    out = {}
    for s in sizes:
        rng = np.random.default_rng(seed + s)
        acc = {d: dict(plan=[], pos=[]) for d in ('random', 'common', 'outward', 'inward')}
        for _ in range(n_scenes):
            gt, sz = scene_sep(s, rng); pred_all, z_all, _ = gen_object(gt, sz, rng)
            cidx, cand = ot_candidates(gt); pred, z = pred_all[cidx], z_all[cidx]
            Cf, a, b, k, _ = ot_problem(pred, z, gt, cand)
            P = sinkhorn(Cf, a, b, iters=1000); r = np.maximum(np.round(k), 1).astype(int)
            gc = (gt[:, :2] + gt[:, 2:]) / 2; pc = (pred[:, :2] + pred[:, 2:]) / 2
            near = np.argmin(((pc[:, None] - gc[None]) ** 2).sum(-1), 1)
            v = pc - gc[near]; nv = np.linalg.norm(v, axis=1, keepdims=True)
            rnd = rng.normal(size=v.shape); v = np.where(nv > 1e-9, v / np.maximum(nv, 1e-9), rnd / np.linalg.norm(rnd, axis=1, keepdims=True))
            ang = rng.uniform(0, 2 * np.pi); com = np.tile([np.cos(ang), np.sin(ang)], (len(pred), 1))
            ang2 = rng.uniform(0, 2 * np.pi, len(pred)); ran = np.stack([np.cos(ang2), np.sin(ang2)], 1)
            for name, d in (('random', ran), ('common', com), ('outward', v), ('inward', -v)):
                p2 = pred + np.concatenate([d, d], 1)
                Cf2, _, _, _, _ = ot_problem(p2, z, gt, cand, k=k)
                P2 = sinkhorn(Cf2, a, b, iters=1000)
                acc[name]['plan'].append(float(np.abs(P - P2).sum()))
                ch = 0; tot = 0
                for j in range(len(gt)):
                    S1 = set(np.argsort(-P[:, j + 1], kind='stable')[:r[j]]); S2 = set(np.argsort(-P2[:, j + 1], kind='stable')[:r[j]])
                    ch += len(S1 ^ S2); tot += len(S1 | S2)
                acc[name]['pos'].append(ch / max(1, tot))
        out[s] = {d: {k: [float(np.mean(v)), se(v)] for k, v in dd.items()} for d, dd in acc.items()}
        print('dir', s, {d: (round(v['plan'][0], 4), round(v['pos'][0], 3)) for d, v in out[s].items()}, flush=True)
    return out

if __name__ == '__main__':
    t0 = time.time(); OUT = {}
    OUT['directions'] = directions()
    OUT['kappa_sweep'] = kappa_sweep()
    OUT['annotation_noise'] = annotation_noise()
    OUT['seconds'] = time.time() - t0
    json.dump(OUT, open('compare_v4.json', 'w'), indent=1)
    print('done', OUT['seconds'])
