"""Third set of checks requested by the methodology reviewer (CPU, numpy).

a) truncated vs converged plan: fraction of columns whose top-r_j set selected from the 50-iteration plan differs
   from the one selected from the converged plan (assigner-faithful scenes)
b) a-priori entrywise certificate: |pi_ij - pi'_ij| <= m^{-1/2} d sqrt(2 beta_fg)/eps with d = lambda_l delta/kappa,
   so column j keeps its top-r_j set for every translation with delta < g_j eps kappa sqrt(m) / (2 lambda_l sqrt(2 beta_fg));
   report the distribution of this certified radius and check it on actual 1-px shifts (must never fail)
c) projective diameter of K for a masked scene: log10(1 - lambda)
"""
import json, numpy as np
import compare_assigners as CA
from compare_v2 import gen_anchor, scene_sep, shift_all
from compare_assigners import ot_candidates, ot_problem, sinkhorn, EPS, KAPPA, LL

def topr(col, r): return set(np.argsort(-col)[:r])

def run(n_scenes=40, sizes=(4, 8, 16, 32, 64), seed=21):
    out = {}
    for s in sizes:
        rng = np.random.default_rng(seed + s)
        cols = diff_trunc = 0; radii = []; cert_viol = 0; cert_checked = 0
        for _ in range(n_scenes):
            gt, sz = scene_sep(s, rng); pred_all, z_all, _ = gen_anchor(gt, sz, rng)
            cidx, cand = ot_candidates(gt); pred, z = pred_all[cidx], z_all[cidx]; m = len(cidx)
            Cf, a, b, k, _ = ot_problem(pred, z, gt, cand)
            P = sinkhorn(Cf, a, b, iters=3000); P50 = sinkhorn(Cf, a, b, iters=50)
            r = np.maximum(np.round(k), 1).astype(int); bfg = 1 - b[0]
            # a 0.01-px shift, small enough that some columns are certified, to test the certificate
            delta = 0.004
            pred2 = shift_all(pred, rng, delta)
            Cf2, _, _, _, _ = ot_problem(pred2, z, gt, cand, k=k); P2 = sinkhorn(Cf2, a, b, iters=3000)
            for j in range(len(gt)):
                col = P[:, j + 1]; o = np.sort(col)[::-1]; g = o[r[j] - 1] - o[r[j]]
                cols += 1; diff_trunc += topr(col, r[j]) != topr(P50[:, j + 1], r[j])
                rad = g * EPS * KAPPA * np.sqrt(m) / (2 * LL * np.sqrt(2 * bfg)); radii.append(rad)
                if delta < rad:
                    cert_checked += 1; cert_viol += topr(col, r[j]) != topr(P2[:, j + 1], r[j])
        rad = np.array(radii)
        out[s] = dict(columns=cols, trunc_differs=diff_trunc / cols, radius_median=float(np.median(rad)),
                      radius_q90=float(np.quantile(rad, 0.9)), radius_max=float(rad.max()),
                      certified_at_0p004=cert_checked / cols, certificate_violations=cert_viol)
        print(s, out[s], flush=True)
    return out

def lam_masked(seed=3, s=16):
    rng = np.random.default_rng(seed)
    gt, sz = scene_sep(s, rng); pred_all, z_all, _ = gen_anchor(gt, sz, rng)
    cidx, cand = ot_candidates(gt); Cf, a, b, k, _ = ot_problem(pred_all[cidx], z_all[cidx], gt, cand)
    L = -Cf / EPS
    # vartheta = max over i,i',j,j' of L_ij + L_i'j' - L_ij' - L_i'j = max_{j,j'} [max_i (L_ij - L_ij') + max_i' (L_i'j' - L_i'j)]
    q = Cf.shape[1]; best = 0.0
    for j in range(q):
        for jj in range(q):
            if j == jj: continue
            best = max(best, (L[:, j] - L[:, jj]).max() + (L[:, jj] - L[:, j]).max())
    # 1 - tanh(x) ~ 2 exp(-2x) for large x, with x = vartheta / 4
    return dict(vartheta=float(best), log10_one_minus_lambda=float(np.log10(2) - (best / 2) / np.log(10)))

if __name__ == '__main__':
    OUT = dict(trunc_and_certificate=run(), masked_lambda=lam_masked())
    print(OUT['masked_lambda'])
    json.dump(OUT, open('compare_v3.json', 'w'), indent=1)
