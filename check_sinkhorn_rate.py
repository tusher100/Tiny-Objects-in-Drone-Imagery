import json, numpy as np
def dH(x, y): r = np.log(x) - np.log(y); return float(r.max() - r.min())
rng = np.random.default_rng(2); viol = 0; lam2s = []; emps = []; nchk = 0
for trial in range(200):
    m = rng.integers(3, 30); n1 = rng.integers(2, 8)
    a = rng.dirichlet(np.ones(m)); b = rng.dirichlet(np.ones(n1))
    eps = rng.choice([0.1, 0.3, 1.0]); C = rng.uniform(0, 1, (m, n1)) * rng.choice([0.2, 0.5, 1.0])
    K = np.exp(-C / eps); v = np.ones(n1)
    for _ in range(20000): u = a / (K @ v); v = b / (K.T @ u)
    vs = v.copy(); lk = np.log(K)
    lam = np.tanh((lk[:, None, :, None] + lk[None, :, None, :] - lk[:, None, None, :] - lk[None, :, :, None]).max() / 4)
    v = np.ones(n1); d = [dH(v, vs)]
    for l in range(60): u = a / (K @ v); v = b / (K.T @ u); d.append(dH(v, vs))
    for l, dl in enumerate(d):
        nchk += 1
        if dl > lam ** (2 * l) * d[0] * (1 + 1e-9) + 1e-12: viol += 1
    lam2s.append(lam ** 2)
    rates = [d[l + 1] / d[l] for l in range(len(d) - 1) if d[l] > 1e-10]
    if rates: emps.append(np.median(rates))
out = dict(trials=200, checks=nchk, violations=viol, median_lam2=float(np.median(lam2s)), median_empirical_rate=float(np.median(emps)))
json.dump(out, open('check_D_fixed.json', 'w'), indent=1); print(out)
