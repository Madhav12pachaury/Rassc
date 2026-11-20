#!/usr/bin/env python3
"""
RAASC: Risk-Aware Adaptive Semantic Cache for a healthcare voice-assistant edge gateway.

Seeded simulator that re-uses the workload and the latency / power parameters of the
baseline edge-cloud study (Montoya Benitez et al. 2025 measurements: t_c = 0.25 s,
t_i = 4.77 s, P = 3.15 W) and compares:

    cloud   : every query goes to the cloud
    ttl     : class-aware TTL semantic cache (static 7 d, semi-static 6 h, volatile 0)
    raasc   : Risk-Aware Adaptive Semantic Cache (this work)

Usage:  python raasc_sim.py --seeds 20 --out results.json
Needs:  numpy only.
"""
import argparse, json, time
import numpy as np

DAY = 86400.0
K, TOPICS, DIM, ZIPF_S = 400, 40, 64, 1.1
MIX = (0.60, 0.25, 0.15)                       # static / semi-static / volatile
INTERVAL = np.array([30 * DAY, 1 * DAY, 300.0])  # modelled update interval per class
TTL = np.array([7 * DAY, 6 * 3600.0, 0.0])      # baseline class-aware TTL
SEV = np.array([1.0, 2.0, 5.0])                 # assumed clinical severity weights
RATE, HORIZON = 2 / 60.0, 7 * DAY               # 2 queries/min, 7 days
TC, TI, POWER, TIMEOUT = (0.25, 0.24), (4.77, 1.95), 3.15, 2.0
SMIN, NBIN = 0.50, 10                           # candidate floor, calibration bins (0.05 wide)


def lognorm(rng, mean, sd, size):
    s2 = np.log(1 + (sd / mean) ** 2)
    return rng.lognormal(np.log(mean) - s2 / 2, np.sqrt(s2), size)


class World:
    """One seeded week of traffic, intents, updates and link state (shared by all policies)."""

    def __init__(self, seed):
        r = np.random.default_rng(seed)
        topic = r.normal(size=(TOPICS, DIM)); topic /= np.linalg.norm(topic, axis=1, keepdims=True)
        own = r.normal(size=(K, DIM)); own /= np.linalg.norm(own, axis=1, keepdims=True)
        v = np.sqrt(0.6) * topic[np.arange(K) % TOPICS] + np.sqrt(0.4) * own
        self.V = v / np.linalg.norm(v, axis=1, keepdims=True)
        self.cls = r.choice(3, size=K, p=MIX)
        rank = r.permutation(K) + 1
        pop = 1.0 / rank ** ZIPF_S; pop /= pop.sum()
        # update processes (Poisson per intent); u = uniform used to decide "pushed" under kappa
        self.U, ev_t, ev_i, ev_u = [], [], [], []
        for i in range(K):
            n = r.poisson(HORIZON / INTERVAL[self.cls[i]])
            ut = np.sort(r.uniform(0, HORIZON, n)); self.U.append(ut)
            if self.cls[i] != 2:
                ev_t.append(ut); ev_i.append(np.full(n, i)); ev_u.append(r.uniform(size=n))
        et, ei, eu = np.concatenate(ev_t), np.concatenate(ev_i), np.concatenate(ev_u)
        o = np.argsort(et); self.ev_t, self.ev_i, self.ev_u = et[o], ei[o], eu[o]
        # queries
        n = int(r.poisson(RATE * HORIZON))
        self.t = np.sort(r.uniform(0, HORIZON, n))
        self.intent = r.choice(K, size=n, p=pop)
        s = r.lognormal(np.log(0.45), 0.40, n)
        w = r.normal(size=(n, DIM)); w /= np.linalg.norm(w, axis=1, keepdims=True)
        q = self.V[self.intent] + s[:, None] * w
        self.Q = q / np.linalg.norm(q, axis=1, keepdims=True)
        self.tc = lognorm(r, *TC, n); self.ti = lognorm(r, *TI, n)
        self.u_out, self.u_aud = r.uniform(size=n), r.uniform(size=n)

    def nupd(self, i, a, b):
        u = self.U[i]
        return int(np.searchsorted(u, b, "right") - np.searchsorted(u, a, "right"))


class Calibrator:
    """Online, bin-wise estimate of P(wrong intent | cosine similarity), pessimistic upper bound."""

    def __init__(self, n0=8.0):
        mid = SMIN + 0.05 * (np.arange(NBIN) + 0.5)
        self.p0 = np.clip(2.0 * (0.95 - mid), 0.01, 0.8); self.n0 = n0
        self.w = np.zeros(NBIN); self.n = np.zeros(NBIN)

    def _b(self, s): return np.minimum(((s - SMIN) / 0.05).astype(int), NBIN - 1)

    def p_wrong(self, s):
        b = self._b(s); m = (self.w[b] + self.n0 * self.p0[b]) / (self.n[b] + self.n0)
        return np.minimum(1.0, m + np.sqrt(m * (1 - m) / (self.n[b] + self.n0 + 1)))

    def update(self, s, wrong):
        b = int(self._b(np.array([s]))[0]); self.n[b] += 1; self.w[b] += float(wrong)


def run(w, cfg):
    pol = cfg["policy"]; N = len(w.t); cap = cfg.get("cap", 100)
    tau, kappa, p_out = cfg.get("tau", 0.80), cfg.get("kappa", 0.0), cfg.get("outage", 0.0)
    rho, eps = cfg.get("rho", 0.05), cfg.get("audit", 0.02)
    use_cal, use_adapt, use_deg = cfg.get("calibrate", True), cfg.get("adaptive_rate", True), cfg.get("degraded", True)
    lam0 = 1.0 / INTERVAL; a0 = 2.0
    kap_c = np.array([kappa, kappa, 0.0])
    emb = np.zeros((cap, DIM)); eint = -np.ones(cap, int); tf = np.zeros(cap); last = np.zeros(cap)
    occ = np.zeros(cap, bool)
    nchg, tobs, lastf = np.zeros(K), np.zeros(K), -np.ones(K)
    cal = Calibrator()
    pm = (w.ev_u < kappa) if kappa > 0 else np.zeros(len(w.ev_t), bool)
    pt, pi_ = w.ev_t[pm], w.ev_i[pm]; ptr = 0
    out = dict(src=np.zeros(N, np.int8), lat=np.zeros(N), en=np.zeros(N), stale=np.zeros(N, bool),
               wrong=np.zeros(N, bool), flagged_stale=np.zeros(N, bool),
               flagged_wrong=np.zeros(N, bool), harm=np.zeros(N))   # src: 0 none,1 cache,2 flagged,3 cloud
    t_dec = 0.0
    # Offline oracle diagnostics: record the selected candidate at every decision.
    # These labels are used only for evaluation, never by the online calibrator.
    cand_diag = []

    def lam_hat(ei):
        if pol == "raasc" and use_adapt:
            return (a0 + nchg[ei]) / (a0 / lam0[w.cls[ei]] + tobs[ei])
        return lam0[w.cls[ei]]

    for k in range(N):
        t, q, I = w.t[k], w.Q[k], w.intent[k]
        while ptr < len(pt) and pt[ptr] <= t:                # event-driven invalidation
            occ[occ & (eint == pi_[ptr])] = False; ptr += 1
        up = w.u_out[k] >= p_out
        E = 0.0; cj = -1; cs = 0.0; cR = np.inf; hit = False
        if pol == "cloud":
            pass
        else:
            t0 = time.perf_counter()
            idx = np.flatnonzero(occ)
            if idx.size:
                sims = emb[idx] @ q
                if pol == "ttl":
                    ok = (sims >= tau) & ((t - tf[idx]) <= TTL[w.cls[eint[idx]]])
                    if ok.any():
                        j = np.flatnonzero(ok)[sims[ok].argmax()]; cj, cs, hit = idx[j], sims[j], True
                else:
                    m = sims >= SMIN
                    if m.any():
                        ii, ss = idx[m], sims[m]; ei = eint[ii]
                        lam = lam_hat(ei) * (1 - kap_c[w.cls[ei]])
                        ps = 1 - np.exp(-lam * (t - tf[ii]))
                        pw = cal.p_wrong(ss) if use_cal else (ss < tau).astype(float)
                        R = SEV[w.cls[ei]] * (1 - (1 - ps) * (1 - pw))
                        j = R.argmin(); cj, cs, cR = ii[j], ss[j], R[j]; hit = cR <= rho
                        pw_selected = float(pw[j])
                        cand_diag.append((int(cal._b(np.array([cs]))[0]),
                                          pw_selected, bool(eint[cj] != I), bool(hit)))
            t_dec += time.perf_counter() - t0
        if hit:
            e = cj; wr = eint[e] != I; st = w.nupd(eint[e], tf[e], t) > 0
            out["src"][k], out["lat"][k], E = 1, w.tc[k], POWER * w.tc[k]
            out["wrong"][k], out["stale"][k] = wr, st
            if wr or st: out["harm"][k] = SEV[w.cls[I]]
            last[e] = t
            if pol == "raasc" and w.u_aud[k] < eps:          # shadow audit of a served hit
                E += POWER * w.ti[k]
                if use_cal: cal.update(cs, wr)
        elif up:
            lk = 0.0 if pol == "cloud" else w.tc[k]            # cloud-only has no cache lookup
            out["src"][k], out["lat"][k] = 3, lk + w.ti[k]; E = POWER * (lk + w.ti[k])
            if pol == "raasc":
                if cj >= 0 and use_cal: cal.update(cs, eint[cj] != I)
                if lastf[I] >= 0:
                    nchg[I] += w.nupd(I, lastf[I], t); tobs[I] += t - lastf[I]
                lastf[I] = t
            if pol == "cloud": admit = False
            elif pol == "ttl": admit = TTL[w.cls[I]] > 0
            else:
                lam = lam_hat(np.array([I]))[0] * (1 - kap_c[w.cls[I]])
                life = -np.log(1 - min(rho / SEV[w.cls[I]], 0.999)) / max(lam, 1e-12)
                admit = life >= cfg.get("min_life", 30.0)
            if admit:
                free = np.flatnonzero(~occ)
                if free.size: s = free[0]
                else:
                    ii = np.arange(cap); age = t - tf
                    if pol == "ttl": dead = age > TTL[w.cls[eint]]
                    else:
                        lam = lam_hat(eint) * (1 - kap_c[w.cls[eint]])
                        dead = SEV[w.cls[eint]] * (1 - np.exp(-lam * age)) > rho
                    s = ii[dead][age[dead].argmax()] if dead.any() else last.argmin()
                emb[s], eint[s], tf[s], last[s], occ[s] = q, I, t, t, True
        else:                                                 # link down, no valid cache hit
            E = POWER * ((0.0 if pol == "cloud" else w.tc[k]) + TIMEOUT)
            if pol == "raasc" and use_deg and cj >= 0 and w.cls[eint[cj]] != 2 and cR <= 4 * rho:
                out["src"][k], out["lat"][k] = 2, w.tc[k]; E = POWER * w.tc[k]
                # Record the hidden ground-truth status of every degraded-mode answer.
                # These labels are available in simulation only; a deployed gateway
                # would need later cloud resolution/audit to observe them.
                out["flagged_wrong"][k] = (eint[cj] != I)
                out["flagged_stale"][k] = (w.nupd(eint[cj], tf[cj], t) > 0)
        out["en"][k] = E
    s = out["src"]; ans = s > 0; uns = ~(out["stale"] | out["wrong"]); unsafe = out["stale"] | out["wrong"]
    pct = lambda x: 100.0 * float(np.mean(x))
    nflag = int(np.sum(s == 2))
    diag = {}
    for b in range(NBIN):
        rows = [x for x in cand_diag if x[0] == b]
        if rows:
            diag[f"cal_bin_{b+1}_n"] = len(rows)
            diag[f"cal_bin_{b+1}_pred"] = float(np.mean([x[1] for x in rows]))
            diag[f"cal_bin_{b+1}_empirical"] = 100.0 * float(np.mean([x[2] for x in rows]))
            diag[f"cal_bin_{b+1}_served_pct"] = 100.0 * float(np.mean([x[3] for x in rows]))
        else:
            diag[f"cal_bin_{b+1}_n"] = 0
            diag[f"cal_bin_{b+1}_pred"] = 0.0
            diag[f"cal_bin_{b+1}_empirical"] = 0.0
            diag[f"cal_bin_{b+1}_served_pct"] = 0.0
    return dict(hit=pct(s == 1), flagged=pct(s == 2), unanswered=pct(s == 0),
                stale=pct(out["stale"]), wrong=pct(out["wrong"]), unsafe=pct(unsafe),
                flagged_n=nflag,
                flagged_stale_pct=(100.0 * float(np.mean(out["flagged_stale"][s == 2])) if nflag else 0.0),
                flagged_wrong_pct=(100.0 * float(np.mean(out["flagged_wrong"][s == 2])) if nflag else 0.0),
                flagged_unsafe_pct=(100.0 * float(np.mean((out["flagged_stale"] | out["flagged_wrong"])[s == 2])) if nflag else 0.0),
                harm=1000.0 * float(np.mean(out["harm"])),
                lat=float(out["lat"][ans].mean()), energy=float(out["en"].mean()),
                dec_us=1e6 * t_dec / max(N, 1), **diag)


def ci(vals):
    a = np.array(vals); return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


CONFIGS = {
    "cloud":        dict(policy="cloud"),
    "ttl":          dict(policy="ttl"),
    "ttl_inv":      dict(policy="ttl", kappa=0.5),
    "raasc":        dict(policy="raasc"),
    "raasc_inv":    dict(policy="raasc", kappa=0.5),
    "abl_nocal":    dict(policy="raasc", kappa=0.5, calibrate=False),
    "abl_norate":   dict(policy="raasc", kappa=0.5, adaptive_rate=False),
    "out_cloud":    dict(policy="cloud", outage=0.10),
    "out_ttl":      dict(policy="ttl", outage=0.10),
    "out_raasc":    dict(policy="raasc", kappa=0.5, outage=0.10),
    "out_raasc_nd": dict(policy="raasc", kappa=0.5, outage=0.10, degraded=False),
}
for r_ in (0.02, 0.05, 0.10, 0.20, 0.40):
    CONFIGS[f"rho_{r_}"] = dict(policy="raasc", kappa=0.5, rho=r_)


def _one_seed(args):
    seed, names = args
    w = World(seed)
    return {n: run(w, CONFIGS[n]) for n in names}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--only", default=""); ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    names = [n for n in CONFIGS if (not a.only or n in a.only.split(","))]
    acc = {n: [] for n in names}
    import os, pickle
    for sd in range(a.seeds):                      # per-seed cache so an interrupted run can resume
        f = f"/tmp/raasc_seed_{sd}.pkl"
        if os.path.exists(f): rs = pickle.load(open(f, "rb"))
        else:
            rs = _one_seed((1000 + sd, names)); pickle.dump(rs, open(f, "wb"))
        for n in names: acc[n].append(rs[n])
        print("seed", sd, "done", flush=True)
    res = {n: {m: ci([r[m] for r in acc[n]]) for m in acc[n][0]} for n in names}
    json.dump(res, open(a.out, "w"), indent=1)
    for n in names:
        print(f"{n:14s}", " ".join(f"{m}={v[0]:.2f}±{v[1]:.2f}" for m, v in res[n].items()))
