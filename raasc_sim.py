#!/usr/bin/env python3
"""
raasc_sim.py -- reference simulator for RAASC (extended version).

Re-implementation of the gateway logic described in the paper (Sections III-VI),
extended with the experiments requested in the review:
  * both harm conventions (true-intent severity and served-entry severity)
  * explicit, documented calibration prior p0(b) and prior-sensitivity runs
  * per-bin label bookkeeping (miss vs audit labels) and an oracle label for every
    decision, to study selection bias of the calibrator
  * flagged (degraded-mode) answers audited for staleness / wrong intent
  * a clinical deny-list with router recall r and a deny-category workload

Usage:  python raasc_sim.py --seeds 20 --out results.json
Only NumPy is required.
"""
import argparse, json, math, sys, time
import numpy as np
from multiprocessing import Pool

# ----------------------------------------------------------------- constants
CLS_STATIC, CLS_SEMI, CLS_VOL = 0, 1, 2
SIGMA = np.array([1.0, 2.0, 5.0])
IC = np.array([30 * 86400.0, 86400.0, 300.0])       # mean update interval per class
PUSH = np.array([0.5, 0.5, 0.0])                    # push coverage with invalidation
TTL = np.array([7 * 86400.0, 6 * 3600.0, 0.0])
HORIZON = 7 * 86400.0
T_LOOK, T_CLOUD, POWER, T_TIMEOUT = 0.25, 4.77, 3.15, 2.0
SMIN, NBIN, BIN_W = 0.50, 10, 0.05
N0, ALPHA0, EPS_AUDIT, LMIN = 8.0, 2.0, 0.02, 30.0
INTENT_COS = 0.68  # cosine between two intent centres of the same topic
RHO_SD = 0.10   # per-query spread of the paraphrase similarity to the intent centre


def prior_p0(variant="logistic"):
    """Prior wrong-intent probability p0(b) for the ten similarity bins."""
    mid = SMIN + BIN_W * (np.arange(NBIN) + 0.5)
    base = 1.0 / (1.0 + np.exp((mid - 0.75) / 0.04))
    base = np.clip(base, 0.005, 0.99)
    if variant == "logistic":
        return base
    if variant == "optimistic":      # prior four times lower
        return np.clip(base / 4.0, 0.001, 0.99)
    if variant == "pessimistic":     # prior four times higher
        return np.clip(base * 4.0, 0.005, 0.99)
    if variant == "flat":            # uninformative
        return np.full(NBIN, 0.5)
    raise ValueError(variant)


# ----------------------------------------------------------------- workload
def make_workload(seed, K=400, topics=40, D=64, rate_per_min=2.0, outage=0.0,
                  rate_scale=None, n_deny=0, deny_recall=1.0, prior_shift=None, rho_sd=RHO_SD, intent_cos=INTENT_COS):
    rng = np.random.default_rng(seed)
    per = K // topics
    T = rng.standard_normal((topics, D)); T /= np.linalg.norm(T, axis=1, keepdims=True)
    a = math.sqrt(intent_cos)
    V = np.empty((K, D))
    for i in range(K):
        n = rng.standard_normal(D); n -= (n @ T[i // per]) * T[i // per]; n /= np.linalg.norm(n)
        v = a * T[i // per] + math.sqrt(1 - a * a) * n
        V[i] = v / np.linalg.norm(v)
    cls = np.array([CLS_STATIC] * int(0.60 * K) + [CLS_SEMI] * int(0.25 * K) + [CLS_VOL] * int(0.15 * K))
    cls = cls[rng.permutation(K)]
    rank = rng.permutation(K)
    pop = (rank + 1.0) ** -1.1; pop /= pop.sum()

    # true update rates: default = class prior; optional shifts
    mean_int = IC[cls].copy()
    if prior_shift == "wrong":
        u = rng.random(K)
        mean_int = np.where(u < 0.3, mean_int / 10, np.where(u < 0.6, mean_int * 10, mean_int))
    upd, pushed = [], []
    for i in range(K):
        lam = 1.0 / mean_int[i]
        if prior_shift == "sudden":
            n1 = rng.poisson(lam * HORIZON / 2); n2 = rng.poisson(10 * lam * HORIZON / 2)
            t = np.sort(np.concatenate([rng.random(n1) * HORIZON / 2, HORIZON / 2 + rng.random(n2) * HORIZON / 2]))
        elif prior_shift == "drift":
            # rate lam*(1+9 t/H): thinning of a rate 10*lam process
            m = rng.poisson(10 * lam * HORIZON)
            t = np.sort(rng.random(m) * HORIZON)
            keep = rng.random(m) < (1 + 9 * t / HORIZON) / 10.0
            t = t[keep]
        else:
            t = np.sort(rng.random(rng.poisson(lam * HORIZON)) * HORIZON)
        upd.append(t)
        pushed.append(t[rng.random(len(t)) < PUSH[cls[i]]])
    N = rng.poisson(rate_per_min * HORIZON / 60.0)
    tq = np.sort(rng.random(N) * HORIZON)
    iq = rng.choice(K, size=N, p=pop)
    m_rho = math.sqrt(0.895)                       # E[rho_q]^2 = 0.895 = mean paraphrase cosine
    ab = m_rho * (1 - m_rho) / rho_sd ** 2 - 1.0   # Beta(a,b) with mean m_rho and sd rho_sd
    rq = np.clip(rng.beta(m_rho * ab, (1 - m_rho) * ab, N), 0.30, 0.999)
    M = rng.standard_normal((N, D)); M /= np.linalg.norm(M, axis=1, keepdims=True)
    Q = rq[:, None] * V[iq] + np.sqrt(1 - rq ** 2)[:, None] * M
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    fail = rng.random(N) < outage
    ua = rng.random(N)
    deny_set = np.zeros(K, bool)
    if n_deny:
        deny_set[rng.choice(K, n_deny, replace=False)] = True
    det = rng.random(N) < deny_recall
    return dict(K=K, cls=cls, V=V, upd=upd, pushed=pushed, N=N, t=tq, I=iq, Q=Q,
                fail=fail, ua=ua, deny=deny_set, det=det, mean_int=mean_int)


def first_after(arr, t):
    j = np.searchsorted(arr, t, side="right")
    return arr[j] if j < len(arr) else np.inf


# ----------------------------------------------------------------- simulator
def simulate(W, policy, rho=0.05, inv=False, C=100, thr=0.80, calib=True, adaptive=True,
             degraded=True, repl="default", prior="logistic", deny_list=False,
             deny_sev=10.0, tau_only=False, ttl_override=None, degr_factor=4.0):
    """policy in {'cloud','ttl','raasc'}"""
    K, cls, N = W["K"], W["cls"], W["N"]
    D = W["Q"].shape[1]
    p0 = prior_p0(prior)
    kap = PUSH if inv else np.zeros(3)
    ttl = TTL if ttl_override is None else np.array(ttl_override, float)

    E = np.zeros((C, D)); ci = np.full(C, -1); tf = np.zeros(C); lu = np.zeros(C)
    alive = np.zeros(C, bool); nupd = np.full(C, np.inf); npush = np.full(C, np.inf)
    freq = np.zeros(C); arc_t = np.zeros(C)
    nI = np.zeros(K); TI = np.zeros(K); lastf = np.full(K, -1.0)
    cw = np.zeros(NBIN); cn = np.zeros(NBIN)             # calibrator counts
    cw_miss = np.zeros(NBIN); cn_miss = np.zeros(NBIN)   # label bookkeeping
    cw_aud = np.zeros(NBIN); cn_aud = np.zeros(NBIN)
    # oracle (all j* decisions) and served-hit calibration bookkeeping
    or_n = np.zeros(NBIN); or_w = np.zeros(NBIN)
    sv_n = np.zeros(NBIN); sv_w = np.zeros(NBIN); sv_p = np.zeros(NBIN)

    hit = flag = unans = 0
    stale_n = wrong_n = unsafe_n = 0
    harm_true = harm_served = harm_deny = 0.0
    lat_sum = 0.0; lat_cnt = 0; energy = 0.0
    fl_stale = fl_wrong = fl_unsafe = 0
    cls_served = np.zeros(3); cls_unsafe = np.zeros(3); cls_sev_unsafe = np.zeros(3); cls_R = np.zeros(3)
    den_q = den_cache = den_unsafe = den_leak_unflagged = 0
    ttl_bad = 0

    Qm, tq, iq, fail, ua = W["Q"], W["t"], W["I"], W["fail"], W["ua"]
    dset, det = W["deny"], W["det"]
    upd, pushed = W["upd"], W["pushed"]

    for k in range(N):
        t = tq[k]; q = Qm[k]; I = iq[k]; cI = cls[I]
        isdeny = bool(dset[I])
        if isdeny: den_q += 1
        if policy == "cloud":
            if fail[k]:
                unans += 1; energy += POWER * T_TIMEOUT
            else:
                lat_sum += T_CLOUD; lat_cnt += 1; energy += POWER * T_CLOUD
            continue

        # pushed invalidations and (for TTL) expiry
        if inv:
            alive &= ~(npush <= t)
        if policy == "ttl":
            alive &= ~((t - tf) > ttl[cls[np.maximum(ci, 0)]])

        # deny-list
        if deny_list and isdeny and det[k]:
            # purge cached entries of deny intents, bypass cache
            if alive.any():
                alive &= ~dset[np.maximum(ci, 0)]
            if fail[k]:
                unans += 1; energy += POWER * (T_LOOK + T_TIMEOUT)
            else:
                lat_sum += T_LOOK + T_CLOUD; lat_cnt += 1; energy += POWER * (T_LOOK + T_CLOUD)
            continue

        energy += POWER * T_LOOK
        idx = np.nonzero(alive)[0]
        jstar = -1; Rj = np.inf; sj = 0.0
        if idx.size:
            sims = E[idx] @ q
            fl = thr if (policy == "ttl" or (policy == "raasc" and not calib)) else SMIN
            m = sims >= fl
            if m.any():
                cand = idx[m]; cs = sims[m]
                if policy == "ttl":
                    b = int(np.argmax(cs)); jstar = cand[b]; sj = cs[b]; Rj = 0.0
                else:
                    cc = ci[cand]; cl = cls[cc]
                    if adaptive:
                        lam = (nI[cc] + ALPHA0) / (TI[cc] + ALPHA0 * IC[cl])
                    else:
                        lam = 1.0 / IC[cl]
                    ps = 1.0 - np.exp(-(1 - kap[cl]) * lam * (t - tf[cand]))
                    if calib:
                        bi = np.minimum(((cs - SMIN) / BIN_W).astype(int), NBIN - 1)
                        mb = (N0 * p0[bi] + cw[bi]) / (N0 + cn[bi])
                        pw = np.minimum(1.0, mb + np.sqrt(mb * (1 - mb) / (cn[bi] + N0 + 1)))
                    else:
                        pw = np.zeros(cand.size)
                    pu = ps + pw - ps * pw
                    R = pu * SIGMA[cl]
                    b = int(np.argmin(R)); jstar = cand[b]; Rj = R[b]; sj = cs[b]
        served = False; flagged = False
        if policy == "ttl":
            served = jstar >= 0
        else:
            served = jstar >= 0 and Rj <= rho
        if jstar >= 0 and policy == "raasc":
            bj = min(int((sj - SMIN) / BIN_W), NBIN - 1) if calib else None
            wrongj = ci[jstar] != I
            if bj is not None:
                or_n[bj] += 1; or_w[bj] += wrongj
        if served:
            j = jstar
            lu[j] = t; freq[j] += 1
            hit += 1
            lat_sum += T_LOOK; lat_cnt += 1
            wrong = ci[j] != I
            stale = nupd[j] <= t
            u = wrong or stale
            wrong_n += wrong; stale_n += stale; unsafe_n += u
            cj = cls[ci[j]]
            harm_true += SIGMA[cI] * u; harm_served += SIGMA[cj] * u
            cls_served[cj] += 1; cls_unsafe[cj] += u; cls_sev_unsafe[cj] += SIGMA[cj] * u; cls_R[cj] += Rj
            if isdeny:
                den_cache += 1; den_unsafe += u; harm_deny += deny_sev * u
                if deny_list: den_leak_unflagged += 1
            if policy == "raasc" and calib:
                sv_n[bj] += 1; sv_w[bj] += wrong
                mb = (N0 * p0[bj] + cw[bj]) / (N0 + cn[bj])
                sv_p[bj] += min(1.0, mb + math.sqrt(mb * (1 - mb) / (cn[bj] + N0 + 1)))
            if policy == "raasc" and ua[k] < EPS_AUDIT and not fail[k]:
                energy += POWER * T_CLOUD
                if calib:
                    cn[bj] += 1; cw[bj] += wrong; cn_aud[bj] += 1; cw_aud[bj] += wrong
            continue

        # ---------------------------------------------------------- miss path
        if not fail[k]:
            lat_sum += T_LOOK + T_CLOUD; lat_cnt += 1; energy += POWER * T_CLOUD
            if policy == "raasc":
                if jstar >= 0 and calib:
                    cn[bj] += 1; cw[bj] += wrongj; cn_miss[bj] += 1; cw_miss[bj] += wrongj
                if lastf[I] >= 0:
                    nI[I] += np.searchsorted(upd[I], t, "right") - np.searchsorted(upd[I], lastf[I], "right")
                    TI[I] += t - lastf[I]
                lastf[I] = t
            # admission
            admit = True
            if policy == "ttl":
                admit = ttl[cI] > 0
            else:
                lam = (nI[I] + ALPHA0) / (TI[I] + ALPHA0 * IC[cI]) if adaptive else 1.0 / IC[cI]
                L = math.log(1.0 / (1.0 - min(rho / SIGMA[cI], 0.999))) / ((1 - kap[cI]) * lam)
                admit = L >= LMIN
            if admit:
                same = np.nonzero(alive & (ci == I))[0]
                free = np.nonzero(~alive)[0]
                if same.size:
                    s = same[np.argmin(tf[same])]          # refresh the entry of this intent
                elif free.size:
                    s = free[0]
                else:
                    ids = np.nonzero(alive)[0]
                    s = None
                    if policy == "raasc" and repl == "default":
                        cc = ci[ids]; cl = cls[cc]
                        lam = ((nI[cc] + ALPHA0) / (TI[cc] + ALPHA0 * IC[cl])) if adaptive else 1.0 / IC[cl]
                        pst = 1.0 - np.exp(-(1 - kap[cl]) * lam * (t - tf[ids]))
                        over = pst * SIGMA[cl] > rho
                        if over.any():
                            oi = ids[over]; s = oi[np.argmin(tf[oi])]
                    if s is None:
                        if repl == "lfu":
                            s = ids[np.argmin(freq[ids] * 1e9 + lu[ids])]
                        else:
                            s = ids[np.argmin(lu[ids])]
                E[s] = q; ci[s] = I; tf[s] = t; lu[s] = t; alive[s] = True; freq[s] = 1
                nupd[s] = first_after(upd[I], t)
                npush[s] = first_after(pushed[I], t) if inv else np.inf
            continue

        # ---------------------------------------------------- link down path
        if policy == "raasc" and degraded and jstar >= 0 and cls[ci[jstar]] != CLS_VOL and Rj <= degr_factor * rho:
            j = jstar
            flag += 1
            lat_sum += T_LOOK + T_TIMEOUT; lat_cnt += 1; energy += POWER * T_TIMEOUT
            wrong = ci[j] != I; stale = nupd[j] <= t
            fl_wrong += wrong; fl_stale += stale; fl_unsafe += (wrong or stale)
        else:
            unans += 1; energy += POWER * T_TIMEOUT

    n_ans = max(hit, 1)
    return dict(N=N, hit=100.0 * hit / N, flag=100.0 * flag / N, unans=100.0 * unans / N,
                stale=100.0 * stale_n / n_ans, wrong=100.0 * wrong_n / n_ans, unsafe=100.0 * unsafe_n / n_ans,
                harm=1000.0 * harm_true / N, harm_served=1000.0 * harm_served / N,
                lat=lat_sum / max(lat_cnt, 1), energy=energy / N,
                n_hit=hit, n_unsafe=unsafe_n, n_flag=flag, fl_stale=fl_stale, fl_wrong=fl_wrong, fl_unsafe=fl_unsafe,
                cls_served=cls_served.tolist(), cls_unsafe=cls_unsafe.tolist(),
                cls_sev_unsafe=cls_sev_unsafe.tolist(), cls_R=cls_R.tolist(),
                cn=cn.tolist(), cw=cw.tolist(), cn_miss=cn_miss.tolist(), cw_miss=cw_miss.tolist(),
                cn_aud=cn_aud.tolist(), cw_aud=cw_aud.tolist(),
                or_n=or_n.tolist(), or_w=or_w.tolist(), sv_n=sv_n.tolist(), sv_w=sv_w.tolist(), sv_p=sv_p.tolist(),
                p0=p0.tolist(), den_q=den_q, den_cache=den_cache, den_unsafe=den_unsafe,
                harm_deny=1000.0 * harm_deny / N)


# ----------------------------------------------------------------- experiments
def cfg_table():
    """name -> (workload kwargs, simulate kwargs)"""
    c = {}
    c["cloud"] = ({}, dict(policy="cloud"))
    c["ttl"] = ({}, dict(policy="ttl"))
    c["ttl_inv"] = ({}, dict(policy="ttl", inv=True))
    c["raasc"] = ({}, dict(policy="raasc"))
    c["raasc_inv"] = ({}, dict(policy="raasc", inv=True))
    for rho in (0.02, 0.10, 0.20, 0.40):
        c[f"raasc_inv_rho{rho}"] = ({}, dict(policy="raasc", inv=True, rho=rho))
    c["abl_nocal"] = ({}, dict(policy="raasc", inv=True, calib=False))
    c["abl_norate"] = ({}, dict(policy="raasc", inv=True, adaptive=False))
    # outage
    c["out_cloud"] = (dict(outage=0.10), dict(policy="cloud"))
    c["out_ttl"] = (dict(outage=0.10), dict(policy="ttl"))
    c["out_raasc_nodeg"] = (dict(outage=0.10), dict(policy="raasc", inv=True, degraded=False))
    c["out_raasc_deg"] = (dict(outage=0.10), dict(policy="raasc", inv=True, degraded=True))
    c["out_raasc_deg_f2"] = (dict(outage=0.10), dict(policy="raasc", inv=True, degr_factor=2.0))
    c["out_raasc_deg_f8"] = (dict(outage=0.10), dict(policy="raasc", inv=True, degr_factor=8.0))
    # extra baselines / replacement
    c["fixtau"] = ({}, dict(policy="ttl", ttl_override=[1e12, 1e12, 1e12]))
    c["fixtau_inv"] = ({}, dict(policy="ttl", inv=True, ttl_override=[1e12, 1e12, 1e12]))
    c["ttl_vd"] = ({}, dict(policy="ttl", ttl_override=[1.5 * 86400, 36 * 60, 0]))
    c["ttl_vd_inv"] = ({}, dict(policy="ttl", inv=True, ttl_override=[3.1 * 86400, 73 * 60, 0]))
    c["raasc_inv_lru"] = ({}, dict(policy="raasc", inv=True, repl="lru"))
    c["raasc_inv_lfu"] = ({}, dict(policy="raasc", inv=True, repl="lfu"))
    # estimator shifts
    for ps in ("wrong", "sudden", "drift"):
        c[f"{ps}_adapt"] = (dict(prior_shift=ps), dict(policy="raasc", inv=True))
        c[f"{ps}_prior"] = (dict(prior_shift=ps), dict(policy="raasc", inv=True, adaptive=False))
        c[f"{ps}_ttl_inv"] = (dict(prior_shift=ps), dict(policy="ttl", inv=True))
    # calibration prior sensitivity
    for pv in ("optimistic", "pessimistic", "flat"):
        c[f"prior_{pv}"] = ({}, dict(policy="raasc", inv=True, prior=pv))
    # deny-list experiment: 40 deny-category intents, harm weight 10 for them
    for rec in (0.0, 0.7, 0.9, 1.0):
        c[f"deny_r{rec}"] = (dict(n_deny=40, deny_recall=rec), dict(policy="raasc", inv=True, deny_list=rec > 0))
    c["deny_ttl"] = (dict(n_deny=40, deny_recall=1.0), dict(policy="ttl", inv=True, deny_list=True))
    c["deny_ttl_nolist"] = (dict(n_deny=40, deny_recall=1.0), dict(policy="ttl", inv=True, deny_list=False))
    return c


def run_one(args):
    name, seed = args
    wk, sk = cfg_table()[name]
    W = make_workload(seed, **wk)
    r = simulate(W, **sk)
    return name, seed, r


def fixtau_runs(seed):
    """Fixed-threshold semantic cache without TTL: serves best candidate >= 0.80, never expires."""
    out = {}
    for inv in (False, True):
        W = make_workload(seed)
        out[inv] = simulate(W, "ttl", inv=inv, ttl_override=[1e12, 1e12, 1e12])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--only", default=None, help="comma list of config names")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    names = list(cfg_table())
    if a.only:
        names = a.only.split(",")
    jobs = [(n, s) for n in names for s in range(a.seeds)]
    res = {}
    t0 = time.time()
    with Pool(a.procs) as pool:
        for i, (n, s, r) in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1)):
            res.setdefault(n, {})[s] = r
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(jobs)}  {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    json.dump(res, open(a.out, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("done", time.time() - t0)
