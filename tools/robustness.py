#!/usr/bin/env python3
"""Noise-robustness check for the final TPL.

A live webcam hand never reproduces a template exactly. This perturbs each
template with realistic Gaussian noise (openness and distance/spread jitter),
re-runs the app's exact classify()/scoreOne() math, and reports:
  - acc: how often the noisy pose still classifies as the correct mudra
  - lock%: how often the top score clears ENTER=76 AND margin>=MARGIN (would lock)
  - meanTop: average winning score (compare against ENTER=76)
This tells us whether the full-vector templates are still lockable in practice.
"""
import json, os, math
import numpy as np

ENTER, MARGIN = 76, 9
FEATURE_ORDER = ["oI", "oM", "oR", "oP", "oT",
                 "dTI", "dTM", "dTR", "dTP", "sIM", "sMR", "sRP"]

def tol(k):
    return 0.32 if k[0] == "o" else 0.30

def score_one(f, tpl):
    err = n = 0
    for k in tpl:
        if k not in f:
            continue
        d = (f[k] - tpl[k]) / tol(k)
        err += d * d
        n += 1
    return 100 * math.exp(-err / n) if n else 0

def classify(f, tpls):
    ranked = sorted(((score_one(f, tpls[t]), t) for t in tpls), key=lambda x: -x[0])
    (s1, id1), (s2, _) = ranked[0], ranked[1]
    return id1, s1, s1 - s2

def main():
    here = os.path.dirname(__file__)
    tpls = json.load(open(os.path.join(here, "features_final.json")))
    rng = np.random.default_rng(42)
    N = 400
    O_SIGMA, D_SIGMA = 0.14, 0.18   # realistic per-feature live jitter

    print(f"{'mudra':<14}{'acc':>6}{'lock%':>7}{'meanTop':>9}")
    print("-" * 40)
    accs, locks, tops = [], [], []
    worst = []
    for mid, tpl in tpls.items():
        ok = lock = 0
        tsum = 0.0
        for _ in range(N):
            f = {}
            for k, v in tpl.items():
                sig = O_SIGMA if k[0] == "o" else D_SIGMA
                nv = v + rng.normal(0, sig)
                if k[0] == "o":
                    nv = min(1.0, max(0.0, nv))
                else:
                    nv = max(0.0, nv)
                f[k] = nv
            cid, s1, margin = classify(f, tpls)
            if cid == mid:
                ok += 1
            if cid == mid and s1 >= ENTER and margin >= MARGIN:
                lock += 1
            tsum += s1
        acc, lockp, mt = 100*ok/N, 100*lock/N, tsum/N
        accs.append(acc); locks.append(lockp); tops.append(mt)
        if acc < 90 or lockp < 70:
            worst.append((mid, acc, lockp, mt))
        print(f"{mid:<14}{acc:6.0f}{lockp:7.0f}{mt:9.1f}")
    print("-" * 40)
    print(f"avg acc {sum(accs)/len(accs):.0f}%   avg lock {sum(locks)/len(locks):.0f}%   "
          f"avg meanTop {sum(tops)/len(tops):.1f}")
    if worst:
        print("\nweakest (acc<90 or lock<70):")
        for mid, a, l, m in worst:
            print(f"  {mid:<14} acc={a:.0f}% lock={l:.0f}% meanTop={m:.1f}")

if __name__ == "__main__":
    main()
