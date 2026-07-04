#!/usr/bin/env python3
"""Confusion analysis for a TPL, using the app's exact scoreOne() math.

Reads tools/features.json (raw measured feature vectors, written by
gen_templates.py) and scores each mudra's measured vector against every
template. Reports, per mudra: whether it ranks itself #1, its self score, and
the closest *other* mudra with its score (the confusion risk). A healthy TPL
has every mudra ranking itself #1 with a comfortable gap to the runner-up.
"""
import json, os, math, sys

FEATURE_ORDER = ["oI", "oM", "oR", "oP", "oT",
                 "dTI", "dTM", "dTR", "dTP", "sIM", "sMR", "sRP"]


def tol(k):
    return 0.32 if k[0] == "o" else 0.30


def score_one(f, tpl):
    err = n = 0
    for k in tpl:
        if k not in f:          # input vector may be sparse (e.g. fist)
            continue
        d = (f[k] - tpl[k]) / tol(k)
        err += d * d
        n += 1
    return 100 * math.exp(-err / n) if n else 0


def main():
    here = os.path.dirname(__file__)
    # argv[1] = templates json (default: raw features.json)
    # argv[2] = input vectors json (default: same as templates -> separability test)
    tpl_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "features.json")
    tpls = json.load(open(tpl_path))
    feat_path = sys.argv[2] if len(sys.argv) > 2 else tpl_path
    feats = json.load(open(feat_path))

    ids = list(feats.keys())
    print(f"{'mudra':<14}{'self':>6}  {'#1':<14}{'runnerup':<16}{'gap':>6}")
    print("-" * 60)
    problems = 0
    gaps = []
    for mid in ids:
        f = feats[mid]
        ranked = sorted(((score_one(f, tpls[t]), t) for t in tpls),
                        key=lambda x: -x[0])
        top_s, top_id = ranked[0]
        self_s = score_one(f, tpls[mid]) if mid in tpls else 0
        # runner-up = best scoring template that is not itself
        runner = next(((s, t) for s, t in ranked if t != mid), (0, "-"))
        ok = top_id == mid
        gap = self_s - runner[0]
        gaps.append(gap)
        flag = "" if ok else "  <-- MISCLASSIFIED as " + top_id
        if not ok:
            problems += 1
        print(f"{mid:<14}{self_s:6.1f}  {top_id:<14}{runner[1]:<16}{gap:6.1f}{flag}")
    print("-" * 60)
    print(f"self-#1 correct: {len(ids)-problems}/{len(ids)}   "
          f"min gap: {min(gaps):.1f}   avg gap: {sum(gaps)/len(gaps):.1f}")


if __name__ == "__main__":
    main()
