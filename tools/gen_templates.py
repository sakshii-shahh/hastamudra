#!/usr/bin/env python3
"""
Generate a data-driven TPL object for Hastamudra from the reference photos.

For every images/<id>.jpg whose <id> is a mudra key, this runs MediaPipe Hands
in static-image mode, extracts the 21 landmarks, and computes the SAME feature
vector the browser app computes in its JS `features()` function. It then prints a
`TPL = { ... }` object ready to paste into index.html.

The feature math here is a line-for-line port of the JS:

    hs = dist(lm[0], lm[9])
    open(a,b,c) = clamp01((angle(lm[a],lm[b],lm[c]) - 95) / 77)
    nd(a,b)     = dist(lm[a],lm[b]) / hs
    oI=open(5,6,8) oM=open(9,10,12) oR=open(13,14,16) oP=open(17,18,20)
    oT = clamp01((nd(4,5) - 0.35) / 0.75)
    dTI=nd(4,8) dTM=nd(4,12) dTR=nd(4,16) dTP=nd(4,20)
    sIM=nd(8,12) sMR=nd(12,16) sRP=nd(16,20)

dist() and angle() use 3D coords (x,y,z), exactly like the app.

Usage:
    python3 tools/gen_templates.py [images_dir]
"""

import sys
import os
import glob
import math
import json

import cv2
import numpy as np
import mediapipe as mp

# The 28 mudra ids the app knows about (keys of MUDRAS / TPL in index.html).
MUDRA_IDS = [
    "pataka", "tripataka", "ardhapataka", "kartarimukha", "mayura",
    "ardhachandra", "arala", "shukatunda", "mushti", "shikhara",
    "kapitha", "katakamukha", "suchi", "chandrakala", "padmakosha",
    "sarpashirsha", "mrigashirsha", "simhamukha", "kangula", "alapadma",
    "chatura", "bhramara", "hamsasya", "hamsapaksha", "sandamsha",
    "mukula", "tamrachuda", "trishula",
]

# Order used when emitting each template (matches app feature keys).
FEATURE_ORDER = ["oI", "oM", "oR", "oP", "oT",
                 "dTI", "dTM", "dTR", "dTP", "sIM", "sMR", "sRP"]

# --- Detection-failure corrections -------------------------------------------
# MediaPipe is accurate on the reference photos when fingers are visible, but it
# scatters landmarks when fingers are folded/occluded in flat 2D art, reading a
# curled finger as "open" (2D foreshortening). A LIVE webcam hand reads those
# same folded fingers correctly as closed, so for the handful of poses where the
# photo detection demonstrably failed we override the affected OPENNESS values
# with the canonical hand shape (and drop measured distances that came from the
# bad landmarks). Everything not listed here stays 100% data-driven.
# A value of None drops that feature from the template.
OVERRIDES = {
    # Full fist: landmarks scattered across occluded fingers, read as open.
    "mushti": {"oI": 0, "oM": 0, "oR": 0, "oP": 0, "oT": 0,
               "dTI": None, "dTM": None, "dTR": None, "dTP": None,
               "sIM": None, "sMR": None, "sRP": None},
    # Pataka with the ring finger folded; detector confused ring with pinky.
    "tripataka": {"oR": 0},
    # Index & middle extended, ring & little folded; detector scattered the
    # little-finger tip to the palm edge and read it as open.
    "ardhapataka": {"oR": 0, "oP": 0},
    # Ring finger folded down, others erect; detector read the ring as open.
    "kangula": {"oR": 0},
    # Middle & ring fingertips bend to touch the thumb; detector read them as
    # extended and put the middle tip far from the thumb.
    "simhamukha": {"oM": 0.25, "oR": 0.25, "dTM": 0.4},
}


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def angle(a, b, c):
    v1 = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    v2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    m1 = math.hypot(*v1)
    m2 = math.hypot(*v2)
    if not m1 or not m2:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (m1 * m2)))))


def clamp01(x):
    return max(0.0, min(1.0, x))


def features(lm):
    """lm: list of 21 (x, y, z) tuples. Returns dict of the 12 app features."""
    hs = dist(lm[0], lm[9]) or 1e-4

    def open_(a, b, c):
        return clamp01((angle(lm[a], lm[b], lm[c]) - 95.0) / 77.0)

    def nd(a, b):
        return dist(lm[a], lm[b]) / hs

    return {
        "oI": open_(5, 6, 8), "oM": open_(9, 10, 12),
        "oR": open_(13, 14, 16), "oP": open_(17, 18, 20),
        "oT": clamp01((nd(4, 5) - 0.35) / 0.75),
        "dTI": nd(4, 8), "dTM": nd(4, 12), "dTR": nd(4, 16), "dTP": nd(4, 20),
        "sIM": nd(8, 12), "sMR": nd(12, 16), "sRP": nd(16, 20),
    }


def detect_landmarks(hands, image_path):
    """Return list of 21 (x,y,z) tuples, or None if no hand found.

    Tries the raw image plus a horizontal flip and a light upscale, and keeps
    the detection with the highest hand-presence score — reference art can be
    stylized, so a couple of variants noticeably improves the hit rate.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, 0.0

    variants = [img, cv2.flip(img, 1)]
    # upscale small art so MediaPipe has more to work with
    h, w = img.shape[:2]
    if max(h, w) < 500:
        scale = 500.0 / max(h, w)
        big = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
        variants += [big, cv2.flip(big, 1)]

    best = None
    best_score = -1.0
    for v in variants:
        rgb = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            continue
        score = 0.0
        if res.multi_handedness:
            score = res.multi_handedness[0].classification[0].score
        if score > best_score:
            best_score = score
            best = [(p.x, p.y, p.z) for p in res.multi_hand_landmarks[0].landmark]
    return best, max(best_score, 0.0)


def fmt(v):
    # match the app's compact numeric style, 2 decimals, no trailing .0 noise
    return f"{round(v, 2):g}"


def main():
    images_dir = sys.argv[1] if len(sys.argv) > 1 else "images"
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.3,
    )

    results = {}
    missing = []
    for mid in MUDRA_IDS:
        path = os.path.join(images_dir, mid + ".jpg")
        if not os.path.exists(path):
            missing.append((mid, "no file"))
            continue
        lm, score = detect_landmarks(hands, path)
        if lm is None:
            missing.append((mid, "no hand detected"))
            continue
        results[mid] = (features(lm), score)

    hands.close()

    # ---- apply detection-failure overrides ----
    final = {}
    for mid in MUDRA_IDS:
        if mid not in results:
            continue
        f = dict(results[mid][0])
        ov = OVERRIDES.get(mid, {})
        for k, v in ov.items():
            if v is None:
                f.pop(k, None)
            else:
                f[k] = float(v)
        final[mid] = f

    # ---- emit the TPL object ----
    print("  // ===== Data-driven templates (generated by tools/gen_templates.py) =====")
    print("  // 12-feature fingerprint measured from each images/<id>.jpg via MediaPipe.")
    print("  // Openness values for a few folded-finger poses are corrected where the")
    print("  // 2D photo detection failed (see OVERRIDES in tools/gen_templates.py).")
    print("  const TPL = {")
    for mid in MUDRA_IDS:
        if mid not in final:
            continue
        f = final[mid]
        parts = [f"{k}:{fmt(f[k])}" for k in FEATURE_ORDER if k in f]
        tag = "  // corrected" if mid in OVERRIDES else ""
        print(f"    {mid+':':<14}{{{', '.join(parts)}}},{tag}")
    print("  };")

    # write final (post-override) templates for the validator
    with open(os.path.join(os.path.dirname(__file__), "features_final.json"), "w") as fh:
        json.dump({mid: {k: round(v, 4) for k, v in f.items()}
                   for mid, f in final.items()}, fh, indent=2)

    # ---- diagnostics to stderr so stdout stays paste-ready ----
    print("\n===== DIAGNOSTICS =====", file=sys.stderr)
    print(f"detected {len(results)}/{len(MUDRA_IDS)} mudras", file=sys.stderr)
    if missing:
        print("MISSING (kept out of generated TPL — keep the hand-tuned entry):",
              file=sys.stderr)
        for mid, why in missing:
            print(f"  - {mid}: {why}", file=sys.stderr)
    # dump raw features as JSON for inspection / re-use
    dump = {mid: {k: round(f[k], 4) for k in FEATURE_ORDER}
            for mid, (f, _) in results.items()}
    with open(os.path.join(os.path.dirname(__file__), "features.json"), "w") as fh:
        json.dump(dump, fh, indent=2)
    print("raw features written to tools/features.json", file=sys.stderr)


if __name__ == "__main__":
    main()
