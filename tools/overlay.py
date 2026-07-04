#!/usr/bin/env python3
"""Render MediaPipe landmarks onto reference images into tools/overlays/ for visual QA."""
import os, sys, cv2, mediapipe as mp

ids = sys.argv[1:] or ["mushti", "tripataka", "alapadma", "mrigashirsha",
                       "simhamukha", "kangula", "sandamsha", "pataka"]
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
outdir = os.path.join(os.path.dirname(__file__), "overlays")
os.makedirs(outdir, exist_ok=True)

with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                    model_complexity=1, min_detection_confidence=0.3) as hands:
    for mid in ids:
        path = os.path.join("images", mid + ".jpg")
        img = cv2.imread(path)
        if img is None:
            print(f"{mid}: no file"); continue
        h, w = img.shape[:2]
        if max(h, w) < 500:
            s = 500.0 / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        got = "yes" if res.multi_hand_landmarks else "NO"
        if res.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, res.multi_hand_landmarks[0],
                                   mp_hands.HAND_CONNECTIONS)
        outp = os.path.join(outdir, mid + ".png")
        cv2.imwrite(outp, img)
        print(f"{mid}: detected={got} -> {outp}")
