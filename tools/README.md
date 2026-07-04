# Hastamudra — template tooling

Offline scripts that generate and validate the `TPL` (feature templates) object in
`../index.html`. They are **not** part of the deployed site — the site is just
`index.html` + `images/`. These are for regenerating the templates when the
reference photos change.

## Setup

```bash
pip install 'mediapipe==0.10.14' opencv-python-headless numpy
```

## Pipeline

1. **Generate templates** — runs MediaPipe Hands on every `images/<id>.jpg`,
   computes the exact same 12-feature vector the app's `features()` uses, applies
   the documented detection-failure corrections (`OVERRIDES`), and prints a
   paste-ready `const TPL = {…}`:

   ```bash
   python3 gen_templates.py images > tpl.txt      # paste into index.html
   ```

   Side effects: `features.json` (raw measured vectors) and
   `features_final.json` (post-override, i.e. what's in the app).

2. **Validate separability** — scores every template against every other using
   the app's exact `scoreOne()`; every mudra should rank itself #1 with a
   comfortable gap to the runner-up:

   ```bash
   python3 validate.py features_final.json
   ```

3. **Robustness** — perturbs each template with realistic live-camera jitter and
   checks the correct mudra still wins and clears the `ENTER` lock threshold.
   Use it to retune `ENTER/EXIT` in `index.html`:

   ```bash
   python3 robustness.py
   ```

4. **QA overlays** (optional) — draws detected landmarks onto the photos so you
   can eyeball where MediaPipe mis-detected folded fingers (output in
   `overlays/`, gitignored):

   ```bash
   python3 overlay.py mushti tripataka simhamukha
   ```

## Browser check (image rendering)

`verify_image.mjs` loads `index.html` over `file://` in headless Chromium and
asserts the reference image actually renders (guards the `#ref` display bug):

```bash
node verify_image.mjs ../index.html
```

## The OVERRIDES

MediaPipe is accurate on the reference photos when fingers are visible, but it
scatters landmarks on fully folded/occluded fingers in flat 2D art, reading a
curled finger as "open". A live 3D webcam hand reads those correctly as closed,
so for a few poses the affected **openness** values are set to the canonical hand
shape (see `OVERRIDES` in `gen_templates.py`): `mushti`, `tripataka`,
`ardhapataka`, `simhamukha`, `kangula`. Everything else is 100% measured.
