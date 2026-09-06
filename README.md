# AudioReactiveVisuals

Audio-reactive video generator. Input: audio → output: video where the camera
**flies forward forever** and **punches in on every kick**, with the punch
strength driven by low-frequency (<150 Hz) energy. Snares whisper, kicks punch.

Reference: `reference_video.mp4` (style/behavior only — never overwrite it).

## How it works

```
audio (librosa) → per-beat kick/snare strengths → beat canvases (Z-Image) → frames → ffmpeg
```

**Beat chain (default, `--beat-mode chain`).** One canvas per beat (at 200 BPM
that's a fresh image every 0.3 s). Each canvas is img2img'd from the *previous
beat's zoomed end view* (rebase), so the flight zoom **never steps back**.
Inside a beat the zoom is linear plus a 2-frame kick step:

```
z_beat = min(--zmax, --drift + --kick-gain * kick^1.5 + --snare-gain * snare)
```

- `kick` = <150 Hz spectral-flux envelope (max near the beat), `snare` = 150–500 Hz.
- Seams hidden with a 2-frame dissolve (`--beat-blend`), per-beat palette
  locked with a LAB color match, opening blooms from white (`--intro-fade`).
- Beat 0 comes from `--init-image` (a proven canvas) or auto-picked from
  `--seed-tries` txt2img candidates.
- Framing is texture-seeking: each beat crops the most detailed window, so
  the flight never gets stranded in flat sky (plus: put texture in every
  prompt's sky).
- Grade tames neon extremes + highlight rolloff + S-curve (skip: `--no-grade`).

Legacy `--beat-mode cycle` (crossfading keyframes) is kept for `--no-diffusion`
pipeline tests only — its blend resets cause visible zoom-back.

## Project layout

```
generate.py            # thin entry shim (all logic in the package below)
audioreactive/         # the actual codebase
  config.py            # models, prompts constants, ALL tuning numbers
  audio.py             # tempo / onsets / kick-snare bands / beat grid
  camera.py            # rebase crops, texture-seeking + clamped pans
  color.py             # LAB match, tame_blue, anchor_pull, grade, upscale
  diffusion.py         # Z-Image txt2img/img2img canvases (GPU only)
  render.py            # procedural mocks, beat-chain + legacy cycle renderers
  cli.py               # argparse, chain/cycle orchestration, run metadata
run_*.sh               # SLURM jobs: sample test, full HD, low-res iteration
scripts/               # dev tools: ztest.py (model kick test), check_drift.py
tests/                 # CPU-only pytest suite (camera/color/cli/audio)
```

`python -m audioreactive` works too (same flags). Tuning numbers live in
`audioreactive/config.py` — change them there, never as magic literals.

## Setup

Conda env `nibbler` (`/home/d.ramos/miniconda/envs/nibbler`: torch+CUDA,
diffusers 0.40, librosa, opencv). On this cluster, always:

```bash
export LD_LIBRARY_PATH=/home/d.ramos/miniconda/envs/nibbler/lib:$LD_LIBRARY_PATH
```

(without it, the ohpc `libstdc++` breaks scipy/diffusers imports). There is no
system ffmpeg — `generate.py` falls back to the imageio-ffmpeg binary.
Model `Tongyi-MAI/Z-Image-Turbo` is cached in `~/.cache/huggingface/hub`
(reused offline via `HF_HUB_OFFLINE=1`); the login node has **no GPU**, so
diffusion runs on the SLURM `gpu` partition (see `run_gpu.sh`).

## Usage

Fast pipeline check, no GPU (procedural canvases, exercises the full
audio → zoom → mux path):

```bash
export LD_LIBRARY_PATH=/home/d.ramos/miniconda/envs/nibbler/lib:$LD_LIBRARY_PATH
/home/d.ramos/.local/share/mamba/envs/comfy/bin/python generate.py \
  --audio audio_sample.wav --output output/test.mp4 \
  --no-diffusion --beat-mode chain --tempo 200 --prompt "abstract thing"
```

CPU-only test suite (no GPU, runs on the login node):

```bash
/home/d.ramos/miniconda/envs/nibbler/bin/python -m pytest tests/ -q
```

The test command used during development (same, with explicit 200 BPM grid):

```bash
python generate.py --audio audio_sample.wav --output output/chain_mock.mp4 \
  --no-diffusion --beat-mode chain --tempo 200 --prompt "abstract thing"
```

Full diffusion render on the GPU node (example: 10 s sample, 34 beats):

```bash
sbatch run_gpu.sh
```

Full-song HD render (example: `NewBeginning.wav`, 188 s ≈ 628 beats, 1280×720):

```bash
sbatch run_newbeginning.sh
```

(or run `bash run_newbeginning.sh` directly on the GPU machine with
`CUDA_VISIBLE_DEVICES` set — the login node has no GPU).

Key flags: `--tempo` (else estimated — librosa often halves 200→99, pass it
manually), `--width/--height` (HD = 1280×720), `--model-id`, `--steps` (8 turbo
/ 30 full), `--guidance-scale` (full model only), `--gen-w/--gen-h` (640×368
turbo sharp domain, 960×544 full-model HD), `--canvas-scale` (1.2 output
headroom), `--beat-strength` (0.4 chain continuity), `--beats-per-prompt` (4),
`--kick-gain` (0.30), `--snare-gain` (0.05), `--drift` (0.03), `--zmax` (0.30),
`--kick-step` (0.05, 2-frame attack), `--beat-blend` (2-frame seam dissolve),
`--intro-fade` (0.35 s bloom opening), `--init-image` (proven beat-0 canvas),
`--seed-tries` (auto-pick contentful txt2img), `--max-duration` (trim for tests).

Outputs land in `output/` (gitignored): video, `.json` (model rev, seed, beats,
kick strengths, zoom params), `keys/` canvases, slurm logs.

## Prompt packs

Every prompt shares a STYLE prefix; each one **must describe sky texture**
(clouds/birds/branches) — flat unique-color skies read as dead air on quiet
beats. Prompts cycle (`--beats-per-prompt` beats each).

### Pack A — Greek temples (used for `newbeginning_hd.mp4`)

STYLE = `painterly matte painting, ancient greek marble temples, dramatic
textured skies, strong contrast, deep shadows, no text, no watermark`
(negative: `blurry, low detail, watermark, text, night, flat sky, empty sky,
gradient background, oversaturated, neon, monochrome, monochromatic, solid
blue, blue tint, cyan tint, flat blue`)

1. white marble temple on clifftop, swirling storm clouds with sun rays breaking through, eagles circling, pine branches framing the view
2. pushing through pine branches toward sunlit temple stairs, tiny robed figure climbing, clouds churning orange and teal
3. temple courtyard with tall columns, dusk sky with first stars and dramatic cloud bands, drifting mist
4. looking up steep cliff stairs to a temple against burning sunset clouds, flocks of birds
5. sea of clouds with a floating rock shrine island, shafts of sunlight, distant birds
6. storm clouds parting over a sunlit temple, glowing cloud edges, dark pine branches in the foreground
7. ancient stone arch corridor framing a moonlit acropolis, textured night clouds with moon glow
8. robed philosopher on a rock ledge gazing at a radiant temple above churning clouds, sunbeams
9. waterfall of clouds spilling over a cliff edge beneath a temple, mist and light shafts, pines
10. temple reflected in a still mountain lake at dawn, pink cloud streaks, dark cypress trees
11. close flight between giant marble columns toward a bright courtyard, doves, sun flare
12. panoramic vista with distant temples on islands, dramatic sunset sea of clouds, birds

### Pack B — abstract shapes (untested — start with `--test` / short `--max-duration`)

STYLE = `abstract digital painting, flowing three-dimensional shapes,
intricate surface detail everywhere, no flat backgrounds, no text, no watermark`
(negative: `blurry, low detail, watermark, text, flat color, empty background,
gradient background`)

1. molten gold and teal silk ribbons folding in dark space, glowing filaments, drifting sparks
2. crystalline cavern of refractive prisms, light shafts, floating dust motes
3. close flight through giant translucent membranes with bioluminescent veins, drifting motes
4. slow cosmic lava blobs of magenta and cyan, grainy nebula texture, tiny stars
5. endless corridor of glowing arches over a reflective floor, luminous haze
6. macro shot of iridescent oil film, swirling rainbow slick, tiny bubbles
7. layered frozen glass waves, refractions and caustics, suspended droplets
8. vortex tunnel of glowing embers and ice crystals spiraling inward

## Validating motion

`/tmp/opencode/measure_zoom.py` reports per-frame zoom % (optical-flow
divergence). Reference: median ≈ +1.9 %/frame, spikes to +9 %, never negative.
Procedural mocks should read median ≳ 1.2 with 0 % zoom-back within beats;
on rich diffusion content the meter under-reads ~2× (dark/pastel areas go
quiet), so judge cruising speed on mocks and aesthetics on contact sheets.

## Models

- Default: `Tongyi-MAI/Z-Image-Turbo` (8 steps, no CFG) — fast checks.
- Full `Tongyi-MAI/Z-Image` (30 steps + `guidance_scale` 4): use
  `--model-id Tongyi-MAI/Z-Image --steps 30 --guidance-scale 4.0`.
- **Kick test verdict (H200 NVL, 640×368, 30 steps): txt2img 3.1 s,
  img2img ~1.4 s → a 188 s / 200 BPM song (≈628 beats) renders in ~15 min —
   far under 2 h. Quality is a clear step up (painterly detail, natural skies,
   prompt adherence). Tested via `scripts/ztest.py` / `test_zimage.sh`
  (`output/ztest_*.png`, verdict `UNDER_2H` in the job log). Recommended for
  final renders; generate at 960×544 for HD output.
- **Head-to-head at HD gen res (960×544, same prompt/seed, foreground `srun`,
  `output/cmp_*.png`): Turbo (8 steps) txt2img 1.9 s / img2img 0.8 s → song
  ≈ 8 min, BUT on this seed it drew an abstract yellow-red gradient (weak
  prompt adherence); full Z-Image (30 steps + guidance 4) txt2img 4.8 s /
  img2img 2.0 s → song ≈ 21 min (UNDER_2H) with detailed temples, storm
  clouds, eagles. Outputs differ hugely (mean abs diff 94.8 → genuinely
  different weights, not a relabel). Full model wins for finals; Turbo stays
  for fast checks. Proof that `--model-id` selects the weights: runs log
  `model=` + `rev=` (`[chain] model=...` line) and the `.json` records both —
  Turbo rev `013496ad`, full rev `04cc4abb`.
