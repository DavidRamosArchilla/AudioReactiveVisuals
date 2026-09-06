# AGENTS.md — video_generation

## Intent
- Audio-reactive video gen in Python. Input: audio → output: video.
- `reference_video.mp4` (1918x1078, h264, 30fps, ~8.8s) is style/behavior ref only — never overwrite; mimic kick-reactive changes but smoother (interpolate, hard cuts only on strong hits).

## Environment — this cluster
- Login node has NO GPU (only ASPEED BMC). Diffusion runs on the SLURM `gpu` partition (`sbatch run_gpu.sh`: 1 GPU, nibbler env).
- Env: `nibbler` (`/home/d.ramos/miniconda/envs/nibbler`, torch 2.12+cu130, diffusers 0.40). Deps pinned in `requirements.txt`.
- Cluster gotchas: prepend conda lib to `LD_LIBRARY_PATH` (else ohpc libstdc++ breaks scipy/diffusers import); no system ffmpeg → `generate.py` falls back to the imageio-ffmpeg binary.

## Image model — Z-Image-Turbo (locked)
- `Tongyi-MAI/Z-Image-Turbo` (6B, 8 steps, Apache-2.0). Already cached at `~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/` — reuse, don't re-download.
- Full `Tongyi-MAI/Z-Image` (30 steps + guidance 4) for final renders: kick-tested on H200 NVL (`scripts/ztest.py`/`test_zimage.sh`, `output/ztest_direct.log`) at txt2img 3.1s / img2img ~1.4s → 188s song ≈14 min (UNDER_2H). Generate at 960x544 for HD output.
- Use `ZImagePipeline` (beat 0) + `ZImageImg2ImgPipeline` (chain), `torch.bfloat16`, on `cuda`. Pin revision + seed. Free txt2img weights before img2img (both exceed small VRAM together).
- Generate at native 640x368 (model's sharp domain), upscale canvases x1.2 for render headroom. Never chain from upscaled crops (blur compounds); downscale to gen res for img2img init.

## Code layout
- `generate.py` is a thin shim; all logic lives in `audioreactive/`: `config.py` (every tuning number — change values here, never as literals), `audio.py`, `camera.py`, `color.py`, `diffusion.py` (GPU-only), `render.py`, `cli.py` (argparse + orchestration). `python -m audioreactive` == `python generate.py`.
- Dev tools in `scripts/` (`ztest.py`, `check_drift.py`); CPU-only pytest suite in `tests/` (must stay green: run before pushing behavior changes).
- Procedural `--no-diffusion` outputs are deterministic: verify refactors by md5 against baselines before/after (chain + cycle modes).

## Conditioning / pipeline
- Conditioning flexible: 0/1/N prompts OK; `abstract thing` valid for tests; unconditioned/procedural frames allowed.
- Shape: `audio bands (librosa)` → `beat canvases (Z-Image)` → `assemble (ffmpeg)`; outputs in `output/` (gitignored). Log model rev, seed, beats + kick strengths.
- Key CLI: `python generate.py --audio X.wav --output output/Y.mp4 --tempo 200 [--beat-mode chain] [--test] [--no-diffusion]`. Tempo optional (else estimated, often octave-halved — pass manually).
- Beat chain (default): one canvas per beat (200 BPM = 0.3 s), each img2img'd from the previous beat's end view (rebase) at `--beat-strength`, so the zoom NEVER steps back. Intra-beat zoom is linear + 2-frame kick step, amount `min(--zmax, --drift + --kick-gain*kick^1.5 + --snare-gain*snare)` with kick from <150 Hz flux, snare 150-500 Hz. 2-frame dissolve (`--beat-blend`) hides seams; LAB color match kills flicker; `--intro-fade` blooms the opening; `--init-image` seeds beat 0 from a proven canvas; `--seed-tries` auto-picks the most contentful txt2img otherwise.
- Pan is texture-seeking (`best_window`) BUT temporally clamped (`smooth_pan`, ≤0.10/beat): unclamped global-max picks teleported across the frame (measured jumps ~1.25 → abrupt camera yanks). Never promise "textured sky" via prompts alone — prompts + clamp + `tame_blue` guard (desats canvases >45% electric blue/cyan) together prevent sky-lock runs where a sky zoom self-reinforces into a blue wash. NEG must always carry anti-blue terms: `monochrome, monochromatic, solid blue, blue tint, cyan tint, flat blue`. Never write "blue sky" in prompts — use "textured/dramatic sky".
- Palette drift: every beat chains from the previous output, so small biases integrate (measured L 129→246 + S 127→46 + hue casts over 629 beats). Two defenses: (1) `anchor_pull` — weak per-beat pull of L/S/a/b means toward the fixed beat-0 canvas (L 0.15, S 0.40-lo/0.15-hi asymmetric, ab 0.10); (2) `--match-ab 0.0` (default) — per-beat LAB match transfers luminance only. Full chroma match was PROVEN to desaturate (iter tests: 0.5 → flat S~55 gray; 0.0 → scene-driven S 50-120 with blue skies + orange sunsets). Check drift with `scripts/check_drift.py` (L/S trajectory + t=10/75/150 strip).
- Legacy `--beat-mode cycle` (key crossfade flight) kept for `--no-diffusion` pipeline tests only — its blend resets cause visible zoom-back.
- Grade (`grade_keys`, skip with `--no-grade`): desaturate yellow-green/blue/magenta extremes + highlight rolloff + S-curve.
- Reference motion (measured Farneback divergence): perpetual push-in median ~+1.9%/frame, kick spikes to ~+9%/frame, never zooms out, no hard cuts. Validate renders with `/tmp/opencode/measure_zoom.py` (note: meter under-reads ~2x on dark/pastel content; verify geometry on procedural mocks, aesthetics on contact sheets).
