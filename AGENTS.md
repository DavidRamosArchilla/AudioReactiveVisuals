# AGENTS.md — video_generation

## Intent
- Audio-reactive video gen in Python. Input: audio → output: video.
- `reference_video.mp4` (1918x1078, h264, 30fps, ~8.8s) is style/behavior ref only — never overwrite; mimic kick-reactive changes but smoother (interpolate, hard cuts only on strong hits).

## Environment — base conda only
- Use base (`/home/david/miniconda3`, py3.11): torch+CUDA, diffusers, librosa, moviepy, opencv, ffmpeg 4.4.2 already present.
- Install missing deps into base (`pip install <pkg>`), pin in `requirements.txt`. Ignore `audio2vid` env.

## Image model — Z-Image-Turbo (locked)
- `Tongyi-MAI/Z-Image-Turbo` (6B, 8 steps, Apache-2.0). Already cached at `~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/` — reuse, don't re-download.
- Why: best quality that still runs on RTX 4060 8GB (Elo 1133); klein-9B/dev/Ideogram need 18-29GB + non-commercial, klein-4B/schnell score lower.
- Use `ZImagePipeline`, `torch.bfloat16`, on `cuda`. Pin revision + seed.
- Machine has 15GB CPU RAM / 8GB VRAM: keep weights on CUDA; `enable_model_cpu_offload()` OOM-kills. Generate keyframes small (640x368), upscale for render.

## Conditioning / pipeline
- Conditioning flexible: 0/1/N prompts OK; `abstract thing` valid for tests; unconditioned/procedural frames allowed.
- Shape: `audio onset/kick (librosa)` → `frames (Z-Image)` → `assemble (ffmpeg/moviepy)`; outputs in `output/` (gitignored). Log model rev, seed, hit timestamps.
- Key CLI: `python generate.py --audio X.wav --output output/Y.mp4 [--tempo BPM] [--beats-per-scene 2] [--key-every auto] [--test] [--no-diffusion]`. Tempo optional (else estimated); scenes change on beat grid; kicks drive spring zoom punch (`--zoom-punch/--zoom-tau`). Keys are chained img2img for coherence; yellow skies sage-graded unless `--no-grade`.
