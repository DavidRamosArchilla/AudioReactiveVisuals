"""Audio-reactive video generator.

Pipeline: audio (librosa onset/kick) -> keyframes (Z-Image-Turbo) -> per-frame
render (opencv zoom/pulse/crossfade) -> mux audio (ffmpeg).

Usage:
    python generate.py --audio audio_sample.wav --output output/demo.mp4
    python generate.py --audio audio_sample.wav --output output/test.mp4 \
        --prompt "abstract thing" --max-duration 4 --width 640 --height 360
    python generate.py --audio audio_sample.wav --output output/fast.mp4 \
        --no-diffusion --max-duration 4   # pipeline check without GPU gen
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone

import cv2
import librosa
import numpy as np

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo")


def resolve_model_rev():
    """Return pinned snapshot hash(es) from HF cache without downloading."""
    snaps = []
    d = os.path.join(CACHE_DIR, "snapshots")
    if os.path.isdir(d):
        snaps = sorted(os.listdir(d))
    return snaps


def analyze_audio(path, fps, max_duration=None):
    y, sr = librosa.load(path, sr=22050, mono=True, duration=max_duration)
    dur = len(y) / sr
    n_frames = max(1, int(dur * fps))
    # Onset envelope at frame resolution.
    hop = 512
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop)
    frame_times = np.arange(n_frames) / fps
    energy = np.interp(frame_times, times, onset).astype(np.float32)
    # Normalize 0..1 (robust to outliers).
    hi = float(np.quantile(energy, 0.98)) or 1.0
    energy = np.clip(energy / hi, 0, 1)
    # Smooth with small moving average.
    k = max(1, int(fps * 0.08))
    energy = np.convolve(energy, np.ones(k) / k, mode="same").astype(np.float32)
    # Hit picking: peaks of onset envelope, fallback to energy threshold.
    peaks = librosa.util.peak_pick(
        energy, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.25, wait=5
    )
    if len(peaks) == 0:
        peaks = np.where(energy > 0.65)[0]
        peaks = peaks[np.concatenate([[True], np.diff(peaks) > int(fps * 0.25)])]
    strengths = energy[peaks] if len(peaks) else np.array([], dtype=np.float32)
    # Hit decay envelope for pulse effects.
    hit_env = np.zeros(n_frames, dtype=np.float32)
    is_hit = np.zeros(n_frames, dtype=bool)
    is_hit[peaks] = True
    decay = np.exp(-np.arange(n_frames) / (fps * 0.25))
    for p, s in zip(peaks.tolist(), strengths.tolist()):
        w = min(n_frames - p, len(decay))
        hit_env[p : p + w] = np.maximum(hit_env[p : p + w], decay[:w] * float(s))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return {
        "y": y,
        "sr": sr,
        "dur": dur,
        "n_frames": n_frames,
        "energy": energy,
        "hit_env": hit_env,
        "is_hit": is_hit,
        "hits": sorted((float(p / fps), float(s)) for p, s in zip(peaks.tolist(), strengths.tolist())),
        "tempo": float(np.atleast_1d(tempo)[0]),
    }


def procedural_keyframes(prompts, w, h, seed):
    """Deterministic gradient/fractal-ish fallback (no GPU)."""
    rng = np.random.default_rng(seed)
    keys = []
    for i, p in enumerate(prompts):
        digest = hashlib.md5(f"{p}-{seed}".encode()).digest()
        hue = digest[0] / 255.0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.zeros((h, w, 3), np.float32)
        img[..., 0] = (0.5 + 0.5 * np.sin(xx / (30 + digest[1]) + hue * 6.28)) * (0.4 + 0.6 * yy / h)
        img[..., 1] = (0.5 + 0.5 * np.cos(yy / (25 + digest[2]) + i)) * (0.4 + 0.6 * xx / w)
        img[..., 2] = 0.3 + 0.7 * ((xx + yy) / (w + h))
        noise = rng.random((h // 4, w // 4, 3), dtype=np.float32)
        noise = cv2.resize(noise, (w, h))
        img = np.clip(0.7 * img + 0.3 * noise, 0, 1)
        keys.append((img * 255).astype(np.uint8))
    return keys


def diffusion_keyframes_chained(prompts, seed, steps=8, gen_w=640, gen_h=368,
                                chain_strength=0.55,
                                negative_prompt="blurry, low detail, watermark, text"):
    """Temporally coherent keyframes: key0 = txt2img, key_i = img2img(key_{i-1}).

    Chaining (instead of independent txt2img per prompt) keeps one evolving
    world like the reference: same palette/composition, morphing details.
    """
    import gc
    import torch
    from diffusers import ZImageImg2ImgPipeline, ZImagePipeline
    from PIL import Image

    revs = resolve_model_rev()
    # NOTE: keep weights on CUDA (15GB CPU RAM: cpu_offload OOM-kills).
    t2i = ZImagePipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
    )
    t2i.to("cuda")
    g0 = torch.Generator("cuda").manual_seed(seed)
    img = t2i(prompt=prompts[0], height=gen_h, width=gen_w,
              num_inference_steps=steps, generator=g0,
              negative_prompt=negative_prompt).images[0]
    print(f"[keyframes] 1/{len(prompts)} txt2img {prompts[0]!r}", flush=True)
    keys = [np.array(img)]
    # Free txt2img weights BEFORE loading img2img: both exceed 8GB together.
    del t2i
    gc.collect()
    torch.cuda.empty_cache()
    if len(prompts) > 1:
        i2i = ZImageImg2ImgPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
        )
        i2i.to("cuda")
        try:
            i2i.enable_vae_slicing()
        except Exception:
            pass
        for i, p in enumerate(prompts[1:], 2):
            gi = torch.Generator("cuda").manual_seed(seed + i)
            prev = Image.fromarray(keys[-1]).resize((gen_w, gen_h))
            img = i2i(prompt=p, image=prev, strength=chain_strength,
                      num_inference_steps=steps, generator=gi,
                      negative_prompt=negative_prompt).images[0]
            keys.append(np.array(img))
            print(f"[keyframes] {i}/{len(prompts)} img2img {p!r}", flush=True)
        del i2i
    gc.collect()
    torch.cuda.empty_cache()
    return keys, (revs[0] if revs else "unknown")


def upscale_keys(keys, w, h):
    """Lanczos upscale + light unsharp mask for crisper render frames."""
    out = []
    for k in keys:
        up = cv2.resize(k, (w, h), interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(up, (0, 0), 2.0)
        sharp = cv2.addWeighted(up, 1.35, blur, -0.35, 0)
        out.append(sharp)
    return out


def plan_segments(n_frames, is_hit, hit_env, n_keys, fps, key_every=2.5):
    """Slow, coherent scene evolution: advance on a fixed cadence; hits only
    drive camera pulse/brightness (not scene switches), except extreme hits."""
    seg = np.zeros(n_frames, dtype=int)
    cur, last_switch = 0, 0
    auto_every = int(fps * key_every)
    for f in range(n_frames):
        extreme = bool(is_hit[f]) and float(hit_env[f]) > 0.90
        if extreme and f - last_switch > int(fps * 1.0):
            cur = (cur + 1) % n_keys
            last_switch = f
        elif f - last_switch >= auto_every:
            cur = (cur + 1) % n_keys
            last_switch = f
        seg[f] = cur
    return seg


def render_frames(keys, analysis, seg, w, h, fps, seed):
    rng = random.Random(seed)
    shakes = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(len(keys) * 8 + 8)]
    energy, hit_env = analysis["energy"], analysis["hit_env"]
    n = analysis["n_frames"]
    # Pre-scale keys 1.35x for zoom/pan headroom.
    big = [(int(w * 1.35), int(h * 1.35))]
    KB = [cv2.resize(k, big[0], interpolation=cv2.INTER_LINEAR) for k in keys]
    bw, bh = big[0]
    frames = []
    blend = max(2, int(fps * 0.6))  # long dreamy morph between coherent keys
    for f in range(n):
        cur = int(seg[f])
        # Find segment start for blend factor.
        s = f
        while s > 0 and seg[s - 1] == cur:
            s -= 1
        t = min(1.0, (f - s) / blend)
        # On very strong hits use a near-cut (fast blend).
        if bool(analysis["is_hit"][f]) and float(hit_env[f]) > 0.8:
            t = 1.0
        prev = int(seg[s - 1]) if s > 0 else cur
        e, he = float(energy[f]), float(hit_env[f])
        zoom = 1.0 + 0.13 * e + 0.18 * he
        # Crop window with slow pan + hit shake.
        cw, ch = int(w / zoom), int(h / zoom)
        px = (bw - cw) * (0.5 + 0.4 * np.sin(f / (fps * 3.0)))
        py = (bh - ch) * (0.5 + 0.4 * np.cos(f / (fps * 4.0)))
        sx, sy = shakes[(f // 3) % len(shakes)]
        px = np.clip(px + sx * he * 14, 0, bw - cw)
        py = np.clip(py + sy * he * 14, 0, bh - ch)
        a = KB[prev][int(py) : int(py) + ch, int(px) : int(px) + cw]
        b = KB[cur][int(py) : int(py) + ch, int(px) : int(px) + cw]
        frame = cv2.addWeighted(a, 1 - t, b, t, 0)
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        # Brightness pulse on hits.
        frame = np.clip(frame.astype(np.float32) * (1.0 + 0.22 * he), 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def write_video(frames, fps, audio_path, out_path, dur):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".silent.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    for fr in frames:
        vw.write(fr)
    vw.release()
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", tmp, "-i", audio_path,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-shortest", out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser(description="Audio-reactive video generator (Z-Image-Turbo).")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt", action="append", default=[], help="repeatable; default: abstract thing")
    ap.add_argument("--num-keyframes", type=int, default=0, help="cycle prompts up to N keyframes")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--max-duration", type=float, default=None, help="trim audio (s) for quick tests")
    ap.add_argument("--no-diffusion", action="store_true", help="procedural keyframes, no GPU")
    ap.add_argument("--chain-strength", type=float, default=0.55, help="img2img strength between chained keys (lower=more stable)")
    ap.add_argument("--negative-prompt", type=str, default="blurry, low detail, watermark, text")
    ap.add_argument("--key-every", type=float, default=2.5, help="seconds per keyframe scene")
    ap.add_argument("--test", action="store_true", help="shortcut: 4s, 640x360")
    args = ap.parse_args()

    if args.test:
        args.max_duration = args.max_duration or 4.0
        args.width, args.height = 640, 368  # multiples of 16 (Z-Image requirement)
    # Z-Image requires H/W divisible by 16; VideoWriter needs even dims.
    args.width = max(16, round(args.width / 16) * 16)
    args.height = max(16, round(args.height / 16) * 16)
    prompts = args.prompt or ["abstract thing"]
    if args.num_keyframes and args.num_keyframes > len(prompts):
        reps = (args.num_keyframes + len(prompts) - 1) // len(prompts)
        prompts = (prompts * reps)[: args.num_keyframes]

    print(f"[audio] analyzing {args.audio} ...", flush=True)
    an = analyze_audio(args.audio, args.fps, args.max_duration)
    print(f"[audio] dur={an['dur']:.2f}s frames={an['n_frames']} tempo={an['tempo']:.1f} hits={len(an['hits'])}", flush=True)

    rev = "procedural"
    if args.no_diffusion:
        keys = procedural_keyframes(prompts, args.width, args.height, args.seed)
    else:
        keys, rev = diffusion_keyframes_chained(
            prompts, args.seed, args.steps, chain_strength=args.chain_strength,
            negative_prompt=args.negative_prompt)
        keys = upscale_keys(keys, args.width, args.height)

    seg = plan_segments(an["n_frames"], an["is_hit"], an["hit_env"], len(keys),
                        args.fps, key_every=args.key_every)
    print("[render] rendering frames ...", flush=True)
    frames = render_frames(keys, an, seg, args.width, args.height, args.fps, args.seed)
    print(f"[mux] writing {args.output} ...", flush=True)
    write_video(frames, args.fps, args.audio, args.output, an["dur"])

    meta = {
        "model": MODEL_ID if not args.no_diffusion else "procedural",
        "model_rev": rev,
        "seed": args.seed,
        "prompts": prompts,
        "fps": args.fps,
        "size": [args.width, args.height],
        "hits": an["hits"],
        "tempo": an["tempo"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.splitext(args.output)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {args.output} hits={len(an['hits'])} rev={rev}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
