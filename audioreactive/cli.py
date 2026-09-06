"""CLI: argument parsing plus chain/cycle orchestration and run metadata."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import cv2

from . import config
from .audio import analyze_audio
from .color import grade_keys, upscale_keys
from .diffusion import diffusion_chain_canvases, diffusion_keyframes_chained
from .render import (
    plan_segments,
    procedural_chain_canvases,
    procedural_keyframes,
    render_chain,
    render_frames,
    write_video,
)


def build_parser():
    ap = argparse.ArgumentParser(description="Audio-reactive video generator (Z-Image).")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt", action="append", default=[], help="repeatable; default: abstract thing")
    ap.add_argument("--num-keyframes", type=int, default=0, help="cycle prompts up to N keyframes")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--steps", type=int, default=8,
                    help="diffusion steps (8 = turbo default, 28-50 = full Z-Image)")
    ap.add_argument("--model-id", type=str, default=config.MODEL_ID,
                    help="image model (default Z-Image-Turbo; full: Tongyi-MAI/Z-Image)")
    ap.add_argument("--guidance-scale", type=float, default=None,
                    help="CFG strength, full Z-Image only (recommended 3-5); omit for turbo")
    ap.add_argument("--max-duration", type=float, default=None, help="trim audio (s) for quick tests")
    ap.add_argument("--no-diffusion", action="store_true", help="procedural keyframes, no GPU")
    ap.add_argument("--chain-strength", type=float, default=0.55, help="img2img strength between chained keys (lower=more stable)")
    ap.add_argument("--negative-prompt", type=str, default="blurry, low detail, watermark, text")
    ap.add_argument("--no-grade", action="store_true", help="skip sage-grade of yellow skies")
    ap.add_argument("--key-every", type=str, default="auto",
                    help="'auto' = beat grid (beats-per-scene), or fixed seconds")
    ap.add_argument("--beats-per-scene", type=int, default=2,
                    help="scene length in beats when key-every=auto")
    ap.add_argument("--tempo", type=float, default=None,
                    help="BPM override; if omitted, estimated from audio")
    ap.add_argument("--zoom-punch", type=float, default=1.6,
                    help="kick attack: zoom velocity (1/s, log units) added per hit x strength")
    ap.add_argument("--zoom-tau", type=float, default=0.10,
                    help="zoom velocity relax time constant (s) toward cruise")
    ap.add_argument("--zoom-cruise", type=float, default=0.62,
                    help="sustained forward push-in speed (1/s, log units)")
    ap.add_argument("--zoom-step", type=float, default=0.05,
                    help="instant log-zoom jump per hit x strength (single-frame punch)")
    ap.add_argument("--zoom-max", type=float, default=1.85,
                    help="per-key max zoom (outgoing layer keeps flying through blends)")
    ap.add_argument("--blend-frac", type=float, default=0.5,
                    help="crossfade length as fraction of scene length (cap 0.6s)")
    ap.add_argument("--beat-mode", type=str, default="chain", choices=["chain", "cycle"],
                    help="chain: one canvas per beat rebased from prev end view "
                    "(zoom never steps back); cycle: legacy key crossfade")
    ap.add_argument("--kick-gain", type=float, default=config.KICK_GAIN,
                    help="per-beat zoom from low-band (<150Hz) kick strength")
    ap.add_argument("--snare-gain", type=float, default=config.SNARE_GAIN,
                    help="per-beat zoom from mid-band (150-500Hz) snare strength")
    ap.add_argument("--drift", type=float, default=config.DRIFT,
                    help="baseline per-beat push-in (beats with no kick still creep)")
    ap.add_argument("--zmax", type=float, default=config.ZMAX,
                    help="cap on single-beat zoom (fraction, e.g. 0.30 = +30%%)")
    ap.add_argument("--kick-step", type=float, default=config.KICK_STEP,
                    help="instant zoom step on kick attack x beat kick (over ~2 frames)")
    ap.add_argument("--beat-blend", type=int, default=config.BEAT_BLEND,
                    help="dissolve frames over prev beat end view (hides img2img seams)")
    ap.add_argument("--intro-fade", type=float, default=config.INTRO_FADE,
                    help="bloom opening: fade from white over first seconds (0 disables)")
    ap.add_argument("--seed-tries", type=str, default="7",
                    help="comma-separated beat-0 seeds; most contentful wins")
    ap.add_argument("--init-image", type=str, default=None,
                    help="proven canvas PNG: beat-0 via img2img from it (skips txt2img lottery)")
    ap.add_argument("--beat-strength", type=float, default=config.BEAT_STRENGTH,
                    help="img2img strength between consecutive beat canvases")
    ap.add_argument("--match-ab", type=float, default=0.0,
                    help="chroma transfer in per-beat LAB match (1=full, "
                    "0=luminance only; full chroma match was proven to "
                    "ratchet color out over 600+ beats)")
    ap.add_argument("--beats-per-prompt", type=int, default=config.BEATS_PER_PROMPT,
                    help="beats each prompt lasts in chain mode (prompts cycle)")
    ap.add_argument("--gen-w", type=int, default=640)
    ap.add_argument("--gen-h", type=int, default=368)
    ap.add_argument("--canvas-scale", type=float, default=1.2,
                    help="display canvas = output x scale (zoom/pan headroom; "
                    "generation stays at native gen res to avoid blur buildup)")
    ap.add_argument("--test", action="store_true", help="shortcut: 4s, 640x368")
    return ap


def save_keys(keys, keydir, base):
    for i, k in enumerate(keys):
        cv2.imwrite(os.path.join(keydir, f"{base}_key{i:02d}.png"),
                    cv2.cvtColor(k, cv2.COLOR_RGB2BGR))


def base_meta(args, prompts, an):
    return {
        "model": args.model_id if not args.no_diffusion else "procedural",
        "seed": args.seed,
        "prompts": prompts,
        "negative_prompt": args.negative_prompt,
        "fps": args.fps,
        "size": [args.width, args.height],
        "hits": an["hits"],
        "beats": an["beats"],
        "tempo": an["tempo"],
        "tempo_source": an["tempo_source"],
        "beat_mode": args.beat_mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_meta(meta, out_path):
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def run_chain(args, prompts, an, meta, keydir, base):
    # Per-beat zoom from band strengths: kicks punch, snares whisper.
    # Kick term uses a gamma (1.5) so weak low-band leakage barely moves
    # the camera while true kicks punch through.
    z_beats = [min(args.zmax, args.drift + args.kick_gain * (b[1] ** 1.5)
                   + args.snare_gain * b[2])
               for b in an["beats"]]
    meta.update({"drift": args.drift, "kick_gain": args.kick_gain,
                 "snare_gain": args.snare_gain, "zmax": args.zmax,
                 "kick_step": args.kick_step, "beat_blend": args.beat_blend,
                 "intro_fade": args.intro_fade,
                 "beat_strength": args.beat_strength,
                 "beats_per_prompt": args.beats_per_prompt,
                 "match_ab": args.match_ab,
                 "anchor_pull_l": config.ANCHOR_PULL_L,
                 "anchor_pull_s_lo": config.ANCHOR_PULL_S_LO,
                 "anchor_pull_s_hi": config.ANCHOR_PULL_S_HI,
                 "anchor_pull_ab": config.ANCHOR_PULL_AB,
                 "tame_blue_thresh": config.TAME_BLUE_FRAC,
                 "pan_max_step": config.PAN_MAX_STEP,
                 "z_beats": [round(z, 4) for z in z_beats]})
    rev = "procedural"
    if args.no_diffusion:
        canvases = procedural_chain_canvases(
            prompts, len(an["beats"]), args.width, args.height, args.seed,
            beats_per_prompt=args.beats_per_prompt)
        used_pans = None  # render falls back to random-walk pans
    else:
        canvases, rev, used_seed, used_pans = diffusion_chain_canvases(
            prompts, an["beats"], z_beats, args.seed, args.steps,
            args.gen_w, args.gen_h, beat_strength=args.beat_strength,
            beats_per_prompt=args.beats_per_prompt,
            seed_tries=args.seed_tries, init_image=args.init_image,
            model_id=args.model_id, guidance=args.guidance_scale,
            match_ab=args.match_ab,
            negative_prompt=args.negative_prompt)
        meta["seed_used"] = used_seed
        meta["init_image"] = args.init_image
        meta["model"] = args.model_id
        meta["pans"] = [[round(float(x), 4), round(float(y), 4)] for x, y in used_pans]
        if not args.no_grade:
            canvases = grade_keys(canvases)
        cw = max(16, round(args.width * args.canvas_scale / 2) * 2)
        ch = max(16, round(args.height * args.canvas_scale / 2) * 2)
        canvases = upscale_keys(canvases, cw, ch)
        save_keys(canvases, keydir, base)
    meta["model_rev"] = rev
    print("[render] rendering beat chain ...", flush=True)
    frames = render_chain(canvases, an, z_beats, args.width, args.height,
                          args.fps, args.seed,
                          kick_step=args.kick_step,
                          beat_blend=args.beat_blend,
                          intro_fade=args.intro_fade, pans=used_pans)
    print(f"[mux] writing {args.output} ...", flush=True)
    write_video(frames, args.fps, args.audio, args.output, an["dur"])
    write_meta(meta, args.output)
    print(f"[done] {args.output} beats={len(an['beats'])} rev={rev}", flush=True)
    return 0


def run_cycle(args, prompts, an, meta, keydir, base):
    t0 = an["hits"][0][0] if an["hits"] else 0.0  # anchor beat grid on first kick
    rev = "procedural"
    if args.no_diffusion:
        keys = procedural_keyframes(prompts, args.width, args.height, args.seed)
    else:
        keys, rev = diffusion_keyframes_chained(
            prompts, args.seed, args.steps, args.gen_w, args.gen_h,
            chain_strength=args.chain_strength, model_id=args.model_id,
            guidance=args.guidance_scale,
            negative_prompt=args.negative_prompt)
        if not args.no_grade:
            keys = grade_keys(keys)
        keys = upscale_keys(keys, args.width, args.height)
        # Persist keyframes for inspection / reuse.
        save_keys(keys, keydir, base)

    seg = plan_segments(an["n_frames"], an["is_hit"], an["hit_env"], len(keys),
                        args.fps, key_every=args.key_every, tempo=an["tempo"],
                        beats_per_scene=args.beats_per_scene, t0=t0)
    print("[render] rendering frames ...", flush=True)
    frames = render_frames(keys, an, seg, args.width, args.height, args.fps,
                           args.seed, tempo=an["tempo"],
                           beats_per_scene=args.beats_per_scene,
                           zoom_punch=args.zoom_punch, zoom_tau=args.zoom_tau,
                           zoom_cruise=args.zoom_cruise, zoom_max=args.zoom_max,
                           zoom_step=args.zoom_step, blend_frac=args.blend_frac)
    print(f"[mux] writing {args.output} ...", flush=True)
    write_video(frames, args.fps, args.audio, args.output, an["dur"])

    meta.update({
        "model_rev": rev,
        "beats_per_scene": args.beats_per_scene,
        "zoom_punch": args.zoom_punch,
        "zoom_tau": args.zoom_tau,
        "zoom_cruise": args.zoom_cruise,
        "zoom_step": args.zoom_step,
        "zoom_max": args.zoom_max,
        "blend_frac": args.blend_frac,
        "key_every": args.key_every,
    })
    write_meta(meta, args.output)
    print(f"[done] {args.output} hits={len(an['hits'])} rev={rev}", flush=True)
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.seed_tries = [int(x) for x in str(args.seed_tries).split(",") if x.strip()]

    if args.test:
        args.max_duration = args.max_duration or 4.0
        args.width, args.height = 640, 368  # multiples of 16 (Z-Image requirement)
    # Z-Image requires H/W divisible by 16; VideoWriter needs even dims.
    args.width = max(16, round(args.width / config.DIM_MULTIPLE) * config.DIM_MULTIPLE)
    args.height = max(16, round(args.height / config.DIM_MULTIPLE) * config.DIM_MULTIPLE)
    prompts = args.prompt or ["abstract thing"]
    if args.num_keyframes and args.num_keyframes > len(prompts):
        reps = (args.num_keyframes + len(prompts) - 1) // len(prompts)
        prompts = (prompts * reps)[: args.num_keyframes]

    print(f"[audio] analyzing {args.audio} ...", flush=True)
    an = analyze_audio(args.audio, args.fps, args.max_duration, tempo=args.tempo)
    print(f"[audio] dur={an['dur']:.2f}s frames={an['n_frames']} "
          f"tempo={an['tempo']:.1f} ({an['tempo_source']}) hits={len(an['hits'])} "
          f"beats={len(an['beats'])}",
          flush=True)

    keydir = os.path.join(os.path.dirname(args.output) or ".", "keys")
    os.makedirs(keydir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.output))[0]
    meta = base_meta(args, prompts, an)

    if args.beat_mode == "chain":
        return run_chain(args, prompts, an, meta, keydir, base)
    return run_cycle(args, prompts, an, meta, keydir, base)


if __name__ == "__main__":
    sys.exit(main())
