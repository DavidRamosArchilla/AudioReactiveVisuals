"""Diffusion: Z-Image txt2img/img2img canvases (GPU-only, runs on SLURM gpu)."""
import os
import random as _random
import time as _time

import cv2
import numpy as np

from .camera import best_window, center_crop_end, smooth_pan
from .color import anchor_pull, blend_chroma, match_color, tame_blue
from .config import MODEL_ID


def resolve_model_rev(model_id=MODEL_ID):
    """Return pinned snapshot hash(es) from HF cache without downloading."""
    cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--" + model_id.replace("/", "--"))
    snaps = []
    d = os.path.join(cache, "snapshots")
    if os.path.isdir(d):
        snaps = sorted(os.listdir(d))
    return snaps


def _load_pipeline(kind, model_id):
    """Load a Z-Image pipeline onto CUDA. kind: 't2i' or 'i2i'.

    NOTE: keep weights on CUDA (15GB CPU RAM: cpu_offload OOM-kills). Callers
    must free txt2img weights BEFORE loading img2img: both exceed 8GB together.
    """
    import torch
    from diffusers import ZImageImg2ImgPipeline, ZImagePipeline

    cls = ZImagePipeline if kind == "t2i" else ZImageImg2ImgPipeline
    pipe = cls.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    pipe.to("cuda")
    if kind == "i2i":
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    return pipe


def diffusion_keyframes_chained(prompts, seed, steps=8, gen_w=640, gen_h=368,
                                chain_strength=0.55, model_id=MODEL_ID,
                                guidance=None,
                                negative_prompt="blurry, low detail, watermark, text"):
    """Temporally coherent keyframes: key0 = txt2img, key_i = img2img(key_{i-1}).

    Chaining (instead of independent txt2img per prompt) keeps one evolving
    world like the reference: same palette/composition, morphing details.
    Legacy cycle-mode path only; the beat chain uses diffusion_chain_canvases.
    """
    import gc
    import torch
    from PIL import Image

    revs = resolve_model_rev(model_id)
    _gkw = {} if guidance is None else {"guidance_scale": guidance}
    t2i = _load_pipeline("t2i", model_id)
    g0 = torch.Generator("cuda").manual_seed(seed)
    img = t2i(prompt=prompts[0], height=gen_h, width=gen_w,
              num_inference_steps=steps, generator=g0,
              negative_prompt=negative_prompt, **_gkw).images[0]
    print(f"[keyframes] 1/{len(prompts)} txt2img {prompts[0]!r}", flush=True)
    keys = [np.array(img)]
    # Free txt2img weights BEFORE loading img2img: both exceed 8GB together.
    del t2i
    gc.collect()
    torch.cuda.empty_cache()
    if len(prompts) > 1:
        i2i = _load_pipeline("i2i", model_id)
        for i, p in enumerate(prompts[1:], 2):
            gi = torch.Generator("cuda").manual_seed(seed + i)
            prev = Image.fromarray(keys[-1]).resize((gen_w, gen_h))
            img = i2i(prompt=p, image=prev, strength=chain_strength,
                      num_inference_steps=steps, generator=gi,
                      negative_prompt=negative_prompt, **_gkw).images[0]
            keys.append(np.array(img))
            print(f"[keyframes] {i}/{len(prompts)} img2img {p!r}", flush=True)
        del i2i
    gc.collect()
    torch.cuda.empty_cache()
    return keys, (revs[0] if revs else "unknown")


def diffusion_chain_canvases(prompts, beats, z_beats, seed, steps=8,
                             gen_w=1024, gen_h=576, beat_strength=0.5,
                             beats_per_prompt=4, seed_tries=(7,), init_image=None,
                             model_id=MODEL_ID, guidance=None, match_ab=0.0,
                             negative_prompt="blurry, low detail, watermark, text"):
    """One canvas per beat, each img2img'd from the previous beat's end view.

    Because beat b starts exactly where beat b-1 ended (rebase), the flight
    zoom can NEVER step back — unlike crossfading key cycles, where the fresh
    layer restarts wide and the image visibly breathes out. img2img also
    regenerates detail, so the perpetual zoom never goes soft.
    """
    import gc
    import torch
    from PIL import Image

    revs = resolve_model_rev(model_id)
    _rng = _random.Random(seed)
    pans = [(0.5, 0.5)] * (len(beats) + 1)  # filled progressively (texture-seeking)
    _gkw = {} if guidance is None else {"guidance_scale": guidance}
    t2i = None
    if init_image is None:
        t2i = _load_pipeline("t2i", model_id)
        # Beat-0 seed search: turbo txt2img sometimes draws abstract sky bands;
        # try candidates, keep the most contentful (texture - neon penalty).
        tried = {}
        for s in seed_tries:
            gs = torch.Generator("cuda").manual_seed(s)
            img = t2i(prompt=prompts[0], height=gen_h, width=gen_w,
                      num_inference_steps=steps, generator=gs,
                      negative_prompt=negative_prompt, **_gkw).images[0]
            c = np.array(img)
            gray = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
            tex = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
            neon = float((hsv[..., 1].astype(np.float32) > 150).mean())
            tried[s] = tex - 200.0 * neon
            print(f"[chain] beat0 seed {s}: texture={tex:.1f} neon={neon:.3f}", flush=True)
        seed = max(tried, key=tried.get)
        print(f"[chain] beat0 seed -> {seed}", flush=True)
    if init_image is not None:
        print(f"[chain] beat0 from init image {init_image}", flush=True)
        _init = Image.open(init_image).convert("RGB").resize((gen_w, gen_h))
        _i2i0 = _load_pipeline("i2i", model_id)
        g0 = torch.Generator("cuda").manual_seed(seed)
        img = _i2i0(prompt=prompts[0], image=_init, strength=beat_strength,
                    num_inference_steps=steps, generator=g0,
                    negative_prompt=negative_prompt, **_gkw).images[0]
        del _i2i0
        gc.collect()
        torch.cuda.empty_cache()
        t2i = None
    else:
        g0 = torch.Generator("cuda").manual_seed(seed)
        img = t2i(prompt=prompts[0], height=gen_h, width=gen_w,
                  num_inference_steps=steps, generator=g0,
                  negative_prompt=negative_prompt, **_gkw).images[0]
    print(f"[chain] beat 1/{len(beats)} ready ({prompts[0][:60]!r}...)", flush=True)
    canvas = np.array(img)
    if t2i is not None:
        del t2i
        gc.collect()
        torch.cuda.empty_cache()
    canvases = [canvas]
    _lab0 = cv2.cvtColor(canvas, cv2.COLOR_RGB2LAB).astype(np.float32)
    _hsv0 = cv2.cvtColor(canvas, cv2.COLOR_RGB2HSV).astype(np.float32)
    _ref_l, _ref_s = float(_lab0[..., 0].mean()), float(_hsv0[..., 1].mean())
    _ref_a, _ref_b = float(_lab0[..., 1].mean()), float(_lab0[..., 2].mean())
    print(f"[chain] anchor L={_ref_l:.1f} S={_ref_s:.1f} A={_ref_a:.1f} B={_ref_b:.1f} "
          f"(beat-0, drift pull target)", flush=True)
    _g0 = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    _h0, _w0 = _g0.shape[:2]
    _t0 = cv2.resize(_g0, (160, max(16, int(160 * _h0 / _w0))))
    _f0 = 1.0 / (1.0 + z_beats[0])
    pans[1] = smooth_pan(pans[0], best_window(_t0, _f0, _f0, _rng))
    i2i = _load_pipeline("i2i", model_id)
    print(f"[chain] model={model_id} rev={(revs[0] if revs else 'unknown')} "
          f"steps={steps} strength={beat_strength} size={gen_w}x{gen_h}", flush=True)
    _t_chain = _time.time()
    for b in range(1, len(beats)):
        p = prompts[(b // beats_per_prompt) % len(prompts)]
        gb = torch.Generator("cuda").manual_seed(seed + b)
        init = center_crop_end(canvases[-1], z_beats[b - 1], pans[b])
        init = Image.fromarray(init).resize((gen_w, gen_h))
        img = i2i(prompt=p, image=init, strength=beat_strength,
                  num_inference_steps=steps, generator=gb,
                  negative_prompt=negative_prompt, **_gkw).images[0]
        new = np.array(img)
        # Lock luminance to the view we came from (kills flicker); chroma
        # transfer is scaled by match_ab (0 = model keeps its own color).
        new = blend_chroma(new, match_color(new, np.array(init)), match_ab)
        # Sky-lock guard: break acute blue runs before they compound.
        new = tame_blue(new)
        # Drift anchor: weak pull toward beat-0 so small per-beat biases
        # can't integrate into a pale/desat run over 600+ beats.
        new, _lm, _sm, _am, _bm = anchor_pull(new, _ref_l, _ref_s, _ref_a, _ref_b)
        canvases.append(new)
        # Steer next beat's framing toward texture (escape flat skies), clamped
        # to a small step so the camera never yanks across the frame.
        _g = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)
        _gh, _gw = _g.shape[:2]
        _thumb = cv2.resize(_g, (160, max(16, int(160 * _gh / _gw))))
        _f = 1.0 / (1.0 + z_beats[b])
        pans[b + 1] = smooth_pan(pans[b], best_window(_thumb, _f, _f, _rng))
        if (b + 1) % 5 == 0 or b + 1 == len(beats):
            _el = _time.time() - _t_chain
            _done, _tot = b + 1, len(beats)
            _eta = _el / max(_done - 1, 1) * (_tot - _done)
            print(f"[chain] beat {_done}/{_tot} img2img (kick~{beats[b][1]:.2f}) "
                  f"elapsed={_el / 60:.1f}min ETA={_eta / 60:.1f}min", flush=True)
        if (b + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()  # keep VRAM flat over hundred-beat songs
            print(f"[chain] stats beat {b + 1}: L={_lm:.1f} (ref {_ref_l:.1f}) "
                  f"S={_sm:.1f} (ref {_ref_s:.1f}) A={_am:.1f} (ref {_ref_a:.1f}) "
                  f"B={_bm:.1f} (ref {_ref_b:.1f})", flush=True)
    del i2i
    gc.collect()
    torch.cuda.empty_cache()
    return canvases, (revs[0] if revs else "unknown"), seed, pans
