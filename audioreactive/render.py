"""Frame rendering: procedural mocks, beat-chain flight, legacy cycle flight."""
import hashlib
import os
import random
import shutil
import subprocess

import cv2
import numpy as np

from .camera import pan_targets


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


def procedural_chain_canvases(prompts, n_beats, w, h, seed, beats_per_prompt=4):
    """Mock chain for logic validation (no GPU): fresh art per prompt block."""
    canvases = []
    for b in range(n_beats):
        p = prompts[(b // beats_per_prompt) % len(prompts)]
        canvases.extend(procedural_keyframes([p], w, h, seed + b))
    return canvases


def render_chain(canvases, analysis, z_beats, w, h, fps, seed,
                 gain_cap=0.08, kick_step=0.05, beat_blend=2,
                 intro_fade=0.6, pans=None):
    """Beat-synced forward flight. Within a beat, zoom is linear (constant
    cruise, meter-honest) plus a 2-frame kick step for attack, and it NEVER
    decreases; the next beat's canvas starts from the previous end view, so
    the zoom curve is globally monotonic. A 2-frame dissolve masks any
    residual img2img seam. z_b comes from low-band kick strength (snares
    whisper, kicks punch). Pan is a slow clamped random walk."""
    beats = analysis["beats"]
    beat_dur = analysis["beat_dur"]
    t0 = analysis["t0"]
    kick_env = analysis["kick_env"]
    n = analysis["n_frames"]
    # Pan walk shared with canvas chaining: targets per beat boundary.
    targets = pans if pans is not None else pan_targets(len(beats), seed)
    frames = []

    def view(bi, zoom, fx, fy):
        c = canvases[bi]
        bh, bw = c.shape[:2]
        cw, ch = int(w / zoom), int(h / zoom)
        px = float(np.clip(fx * (bw - cw), 0, bw - cw))
        py = float(np.clip(fy * (bh - ch), 0, bh - ch))
        frame = c[int(py) : int(py) + ch, int(px) : int(px) + cw]
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    for f in range(n):
        t = f / fps
        bi = int((t - t0) // beat_dur) if t >= t0 else 0
        bi = min(max(bi, 0), len(beats) - 1)
        bt = beats[bi][0]
        u = min(1.0, max(0.0, (t - bt) / beat_dur))
        kb = float(beats[bi][1])
        zoom = 1.0 + z_beats[bi] * u + kick_step * kb * min(1.0, u * 4.5)
        f0x, f0y = targets[bi]
        f1x, f1y = targets[bi + 1]
        fx = f0x + (f1x - f0x) * u
        fy = f0y + (f1y - f0y) * u
        frame = view(bi, zoom, fx, fy)
        # Dissolve over the previous beat's end view to hide img2img seams.
        j = int(u * fps * beat_dur + 1e-6)  # 0-based frame index inside beat
        if beat_blend > 0 and bi > 0 and j < beat_blend:
            a = (j + 1) / beat_blend
            prev = view(bi - 1, 1.0 + z_beats[bi - 1], *targets[bi])
            frame = cv2.addWeighted(prev, 1 - a, frame, a, 0)
        ke = float(kick_env[f])
        gain = 1.0 + min(0.10 * ke, gain_cap)
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        # Bloom opening: fade from white (hides early-chain abstraction).
        if intro_fade > 0 and t < intro_fade:
            u0 = t / intro_fade
            a = u0 * u0 * (3.0 - 2.0 * u0)
            frame = cv2.addWeighted(
                np.full_like(frame, 255), 1.0 - a, frame, a, 0)
        frames.append(frame)
    return frames


def plan_segments(n_frames, is_hit, hit_env, n_keys, fps, key_every="auto",
                  tempo=100.0, beats_per_scene=2, t0=0.0):
    """Scene schedule. key_every='auto' = beat grid (beats_per_scene beats per
    scene, anchored at t0); numeric = fixed seconds (legacy). Hits only force a
    switch when extreme; the kick punch is rendered by the spring zoom."""
    seg = np.zeros(n_frames, dtype=int)
    if key_every == "auto":
        scene_len = beats_per_scene * 60.0 / tempo
    else:
        scene_len = float(key_every)
    cur, last_switch = 0, 0
    for f in range(n_frames):
        t = f / fps
        scene_idx = int((t - t0) // scene_len) if t >= t0 else 0
        scheduled = scene_idx % n_keys
        if scheduled != cur and f - last_switch >= int(fps * 0.3):
            cur, last_switch = scheduled, f
        else:
            extreme = bool(is_hit[f]) and float(hit_env[f]) > 0.90
            if extreme and f - last_switch > int(fps * 1.0):
                cur = (cur + 1) % n_keys
                last_switch = f
        seg[f] = cur
    return seg


def render_frames(keys, analysis, seg, w, h, fps, seed, tempo=100.0,
                  beats_per_scene=2, zoom_punch=1.6, zoom_tau=0.10,
                  zoom_cruise=0.62, zoom_max=1.85, headroom=1.60,
                  zoom_step=0.05, blend_frac=0.5):
    """Legacy cycle-mode forward-flight renderer (kept for --no-diffusion
    pipeline tests only; its blend resets cause visible zoom-back).

    Reference measurements (Farneback divergence @30fps): perpetual push-in,
    median ~+1.9%/frame, never zooming out, kick spikes up to ~+9%/frame,
    no hard cuts. So zoom is a *velocity-integrated* log-scale state:

        hit  -> L += zoom_step * strength (instant punch)
                V += zoom_punch * strength (velocity burst)
        V relaxes toward cruise (tau), L += V*dt, zoom = exp(min(L, log max))

    Position never returns to baseline (a spring pulling back to 1.0 is what
    killed the zoom before: it oscillated instead of flying forward).
    Zoom position is tracked PER KEY so the outgoing layer keeps flying
    through the crossfade (no zoom-out pop on scene change); velocity is
    global so the kick rhythm stays continuous across scenes.
    """
    rng = random.Random(seed)
    shakes = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(len(keys) * 8 + 8)]
    energy, hit_env = analysis["energy"], analysis["hit_env"]
    n = analysis["n_frames"]
    dt = 1.0 / fps
    # Pre-scale keys for zoom/pan headroom.
    big = (int(w * headroom), int(h * headroom))
    KB = [cv2.resize(k, big, interpolation=cv2.INTER_LINEAR) for k in keys]
    bw, bh = big
    log_max = float(np.log(zoom_max))
    frames = []
    scene_len = beats_per_scene * 60.0 / tempo
    blend = max(2, min(int(fps * 0.6), int(scene_len * fps * blend_frac)))
    damp = float(np.exp(-dt / max(zoom_tau, 1e-3)))  # velocity relax rate
    Lpos = {}  # key idx -> log-zoom position (each layer flies independently)
    V = float(zoom_cruise)

    def crop(key_idx, he):
        L = min(Lpos.get(key_idx, 0.0), log_max)
        zoom = float(np.exp(L))
        cw, ch = int(w / zoom), int(h / zoom)
        px = (bw - cw) * (0.5 + 0.4 * np.sin(f / (fps * 3.0)))
        py = (bh - ch) * (0.5 + 0.4 * np.cos(f / (fps * 4.0)))
        sx, sy = shakes[(f // 3) % len(shakes)]
        px = float(np.clip(px + sx * he * 14, 0, bw - cw))
        py = float(np.clip(py + sy * he * 14, 0, bh - ch))
        return KB[key_idx][int(py) : int(py) + ch, int(px) : int(px) + cw]

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
        # Kick-punch flight model: instant step + velocity burst on hits.
        if bool(analysis["is_hit"][f]):
            Lpos[cur] = Lpos.get(cur, 0.0) + zoom_step * he
            if prev != cur:
                Lpos[prev] = Lpos.get(prev, 0.0) + zoom_step * he
            V += zoom_punch * he
        cruise_eff = zoom_cruise * (0.75 + 0.5 * e)  # breathe with energy
        V += (cruise_eff - V) * (1.0 - damp)
        Lpos[cur] = min(Lpos.get(cur, 0.0) + V * dt, log_max)
        if prev != cur:
            Lpos[prev] = min(Lpos.get(prev, 0.0) + V * dt, log_max)
        a = crop(prev, he)
        b = crop(cur, he)
        if a.shape != b.shape:  # layers fly at different zooms: match sizes
            a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_LINEAR)
        frame = cv2.addWeighted(a, 1 - t, b, t, 0)
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        # Brightness pulse on hits (capped to avoid blowing highlights).
        gain = 1.0 + min(0.12 * he, 0.08)
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def write_video(frames, fps, audio_path, out_path, dur):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".silent.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))  # frames are RGB
    vw.release()
    ff = shutil.which("ffmpeg")
    if ff is None:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = "ffmpeg"
    cmd = [
        ff, "-y", "-v", "error",
        "-i", tmp, "-i", audio_path,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-shortest", out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(tmp)
