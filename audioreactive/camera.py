"""Camera: rebase crops, texture-seeking pans, temporal pan clamping."""
import math
import random

import cv2
import numpy as np

from .config import BEST_WINDOW_JITTER, PAN_MAX_STEP


def center_crop_end(canvas, z, f=(0.5, 0.5)):
    """Rebase view: crop at end-of-beat zoom AND end pan (what viewer saw)."""
    h, w = canvas.shape[:2]
    cw, ch = max(1, int(w / (1.0 + z))), max(1, int(h / (1.0 + z)))
    px = min(max(f[0] * (w - cw), 0), w - cw)
    py = min(max(f[1] * (h - ch), 0), h - ch)
    return canvas[int(py) : int(py) + ch, int(px) : int(px) + cw]


def pan_targets(n_beats, seed):
    """Slow clamped random walk; target per beat boundary (n_beats+1).
    Fallback for procedural mocks; diffusion uses texture-seeking pans."""
    rng = random.Random(seed)
    fx, fy, targets = 0.5, 0.5, [(0.5, 0.5)]
    for _ in range(n_beats):
        fx = min(0.95, max(0.05, fx + rng.uniform(-0.08, 0.08)))
        fy = min(0.95, max(0.05, fy + rng.uniform(-0.08, 0.08)))
        targets.append((fx, fy))
    return targets


def best_window(gray_thumb, frac_w, frac_h, rng, grid=12,
                jitter=BEST_WINDOW_JITTER):
    """Coarsest-texture window (fractional coords): keeps the rebase crop on
    content (temples/trees), never stranded in flat sky."""
    h, w = gray_thumb.shape[:2]
    tex = np.abs(cv2.Laplacian(gray_thumb, cv2.CV_64F))
    tex = cv2.GaussianBlur(tex, (0, 0), 2.0)
    ii = cv2.integral(tex)
    ww, hh = frac_w * w, frac_h * h
    best, bv = (0.5 - frac_w / 2, 0.5 - frac_h / 2), -1.0
    for gy in range(grid):
        for gx in range(grid):
            fx = gx / (grid - 1) if grid > 1 else 0.5
            fy = gy / (grid - 1) if grid > 1 else 0.5
            x0, y0 = int(fx * (w - ww)), int(fy * (h - hh))
            x1, y1 = min(w, x0 + int(ww)), min(h, y0 + int(hh))
            v = float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])
            if v > bv:
                bv, best = v, (fx, fy)
    jx = min(0.95, max(0.05, best[0] + rng.uniform(-jitter, jitter)))
    jy = min(0.95, max(0.05, best[1] + rng.uniform(-jitter, jitter)))
    return (jx, jy)


def smooth_pan(prev, raw, max_step=PAN_MAX_STEP):
    """Clamp per-beat pan displacement: best_window picks a global texture max
    independently per beat, which can teleport across the frame (measured jumps
    up to ~1.25 in the HD render -> abrupt camera yanks). Clamping keeps the
    flight smooth while still drifting toward texture."""
    dx, dy = raw[0] - prev[0], raw[1] - prev[1]
    d = math.hypot(dx, dy)
    if d <= max_step or d == 0:
        return (min(0.95, max(0.05, raw[0])), min(0.95, max(0.05, raw[1])))
    k = max_step / d
    return (min(0.95, max(0.05, prev[0] + dx * k)),
            min(0.95, max(0.05, prev[1] + dy * k)))
