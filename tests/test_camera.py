"""Pure-function tests: camera geometry (no GPU, no audio needed)."""
import random

import numpy as np

from audioreactive.camera import best_window, center_crop_end, pan_targets, smooth_pan
from audioreactive.config import PAN_MAX_STEP


def test_center_crop_end_geometry():
    c = np.zeros((100, 200, 3), np.uint8)
    crop = center_crop_end(c, 1.0, (0.5, 0.5))  # 2x zoom -> half size, centered
    assert crop.shape == (50, 100, 3)
    crop = center_crop_end(c, 0.0, (0.0, 0.0))  # no zoom -> full canvas
    assert crop.shape == (100, 200, 3)


def test_smooth_pan_clamps_teleports():
    out = smooth_pan((0.5, 0.5), (1.0, 1.0))
    import math
    assert math.hypot(out[0] - 0.5, out[1] - 0.5) <= PAN_MAX_STEP + 1e-9


def test_smooth_pan_keeps_small_steps():
    assert smooth_pan((0.5, 0.5), (0.52, 0.53)) == (0.52, 0.53)


def test_pan_targets_bounded_and_deterministic():
    a = pan_targets(34, 7)
    b = pan_targets(34, 7)
    assert a == b and len(a) == 35
    assert all(0.05 <= x <= 0.95 and 0.05 <= y <= 0.95 for x, y in a)


def test_best_window_seeks_texture():
    gray = np.full((64, 64), 128, np.uint8)
    gray[48:64, 48:64] = np.random.default_rng(0).integers(0, 255, (16, 16)).astype(np.uint8)
    fx, fy = best_window(gray, 0.25, 0.25, random.Random(7))
    assert fx > 0.5 and fy > 0.5  # textured corner wins over flat field
