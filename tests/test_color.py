"""Pure-function tests: color ops (no GPU needed)."""
import cv2
import numpy as np

from audioreactive.color import anchor_pull, blend_chroma, match_color, tame_blue


def _lab_means(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(float)
    return [float(lab[..., i].mean()) for i in range(3)]


def test_match_color_copies_ref_means():
    rng = np.random.default_rng(3)
    src = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    ref = np.full((32, 32, 3), (200, 50, 30), np.uint8)
    out = match_color(src, ref)
    for got, want in zip(_lab_means(out), _lab_means(ref)):
        assert abs(got - want) < 2.0


def test_blend_chroma_zero_keeps_raw_hue():
    rng = np.random.default_rng(4)
    raw = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    ref = np.full((32, 32, 3), (10, 200, 30), np.uint8)
    out = blend_chroma(raw, match_color(raw, ref), 0.0)
    lo, lr = _lab_means(out), _lab_means(raw)
    assert abs(lo[0] - _lab_means(ref)[0]) < 3.0  # luminance follows ref
    assert abs(lo[1] - lr[1]) < 3.0 and abs(lo[2] - lr[2]) < 3.0  # hue stays raw


def test_tame_blue_fires_only_on_blue_fields():
    cyan = np.zeros((32, 32, 3), np.uint8)
    cyan[:, :] = [0, 200, 255]
    out = tame_blue(cyan)
    s_before = cv2.cvtColor(cyan, cv2.COLOR_RGB2HSV)[..., 1].mean()
    s_after = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)[..., 1].mean()
    assert s_after < s_before
    gray = np.full((32, 32, 3), 128, np.uint8)
    assert np.array_equal(tame_blue(gray), gray)


def test_anchor_pull_converges_pale_canvas():
    pale = np.full((32, 32, 3), 230, np.uint8)
    cur = pale
    for _ in range(30):
        cur, *_ = anchor_pull(cur, 130.0, 100.0, 128.0, 128.0)
    assert abs(_lab_means(cur)[0] - 130.0) < 15.0  # bounded near ref, was 230
