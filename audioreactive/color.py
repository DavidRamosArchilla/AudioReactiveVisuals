"""Color: palette matching, drift defenses (tame_blue, anchor_pull), grade."""
import cv2
import numpy as np

from .config import (
    ANCHOR_PULL_AB,
    ANCHOR_PULL_L,
    ANCHOR_PULL_S_HI,
    ANCHOR_PULL_S_LO,
    TAME_BLUE_DESAT,
    TAME_BLUE_FRAC,
)


def match_color(src, ref):
    """Reinhard LAB transfer: kill per-beat exposure/palette flicker so the
    chain reads as one continuous flight, not strobing generations."""
    s = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    for i in range(3):
        ss = float(s[..., i].std()) + 1e-6
        s[..., i] = (s[..., i] - float(s[..., i].mean())) * (float(r[..., i].std()) / ss) \
            + float(r[..., i].mean())
    return cv2.cvtColor(np.clip(s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def blend_chroma(raw, matched, match_ab):
    """Scale the chroma (a/b) transfer of a LAB match.

    1.0 = full match, 0.0 = luminance only (model keeps its own color; the
    fixed anchor still bounds hue drift). Full chroma match forces every
    canvas toward its rebase crop's palette (often bright bare marble), which
    ratchets color out over hundreds of beats (proven: 0.5 -> flat gray,
    0.0 -> scene-driven color).
    """
    if match_ab >= 1.0:
        return matched
    lm = cv2.cvtColor(matched, cv2.COLOR_RGB2LAB).astype(np.float32)
    ln = cv2.cvtColor(raw, cv2.COLOR_RGB2LAB).astype(np.float32)
    lm[..., 1] = ln[..., 1] * (1.0 - match_ab) + lm[..., 1] * match_ab
    lm[..., 2] = ln[..., 2] * (1.0 - match_ab) + lm[..., 2] * match_ab
    return cv2.cvtColor(np.clip(lm, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def tame_blue(canvas, frac_thresh=TAME_BLUE_FRAC, desat=TAME_BLUE_DESAT):
    """Sky-lock guard: if more than frac_thresh of the canvas is electric
    blue/cyan (H 85-120, high S in OpenCV HSV), desaturate those pixels.
    Gentle by design: it only fires on true flat-field sky-lock, because every
    firing also removes saturation the chain can't get back (overuse drifts
    the whole song toward white). The anchor pull in the chain loop owns
    long-run drift; this just breaks acute blue runs."""
    hsv = cv2.cvtColor(canvas, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S = hsv[..., 0], hsv[..., 1]
    m = (H >= 85) & (H <= 120) & (S > 80)
    if float(m.mean()) > frac_thresh:
        S[m] *= desat
        hsv[..., 1] = S
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return canvas


def anchor_pull(canvas, ref_l, ref_s, ref_a, ref_b,
                pull_l=ANCHOR_PULL_L, pull_s_lo=ANCHOR_PULL_S_LO,
                pull_s_hi=ANCHOR_PULL_S_HI, pull_ab=ANCHOR_PULL_AB):
    """Weak pull of exposure/saturation/hue toward the beat-0 canvas (fixed ref).

    Why: every beat is img2img'd from the previous output, so ANY systematic
    bias (texture-seeking pans cropping bright marble, model brightening pale
    inits, tame_blue eating blue saturation, LAB match wandering hue) integrates
    over ~629 beats into a visible run (measured: L 129->246, S 127->46, plus
    purple/green hue casts). A small per-beat pull toward fixed stats bounds
    total drift (geometric series) while leaving single-scene intent
    (dusk/night prompts) mostly intact. Saturation pull is asymmetric: strong
    when the canvas is duller than ref (kills pale runs), gentle when more
    vivid (never dulls a good scene).
    Returns (canvas, l_mean, s_mean, a_mean, b_mean) for logging.
    """
    lab = cv2.cvtColor(canvas, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_mean = float(lab[..., 0].mean())
    a_mean, b_mean = float(lab[..., 1].mean()), float(lab[..., 2].mean())
    lab[..., 0] += pull_l * (ref_l - l_mean)
    lab[..., 1] += pull_ab * (ref_a - a_mean)
    lab[..., 2] += pull_ab * (ref_b - b_mean)
    canvas = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    hsv = cv2.cvtColor(canvas, cv2.COLOR_RGB2HSV).astype(np.float32)
    s_mean = float(hsv[..., 1].mean())
    if s_mean > 1e-6:
        pull_s = pull_s_lo if s_mean < ref_s else pull_s_hi
        f = 1.0 + pull_s * (ref_s / s_mean - 1.0)
        f = min(1.5, max(0.6, f))
        hsv[..., 1] = np.clip(hsv[..., 1] * f, 0, 255)
        canvas = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return canvas, l_mean, s_mean, a_mean, b_mean


def grade_keys(keys):
    """Tame neon gaming colors toward a cinematic matte-painting palette.

    - Yellow/green band (H 18-48): desaturate (sage skies, no chartreuse).
    - Electric blue/cyan band (H 85-120, high S): desaturate (no
    Klein-blue/cyan blowout or sky-lock wash).
    - Neon magenta/red (H<=8 | H>=150, high S): desaturate.
    - Gentle global saturation x0.90 + highlight rolloff (soft-clip V>235).
    Keeps the azure/amber character, pulls extremes.
    """
    out = []
    for k in keys:
        hsv = cv2.cvtColor(k, cv2.COLOR_RGB2HSV).astype(np.float32)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        m = (H >= 18) & (H <= 48) & (S > 40)
        S[m] *= 0.45
        mb = (H >= 85) & (H <= 120) & (S > 80)
        S[mb] *= 0.55
        m2 = ((H <= 8) | (H >= 150)) & (S > 120)
        S[m2] *= 0.70
        S *= 0.95
        V = np.where(V > 235, 235 + (V - 235) * 0.5, V)
        V = np.clip((V - 128.0) * 1.08 + 128.0, 0, 255)  # gentle S-curve
        hsv = np.stack([H, S, V], axis=-1).astype(np.uint8)
        out.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    return out


def upscale_keys(keys, w, h):
    """Lanczos upscale + light unsharp mask for crisper render frames."""
    out = []
    for k in keys:
        up = cv2.resize(k, (w, h), interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(up, (0, 0), 2.0)
        sharp = cv2.addWeighted(up, 1.20, blur, -0.20, 0)
        out.append(sharp)
    return out
