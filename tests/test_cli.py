"""CLI + audio smoke tests (CPU-only; procedural path, no diffusion)."""
import wave
from types import SimpleNamespace

import numpy as np

from audioreactive.audio import analyze_audio
from audioreactive.cli import base_meta, build_parser


def test_parser_defaults_match_config():
    from audioreactive import config
    args = build_parser().parse_args(["--audio", "a.wav", "--output", "o.mp4"])
    assert args.model_id == config.MODEL_ID
    assert args.match_ab == 0.0
    assert args.beat_strength == config.BEAT_STRENGTH
    assert args.seed_tries == "7"


def test_analyze_audio_synthetic_wav(tmp_path):
    sr, dur, bpm = 22050, 4.0, 120
    t = np.arange(int(sr * dur)) / sr
    beat = 60.0 / bpm
    y = (0.5 * np.sin(2 * np.pi * 55 * t)
         * np.maximum(0, np.sin(2 * np.pi * t / beat)) ** 8)
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    wav = str(tmp_path / "kick.wav")
    with wave.open(wav, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())
    an = analyze_audio(wav, fps=30, tempo=bpm)
    assert an["tempo"] == bpm
    assert 7 <= len(an["beats"]) <= 9  # 4s @120BPM = 8 beat slots from anchor
    assert len(an["hits"]) > 0


def test_base_meta_records_model_and_negative():
    args = SimpleNamespace(model_id="Tongyi-MAI/Z-Image", no_diffusion=False,
                           seed=7, negative_prompt="blurry, text", fps=30,
                           width=640, height=368, beat_mode="chain")
    an = {"hits": [(0.1, 0.9)], "beats": [[0.1, 1.0, 0.0]],
          "tempo": 200.0, "tempo_source": "manual"}
    meta = base_meta(args, ["abstract thing"], an)
    assert meta["model"] == "Tongyi-MAI/Z-Image"  # honors --model-id, not default
    assert meta["negative_prompt"] == "blurry, text"
    assert meta["size"] == [640, 368]
