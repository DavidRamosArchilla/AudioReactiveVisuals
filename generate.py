"""Audio-reactive video generator — thin entry shim (logic lives in audioreactive/).

Usage:
    python generate.py --audio audio_sample.wav --output output/demo.mp4
    python generate.py --audio audio_sample.wav --output output/test.mp4 \
        --prompt "abstract thing" --max-duration 4 --width 640 --height 360
    python generate.py --audio audio_sample.wav --output output/fast.mp4 \
        --no-diffusion --max-duration 4   # pipeline check without GPU gen

Equivalent module entry: python -m audioreactive (... same flags ...)
"""
import sys

from audioreactive.cli import main

if __name__ == "__main__":
    sys.exit(main())
