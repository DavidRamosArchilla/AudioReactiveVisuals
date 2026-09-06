#!/bin/bash
# Low-res iteration run: full NewBeginning song at 640x368 for fast
# drift checks (t=10/75/150). Same prompts/NEG/model as run_newbeginning.sh.
# Run FOREGROUND: srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G \
#   --time=01:00:00 --job-name=nb_iter bash run_iter.sh output/nb_iter1.mp4
export LD_LIBRARY_PATH=/home/d.ramos/miniconda/envs/nibbler/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
PY=/home/d.ramos/miniconda/envs/nibbler/bin/python
OUT=${1:-output/nb_iter1.mp4}
EXTRA=${2:-}
STYLE="painterly matte painting, ancient greek marble temples, dramatic textured skies, centered composition, strong contrast, deep shadows, no text, no watermark"
NEG="blurry, low detail, watermark, text, night, flat sky, empty sky, gradient background, oversaturated, neon, monochrome, monochromatic, solid blue, blue tint, cyan tint, flat blue"
$PY -u generate.py --audio NewBeginning.wav --output "$OUT" \
  --tempo 200 --seed 7 --beat-mode chain --width 640 --height 368 \
  --model-id Tongyi-MAI/Z-Image --steps 30 --guidance-scale 4.0 \
  --gen-w 640 --gen-h 368 $EXTRA \
  --init-image output/chain_init.png --negative-prompt "$NEG" \
  --prompt "$STYLE, white marble temple on clifftop, swirling storm clouds with sun rays breaking through, eagles circling, pine branches framing the view" \
  --prompt "$STYLE, pushing through pine branches toward sunlit temple stairs, tiny robed figure climbing, clouds churning orange and teal" \
  --prompt "$STYLE, temple courtyard with tall columns, dusk sky with first stars and dramatic cloud bands, drifting mist" \
  --prompt "$STYLE, looking up steep cliff stairs to a temple against burning sunset clouds, flocks of birds" \
  --prompt "$STYLE, sea of clouds with a floating rock shrine island, shafts of sunlight, distant birds" \
  --prompt "$STYLE, storm clouds parting over a sunlit temple, glowing cloud edges, dark pine branches in the foreground" \
  --prompt "$STYLE, ancient stone arch corridor framing a moonlit acropolis, textured night clouds with moon glow" \
  --prompt "$STYLE, robed philosopher on a rock ledge gazing at a radiant temple above churning clouds, sunbeams" \
  --prompt "$STYLE, waterfall of clouds spilling over a cliff edge beneath a temple, mist and light shafts, pines" \
  --prompt "$STYLE, temple reflected in a still mountain lake at dawn, pink cloud streaks, dark cypress trees" \
  --prompt "$STYLE, close flight between giant marble columns toward a bright courtyard, doves, sun flare" \
  --prompt "$STYLE, panoramic vista with distant temples on islands, dramatic sunset sea of clouds, birds"
