#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=nb_smoke2
#SBATCH -o output/job-%j.log
# Smoke test for de-blue + smooth-pan fixes: sample audio (~10s, 34 beats),
# same full-model settings/prompts as run_newbeginning.sh.
export LD_LIBRARY_PATH=/home/d.ramos/miniconda/envs/nibbler/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
PY=/home/d.ramos/miniconda/envs/nibbler/bin/python
STYLE="painterly matte painting, ancient greek marble temples, dramatic textured skies, centered composition, strong contrast, deep shadows, no text, no watermark"
NEG="blurry, low detail, watermark, text, night, flat sky, empty sky, gradient background, oversaturated, neon, monochrome, monochromatic, solid blue, blue tint, cyan tint, flat blue"
$PY generate.py --audio audio_sample.wav --output output/nb_smoke2.mp4 \
  --tempo 200 --seed 7 --beat-mode chain --width 960 --height 544 \
  --model-id Tongyi-MAI/Z-Image --steps 30 --guidance-scale 4.0 \
  --gen-w 960 --gen-h 544 \
  --init-image output/chain_init.png --negative-prompt "$NEG" \
  --prompt "$STYLE, white marble temple on clifftop, swirling storm clouds with sun rays breaking through, eagles circling, pine branches framing the view" \
  --prompt "$STYLE, pushing through pine branches toward sunlit temple stairs, tiny robed figure climbing, clouds churning orange and teal" \
  --prompt "$STYLE, temple courtyard with tall columns, dusk sky with first stars and dramatic cloud bands, drifting mist" \
  --prompt "$STYLE, looking up steep cliff stairs to a temple against burning sunset clouds, flocks of birds" \
  --prompt "$STYLE, sea of clouds with a floating rock shrine island, shafts of sunlight, distant birds" \
  --prompt "$STYLE, storm clouds parting over a sunlit temple, glowing cloud edges, dark pine branches in the foreground" \
  --prompt "$STYLE, ancient stone arch corridor framing a moonlit acropolis, textured night clouds with moon glow" \
  --prompt "$STYLE, robed philosopher on a rock ledge gazing at a radiant temple above churning clouds, sunbeams"
