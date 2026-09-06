#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=audioreactive
#SBATCH -o output/job-%j.log
# Full diffusion render on the gpu node (login node has no GPU).
export LD_LIBRARY_PATH=/home/d.ramos/miniconda/envs/nibbler/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
PY=/home/d.ramos/miniconda/envs/nibbler/bin/python
STYLE="painterly matte painting, ancient greek marble temples on sunlit cliffs, windswept pine trees, dramatic textured sky, white clouds, soft sunlight, strong contrast, deep shadows, dramatic light, no text, no watermark"
NEG="blurry, low detail, watermark, text, night, darkness, yellow sky, green sky, oversaturated, neon, bright blue, electric blue, pure red, monochrome, monochromatic, solid blue, blue tint, cyan tint, flat blue"
$PY generate.py --audio audio_sample.wav --output output/chain_final.mp4 \
  --tempo 200 --seed 7 --beat-mode chain --init-image output/chain_init.png --negative-prompt "$NEG" \
  --prompt "$STYLE, wide establishing shot, white marble temple on sunlit clifftop above a sea of white clouds, giant pines framing the view" \
  --prompt "$STYLE, camera pushing through pine branches toward bright stone temple stairs, tiny robed figure climbing in sunlight" \
  --prompt "$STYLE, sunlit temple courtyard with tall marble columns, textured sky, detailed white clouds, light mist" \
  --prompt "$STYLE, looking up along steep cliff stairs to a white temple against a bright textured sky" \
  --prompt "$STYLE, sea of white clouds with a floating rock island and small shrine, birds in sunlight, textured sky" \
  --prompt "$STYLE, sunbeams breaking through detailed white clouds over a marble temple, pine branch in the foreground, textured sky" \
  --prompt "$STYLE, ancient stone arch corridor framing a distant sunlit acropolis under a textured sky" \
  --prompt "$STYLE, robed philosopher on a rock ledge gazing at a radiant white temple above the clouds, textured sky"
