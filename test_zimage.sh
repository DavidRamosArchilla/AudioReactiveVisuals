#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=ztest
#SBATCH -o output/job-%j.log
# Kick test for full (non-turbo) Z-Image: download + time txt2img/img2img.
export LD_LIBRARY_PATH=/home/d.ramos/miniconda/envs/nibbler/lib:$LD_LIBRARY_PATH
nvidia-smi -L
/home/d.ramos/miniconda/envs/nibbler/bin/python scripts/ztest.py
