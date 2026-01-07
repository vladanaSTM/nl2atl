#!/bin/bash
#SBATCH --job-name=atl_single
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00

# Usage: sbatch jobs/run_single.sh qwen-3b --finetuned --few_shot

MODEL=$1
shift
EXTRA_ARGS=$@

echo "Running: $MODEL $EXTRA_ARGS"

source ~/miniconda/etc/profile.d/conda.sh
conda activate atl_project

cd ~/projects/atl-comparison

python run_single_experiment.py --model $MODEL $EXTRA_ARGS