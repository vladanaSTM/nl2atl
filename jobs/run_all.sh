#!/bin/bash
#SBATCH --job-name=atl_experiments
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

echo "Starting experiments at: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source ~/miniconda/etc/profile.d/conda.sh
conda activate atl_project

cd ~/projects/atl-comparison

python run_all_experiments.py

echo "Finished at: $(date)"