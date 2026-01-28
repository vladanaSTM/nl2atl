#!/usr/bin/env bash
#SBATCH --job-name=nl2atl-llm-judge
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

PYTHON_BIN=${PYTHON_BIN:-python3}
REPO_ROOT=/home/infres/vperlic/projects/nl2atl
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Optional overrides via environment variables.
DATASETS=${DATASETS:-all}
JUDGE_MODELS=${JUDGE_MODELS:-llama-70b}
MODELS_CONFIG=${MODELS_CONFIG:-configs/models.yaml}
PREDICTIONS_DIR=${PREDICTIONS_DIR:-outputs/model_predictions}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/LLM-evaluation}
NO_LLM=${NO_LLM:-0}
OVERWRITE=${OVERWRITE:-0}
HF_MIN_PARAMS_B=${HF_MIN_PARAMS_B:-}
HF_ONLY=${HF_ONLY:-0}

ARGS=(
  --datasets ${DATASETS}
  --models_config ${MODELS_CONFIG}
  --predictions_dir ${PREDICTIONS_DIR}
  --output_dir ${OUTPUT_DIR}
)

if [[ -n "${JUDGE_MODELS}" ]]; then
  # Space-separated list of judge model keys/names.
  ARGS+=(--model ${JUDGE_MODELS})
fi

if [[ "${NO_LLM}" == "1" ]]; then
  ARGS+=(--no_llm)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

if [[ -n "${HF_MIN_PARAMS_B}" ]]; then
  ARGS+=(--hf_min_params_b ${HF_MIN_PARAMS_B})
fi

if [[ "${HF_ONLY}" == "1" ]]; then
  ARGS+=(--hf_only)
fi

$PYTHON_BIN -m src.cli.run_llm_judge "${ARGS[@]}"
