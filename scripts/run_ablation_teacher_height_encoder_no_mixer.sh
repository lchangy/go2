#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-/home/orion/miniconda3/envs/isaacgym}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

export GO2_POLICY_USE_HEIGHT_ENCODER="${GO2_POLICY_USE_HEIGHT_ENCODER:-1}"
export GO2_POLICY_TEACHER_CONTEXT_MODE="${GO2_POLICY_TEACHER_CONTEXT_MODE:-current_privileged}"
export GO2_POLICY_CRITIC_USE_PRIVILEGED_OBS="${GO2_POLICY_CRITIC_USE_PRIVILEGED_OBS:-1}"
export GO2_POLICY_DETACH_CRITIC_CONTEXT="${GO2_POLICY_DETACH_CRITIC_CONTEXT:-1}"

NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-400}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-go2_moe_cts_ablation}"
RUN_NAME="${RUN_NAME:-a1_no_mixer_height_encoder_rawcritic_1024_400}"

cd "${REPO_DIR}"

exec "${CONDA_ENV}/bin/python" legged_gym/scripts/train.py \
  --task=go2_moe_cts \
  --headless \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --experiment_name="${EXPERIMENT_NAME}" \
  --run_name="${RUN_NAME}"
