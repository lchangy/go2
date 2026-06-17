#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-/home/orion/miniconda3/envs/isaacgym}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

TASK="${TASK:-go2_moe_cts}"
NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-150000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-go2_moe_cts_payload_com_wrench}"
RUN_NAME="${RUN_NAME:-gpu4096_payload_com_wrench}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
PIPELINE="${PIPELINE:-gpu}"
RL_DEVICE="${RL_DEVICE:-cuda:0}"

cd "${REPO_DIR}"

exec "${CONDA_ENV}/bin/python" legged_gym/scripts/train.py \
  --task="${TASK}" \
  --headless \
  --sim_device="${SIM_DEVICE}" \
  --pipeline="${PIPELINE}" \
  --rl_device="${RL_DEVICE}" \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --experiment_name="${EXPERIMENT_NAME}" \
  --run_name="${RUN_NAME}" \
  "$@"
