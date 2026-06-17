#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x /opt/conda/envs/unitree-rl/bin/python ]]; then
  DEFAULT_CONDA_ENV="/opt/conda/envs/unitree-rl"
else
  DEFAULT_CONDA_ENV="/home/orion/miniconda3/envs/isaacgym"
fi
CONDA_ENV="${CONDA_ENV:-${DEFAULT_CONDA_ENV}}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

export GO2_POLICY_USE_HEIGHT_ENCODER="${GO2_POLICY_USE_HEIGHT_ENCODER:-0}"
export GO2_POLICY_TEACHER_CONTEXT_MODE="${GO2_POLICY_TEACHER_CONTEXT_MODE:-current_privileged}"
export GO2_POLICY_CRITIC_USE_PRIVILEGED_OBS="${GO2_POLICY_CRITIC_USE_PRIVILEGED_OBS:-1}"
export GO2_POLICY_DETACH_CRITIC_CONTEXT="${GO2_POLICY_DETACH_CRITIC_CONTEXT:-1}"

export GO2_POLICY_USE_STABLE_SWAV="${GO2_POLICY_USE_STABLE_SWAV:-1}"
export GO2_POLICY_STABLE_LATENT_DIM="${GO2_POLICY_STABLE_LATENT_DIM:-8}"
export GO2_POLICY_SWAV_NUM_PROTOTYPES="${GO2_POLICY_SWAV_NUM_PROTOTYPES:-64}"
export GO2_ALG_STABLE_SWAV_COEF="${GO2_ALG_STABLE_SWAV_COEF:-0.01}"
export GO2_ALG_NUM_MINI_BATCHES="${GO2_ALG_NUM_MINI_BATCHES:-16}"
export GO2_ALG_LEARNING_RATE="${GO2_ALG_LEARNING_RATE:-0.001}"
export GO2_ALG_STUDENT_ENCODER_LEARNING_RATE="${GO2_ALG_STUDENT_ENCODER_LEARNING_RATE:-0.001}"

export GO2_POLICY_USE_ACTOR_FILM="${GO2_POLICY_USE_ACTOR_FILM:-1}"

NUM_ENVS="${NUM_ENVS:-6500}"
MAX_ITERATIONS="${MAX_ITERATIONS:-400}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-go2_moe_cts_ablation}"
RUN_NAME="${RUN_NAME:-a2_no_mixer_rawpriv_swav8p64_sfilm_6500_mb16_lr1e3_400}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
PIPELINE="${PIPELINE:-gpu}"
RL_DEVICE="${RL_DEVICE:-cuda:0}"

cd "${REPO_DIR}"

exec "${CONDA_ENV}/bin/python" legged_gym/scripts/train.py \
  --task=go2_moe_cts \
  --headless \
  --sim_device="${SIM_DEVICE}" \
  --pipeline="${PIPELINE}" \
  --rl_device="${RL_DEVICE}" \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --experiment_name="${EXPERIMENT_NAME}" \
  --run_name="${RUN_NAME}" \
  "$@"
