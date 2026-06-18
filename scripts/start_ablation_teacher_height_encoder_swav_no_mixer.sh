#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs/go2_moe_cts_ablation/stdout"
RUN_NAME="${RUN_NAME:-a1_no_mixer_height_encoder_swav8p64_rawcritic_1024_400}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
PID_FILE="${LOG_DIR}/${RUN_NAME}.pid"

mkdir -p "${LOG_DIR}"
setsid bash -c '
  pid_file="$1"
  log_file="$2"
  echo "$$" > "${pid_file}"
  exec ./scripts/run_ablation_teacher_height_encoder_swav_no_mixer.sh > "${log_file}" 2>&1 < /dev/null
' _ "${PID_FILE}" "${LOG_FILE}" &
for _ in {1..50}; do
  if [[ -s "${PID_FILE}" ]]; then
    break
  fi
  sleep 0.1
done
PID="$(cat "${PID_FILE}")"

printf 'started pid=%s log=%s\n' "${PID}" "${LOG_FILE}"
