#!/usr/bin/env bash
# On the box: one 1-GPU RL run from the SFT checkpoint in cloud/env.sh. Writes
# ${SAVE_DIR}/${EXPERIMENT_NAME}/DONE or FAILED at the end.
#
#   nohup bash cloud/run_rl.sh > /workspace/rl.log 2>&1 < /dev/null &
#
# Hyperparameters pass through as env vars (TOTAL_STEPS, TRAIN_BATCH_SIZE, GROUP_SIZE,
# RES_LENGTH, MAX_ENV_CALLS, ...); see rl/verl/1_gpu_bash/run_rl_1gpu.sh.
set -u
cd "$(dirname "$0")/.." && source cloud/env.sh

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_ID}_1gpu_bs${TRAIN_BATCH_SIZE:-32}x${GROUP_SIZE:-8}}"
RUN_DIR="${SAVE_DIR}/${EXPERIMENT_NAME}"
mkdir -p "${RUN_DIR}"; rm -f "${RUN_DIR}/DONE" "${RUN_DIR}/FAILED"

if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    export LOGGERS="${LOGGERS:-['console','mlflow']}"
    echo "wandb not logged in: logging to console + mlflow only"
fi

echo "=== start $(date '+%F %T')"
bash rl/verl/1_gpu_bash/run_rl_1gpu.sh
status=$?
ray stop --force >/dev/null 2>&1 || true

# FSDP checkpoints -> HF format (checkpoints_hf_format/global_step_N), even after a
# failure so partial runs keep their saved steps.
echo "=== converting checkpoints to HF $(date '+%F %T')"
(cd rl/verl && LARGE_DIR="${SAVE_DIR}" bash 8_gpu_bash/convert_to_hf_all.sh)

if [ ${status} -eq 0 ]; then
    touch "${RUN_DIR}/DONE"; echo "=== done $(date '+%F %T')"
else
    touch "${RUN_DIR}/FAILED"; echo "=== failed (exit ${status}) $(date '+%F %T')"
fi
