#!/usr/bin/env bash
# On the box: RL with the released 8-GPU settings (rl/verl/8_gpu_bash/sweep_multi_turn.sh +
# run_multi_turn.sh: 256 prompts x 8, mini-batch 256, lr 1e-5, KL 0.001 low_var_kl, 2560-token
# responses, max 6 env calls), then eval the last checkpoint on the rebuilt puzzle test sets.
# Writes /workspace/ALL_DONE (or /workspace/ALL_FAILED) at the end.
#
#   N_GPUS=4 nohup bash cloud/run_paper_setup.sh > /workspace/paper.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.." && source cloud/env.sh
rm -f "${WORK}/ALL_DONE" "${WORK}/ALL_FAILED"

export N_GPUS="${N_GPUS:-4}" TOTAL_STEPS="${TOTAL_STEPS:-250}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-$((N_GPUS + 4))}"   # eval_rl.sh defaults to 4: too few for 4 GPU workers
export TRAIN_BATCH_SIZE=256 GROUP_SIZE=8 RES_LENGTH=2560 ACTOR_LR=1e-5 KL_LOSS_COEF=0.001
export PPO_MICRO_BATCH_SIZE_PER_GPU=32 MAX_NUM_SEQS=2048 MAX_ENV_CALLS=6 SAVE_FREQ=50
export EXPERIMENT_NAME="${MODEL_ID}_paper_bs256x8_${N_GPUS}gpu"
# Validation never runs; keep the val file tiny so startup does not filter 53k prompts twice.
export EVAL_DATA_PATH="${WORK}/data/val_tiny.parquet"
python -c "import pandas as pd; pd.read_parquet('${TRAIN_DATA_PATH}').head(64).to_parquet('${EVAL_DATA_PATH}')"

bash cloud/run_rl.sh
RUN_DIR="${SAVE_DIR}/${EXPERIMENT_NAME}"
CKPT="${RUN_DIR}/checkpoints_hf_format/global_step_${TOTAL_STEPS}"
if [ ! -f "${RUN_DIR}/DONE" ] || [ ! -d "${CKPT}" ]; then
    echo "=== training did not finish"; touch "${WORK}/ALL_FAILED"; exit 1
fi
for f in tokenizer.py tokenizer_config.json vocab.json special_tokens_map.json; do
    [ -f "${CKPT}/${f}" ] || cp "${BASE_MODEL}/${f}" "${CKPT}/"
done

GPUS="$(seq -s, 0 $((N_GPUS - 1)) | sed 's/,$//')" \
    bash cloud/eval_rl.sh "paper_step${TOTAL_STEPS}=${CKPT}"
touch "${WORK}/ALL_DONE"
