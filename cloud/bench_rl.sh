#!/usr/bin/env bash
# On the box: time a few RL steps at the paper's batch (256 prompts x 8, 2560-token responses)
# and project 250 steps. Step time is ~proportional to mean response length (fit on a full
# 250-step 1-GPU run: 2 s + 28.9 s per 1000 tokens, r=0.79), and responses grow during RL, so
# the projection scales the measured steps by the lab run's mean length over steps 1-250.
#
#   N_GPUS=4 nohup bash cloud/bench_rl.sh > /workspace/bench.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.." && source cloud/env.sh

export N_GPUS="${N_GPUS:-1}"
export TOTAL_STEPS="${BENCH_STEPS:-6}" TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}" GROUP_SIZE="${GROUP_SIZE:-8}"
export RES_LENGTH="${RES_LENGTH:-2560}" SAVE_FREQ=-1 MAX_NUM_SEQS="${MAX_NUM_SEQS:-2048}"
export LOGGERS="['console']" EXPERIMENT_NAME="bench_${N_GPUS}gpu" SAVE_DIR="${WORK}/bench_runs"
LAB_MEAN_RESPONSE_LEN="${LAB_MEAN_RESPONSE_LEN:-968}"   # rl_token_curve.csv, 50m alpha0.200, steps 1-250

log="${SAVE_DIR}/${EXPERIMENT_NAME}/logs/log_${EXPERIMENT_NAME}.log"
# Validation never runs (TEST_FREQ=-1), but verl filters the val file at startup: keep it tiny.
export EVAL_DATA_PATH="${WORK}/data/val_tiny.parquet"
python -c "import pandas as pd; pd.read_parquet('${TRAIN_DATA_PATH}').head(64).to_parquet('${EVAL_DATA_PATH}')"
bash rl/verl/1_gpu_bash/run_rl_1gpu.sh
ray stop --force >/dev/null 2>&1 || true

echo "=== bench ${N_GPUS} GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
grep -oE 'step:[0-9]+ .*' "${log}" | tr ' ' '\n' \
    | grep -E '^step:|^timing_s/step:|^timing_s/gen:|^response_length/mean:' | paste - - - - \
    | python3 -c "
import sys
rows = []
for line in sys.stdin:
    d = dict(kv.split(':', 1) for kv in line.split())
    rows.append((int(d['step']), float(d['response_length/mean']), float(d['timing_s/gen']), float(d['timing_s/step'])))
    print(f'step {rows[-1][0]}  resp {rows[-1][1]:.0f}  gen {rows[-1][2]:.1f}s  step {rows[-1][3]:.1f}s')
use = [r for r in rows if r[0] > 1] or rows          # step 1 includes warm-up
t = sum(r[3] for r in use) / len(use)
L = sum(r[1] for r in use) / len(use)
per_step = t * ${LAB_MEAN_RESPONSE_LEN} / L
print(f'=== measured {t:.1f}s/step at resp {L:.0f} -> {per_step:.1f}s/step at resp ${LAB_MEAN_RESPONSE_LEN} '
      f'-> 250 steps = {250 * per_step / 3600:.2f} h')
"
