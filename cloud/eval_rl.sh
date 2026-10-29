#!/usr/bin/env bash
# On the box: evaluate HF checkpoints on the rebuilt puzzle test sets (cloud/rebuild_test_sets.py)
# with rl/verl/eval_bash/verl_eval.sh, then print mean reward (pass@1) per benchmark.
#
#   nohup bash cloud/eval_rl.sh NAME=/path/to/hf_ckpt [NAME=/path ...] > /workspace/eval.log 2>&1 < /dev/null &
#
# Defaults match the released eval runs: 2560-token responses, temperature 1.0. N_SAMPLES=16.
set -u
cd "$(dirname "$0")/.." && source cloud/env.sh

export EVAL_DATA_DIR="${EVAL_DATA_DIR:-${WORK}/data/puzzles}"
export RES_LENGTH="${RES_LENGTH:-2560}" TEMPERATURE="${TEMPERATURE:-1.0}" N_SAMPLES="${N_SAMPLES:-16}"
EVAL_ROOT="${EVAL_ROOT:-${WORK}/eval_results}"
# Same thread caps as rl/verl/1_gpu_bash/run_rl_1gpu.sh (container pid limit).
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 TOKENIZERS_PARALLELISM=false
export RAY_num_server_call_thread=1 GRPC_POLL_STRATEGY=poll
export EVAL_EXTRA_ARGS="ray_init.num_cpus=${RAY_NUM_CPUS:-4}"

for spec in "$@"; do
    name="${spec%%=*}"; model="${spec#*=}"
    echo "=== eval ${name} (${model}) $(date '+%F %T')"
    (cd rl/verl/eval_bash && MODEL_PATH="${model}" OUTPUT_DIR="${EVAL_ROOT}/${name}" \
        EXPERIMENT_NAME="${RES_LENGTH}len_t${TEMPERATURE}_n${N_SAMPLES}" bash verl_eval.sh) \
        || echo "=== eval ${name} FAILED"
    ray stop --force >/dev/null 2>&1 || true
done

echo "=== summary $(date '+%F %T')"
python - "${EVAL_ROOT}" "$@" <<'PY'
import json, sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
for spec in sys.argv[2:]:
    name = spec.split("=", 1)[0]
    scores = defaultdict(list)
    for f in (root / name).glob("*/generations/*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            scores[r["data_source"]].append(float(r["score"]))
    if not scores:
        print(f"{name}: no generations"); continue
    per = {k: sum(v) / len(v) for k, v in sorted(scores.items())}
    pooled = [s for v in scores.values() for s in v]
    print(f"{name}: pooled {sum(pooled) / len(pooled):.4f}  mean over benchmarks {sum(per.values()) / len(per):.4f}  "
          + "  ".join(f"{k} {v:.4f} (n={len(scores[k])})" for k, v in per.items()))
PY
