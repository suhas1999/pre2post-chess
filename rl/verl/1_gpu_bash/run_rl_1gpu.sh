#!/bin/bash

# Single-GPU multi-turn GRPO run (e.g. one RTX A6000 on Vast.ai). Same training
# command as 8_gpu_bash/run_multi_turn.sh, minus conda/SLURM, with one GPU and a
# smaller default batch so a 50M model finishes ~250 steps in a few hours.
#
#   BASE_MODEL=/workspace/sft/C6p5e18_50m_alpha0.200_beta0.023 \
#   TRAIN_DATA_PATH=/workspace/data/train_v4_dataset_balanced_multi_turn.parquet \
#   SAVE_DIR=/workspace/rl_runs EXPERIMENT_NAME=50m_a0.200 bash run_rl_1gpu.sh

required_vars=(BASE_MODEL TRAIN_DATA_PATH SAVE_DIR EXPERIMENT_NAME)
for v in "${required_vars[@]}"; do
  [[ -z "${!v:-}" ]] && { echo "[ERROR] Missing required env var: ${v}" >&2; exit 1; }
done

# Validation is off (TEST_FREQ=-1; verl otherwise validates on the last step before saving),
# but verl still loads a val file, so the train file is a fine default.
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${TRAIN_DATA_PATH}}"
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-${PKG_ROOT}/reward_function_multiturn.py}"
PROJECT_NAME="${PROJECT_NAME:-chess_rl}"

TOTAL_STEPS="${TOTAL_STEPS:-250}"
ACTOR_LR="${ACTOR_LR:-1e-5}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
GROUP_SIZE="${GROUP_SIZE:-8}"
RES_LENGTH="${RES_LENGTH:-2560}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:--1}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-32}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
MAX_ENV_CALLS="${MAX_ENV_CALLS:-6}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"
USE_MULTITURN="${USE_MULTITURN:-True}"
ENABLE_THINKING_MODE="${ENABLE_THINKING_MODE:-True}"
LOGGERS="${LOGGERS:-['console','mlflow','wandb']}"

RUN_DIR="${SAVE_DIR}/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
LOG_FILE="${RUN_DIR}/logs/log_${EXPERIMENT_NAME}.log"
MLFLOW_DIR="${RUN_DIR}/mlflow"
ROLLOUT_DIR="${RUN_DIR}/rollouts/training"
VALIDATION_DIR="${RUN_DIR}/rollouts/validation"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Rented containers can cap threads (Vast: pids.max=1024) while nproc reports the whole
# host, so Ray/torch/vLLM pools sized to nproc die with "Resource temporarily unavailable".
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TORCHINDUCTOR_COMPILE_THREADS=1 TOKENIZERS_PARALLELISM=false
export RAY_num_server_call_thread=1 GRPC_POLL_STRATEGY=poll
export RAY_num_prestart_python_workers=0 RAY_prestart_worker_first_driver=0
export REWARD_MODEL_TYPE="RULE_BASED"
export MLFLOW_TRACKING_URI="sqlite:///${MLFLOW_DIR}/mlflow.db"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

cd "${PKG_ROOT}"
mkdir -p "${CHECKPOINT_DIR}" "$(dirname "${LOG_FILE}")" "${MLFLOW_DIR}" "${ROLLOUT_DIR}" "${VALIDATION_DIR}"

echo "Starting 1-GPU RL job ${EXPERIMENT_NAME}"
echo "Model=${BASE_MODEL}  steps=${TOTAL_STEPS}  batch=${TRAIN_BATCH_SIZE}x${GROUP_SIZE}  res=${RES_LENGTH}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  "data.train_files=${TRAIN_DATA_PATH}" \
  "data.val_files=['${EVAL_DATA_PATH}']" \
  "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=512 \
  "data.max_response_length=${RES_LENGTH}" \
  data.filter_overlong_prompts=True \
  "data.truncation='error'" \
  "+data.use_multiturn=${USE_MULTITURN}" \
  "+data.use_chat_template=False" \
  data.trust_remote_code=True \
  "actor_rollout_ref.model.path=${BASE_MODEL}" \
  "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.trust_remote_code=True \
  "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  "actor_rollout_ref.rollout.dtype=${ROLLOUT_DTYPE}" \
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
  "actor_rollout_ref.rollout.n=${GROUP_SIZE}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
  "actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}" \
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.interactive_mode.enable=True \
  "+actor_rollout_ref.rollout.multi_turn.max_env_calls=${MAX_ENV_CALLS}" \
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  algorithm.use_kl_in_reward=False \
  reward_model.reward_manager=batch \
  "custom_reward_function.path=${CUSTOM_REWARD_PATH}" \
  custom_reward_function.name=compute_score_batch \
  trainer.critic_warmup=0 \
  trainer.default_hdfs_dir=null \
  "trainer.default_local_dir=${CHECKPOINT_DIR}" \
  "trainer.logger=${LOGGERS}" \
  "trainer.project_name=${PROJECT_NAME}" \
  "trainer.experiment_name=${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=1 \
  "ray_init.num_cpus=${RAY_NUM_CPUS}" \
  trainer.nnodes=1 \
  "trainer.save_freq=${SAVE_FREQ}" \
  "trainer.test_freq=${TEST_FREQ}" \
  trainer.total_epochs=9999 \
  "trainer.total_training_steps=${TOTAL_STEPS}" \
  "trainer.rollout_data_dir=${ROLLOUT_DIR}" \
  "trainer.validation_data_dir=${VALIDATION_DIR}" \
  trainer.val_before_train=False \
  $( [[ "${ENABLE_THINKING_MODE}" == "True" ]] && echo "actor_rollout_ref.rollout.thinking_mode.enable=True" ) \
  2>&1 | tee "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
