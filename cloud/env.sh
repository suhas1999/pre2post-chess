# Source on the GPU box before running anything:  source cloud/env.sh
# Everything lives under $WORK so it survives a stop/start.

export WORK="${WORK:-/workspace}"
export REPO_ROOT="${WORK}/pre2post-chess"
export VENV="${WORK}/venv"

export UV_CACHE_DIR="${WORK}/.cache/uv"
export HF_HOME="${WORK}/.cache/huggingface"
export VLLM_CACHE_ROOT="${WORK}/.cache/vllm"
export TRITON_CACHE_DIR="${WORK}/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${WORK}/.cache/torchinductor"
export RAY_TMPDIR="/tmp/ray"

export FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# RL starting point: the released SFT checkpoint for one pretrain spec.
export SFT_HF_REPO="${SFT_HF_REPO:-chess-pre-to-post/sft_trajectory_no_labels}"
export MODEL_ID="${MODEL_ID:-C6p5e18_50m_alpha0.200_beta0.023}"
export SFT_ROOT="${WORK}/sft"
export BASE_MODEL="${SFT_ROOT}/${MODEL_ID}"

export RL_DATA_REPO="${RL_DATA_REPO:-chess-pre-to-post/chess-rl-data-balanced}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${WORK}/data/train_v4_dataset_balanced_multi_turn.parquet}"
export SAVE_DIR="${SAVE_DIR:-${WORK}/rl_runs}"

export WANDB_PROJECT="${WANDB_PROJECT:-chess_rl}"
export WANDB_DIR="${WORK}/wandb"

if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
fi
