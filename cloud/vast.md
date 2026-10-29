# Running RL on Vast.ai (1 GPU)

Copied from `rlvr-landscape/cloud`, adapted for the chess RL stage. Budget is small, so every
rule is about not paying for idle GPUs and never losing results.

## Rules

1. **Never leave a box running without the watcher.** Every run is followed by
   `cloud/finish_and_destroy.sh` under `caffeinate`. It downloads, verifies the HF checkpoints, and
   only then destroys. If it cannot verify, it *stops* the box instead (disk kept, GPU billing ends).
2. **Destroy, don't stop, when done.** Stopped instances still bill for disk.
3. **After every session:** `uvx vastai show instances` says 0 instances; note the credit
   (`uvx vastai show user`).
4. **Keys never go in the chat or in git.** Run `wandb login` yourself on the box. The Vast key lives
   in `~/.config/vastai/vast_api_key`.

## Machine

```bash
bash cloud/vast.sh offers RTX_A6000
```

- RTX A6000 (48GB) on-demand: ~$0.45-0.9/hr. The 50M model is tiny; the multi-turn vLLM rollout
  dominates step time.
- On-demand, not interruptible. One GPU (`trainer.n_gpus_per_node=1`).
- Bad network = drop the box: setup (uv install + ~200MB checkpoint + 53MB data) should take < 10 min.

## Lifecycle

```bash
bash cloud/vast.sh create OFFER_ID     # onstart runs cloud/setup_instance.sh (branch dev)
bash cloud/vast.sh log ID              # wait for "Setup done"
bash cloud/vast.sh ssh ID
```

On the box (everything under `/workspace`):

```bash
cd /workspace/pre2post-chess && source cloud/env.sh
TOTAL_STEPS=250 TRAIN_BATCH_SIZE=32 GROUP_SIZE=8 \
    nohup bash cloud/run_rl.sh > /workspace/rl.log 2>&1 < /dev/null &
```

`cloud/env.sh` picks the SFT checkpoint (`MODEL_ID`, default `C6p5e18_50m_alpha0.200_beta0.023`
from `chess-pre-to-post/sft_trajectory_no_labels`) and the RL data
(`chess-pre-to-post/chess-rl-data-balanced`). After training, `run_rl.sh` converts the FSDP
checkpoints to `checkpoints_hf_format/global_step_N`.

Then on the laptop:

```bash
caffeinate -ims bash cloud/finish_and_destroy.sh ID HOST PORT \
    /workspace/rl_runs/<EXPERIMENT_NAME> results/rl/<EXPERIMENT_NAME>
```

`uvx vastai ssh-url ID` gives HOST and PORT.

## Killing processes on the box

- `pkill -f '[m]ain_ppo'; ray stop --force`
- vLLM can leave an orphan `VLLM::EngineCore` holding GPU memory: `pkill -9 -f 'VLLM::EngineCore'`,
  then check `nvidia-smi`.

## Reference

The released 8-GPU run of the same 50M model (bs 256 x 8, 2560-token responses) went from eval
reward 0.149 (SFT) to 0.226 at step 250 (`collected_pre_to_post_data/rl_curve_points.csv`).
The 1-GPU default batch is 8x smaller, so per-step progress is not directly comparable.
