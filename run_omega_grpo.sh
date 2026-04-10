#!/bin/bash
# ============================================================================
# Omega Escort GRPO Training — 2x H100 (SLURM interactive session)
# ============================================================================
# Usage: bash run_omega_grpo.sh [optional hydra overrides...]
#
# Recipes:
#   Omega only:
#     keep algorithm.omega_escort_alpha>0 and actor_rollout_ref.actor.entropy_coeff=0
#   Tsallis only:
#     set algorithm.omega_escort_alpha=0 actor_rollout_ref.actor.entropy_type=tsallis actor_rollout_ref.actor.entropy_coeff>0
#   Omega + Tsallis:
#     keep algorithm.omega_escort_alpha>0 and set actor_rollout_ref.actor.entropy_type=tsallis actor_rollout_ref.actor.entropy_coeff>0
#
# Prereqs:
#   - SLURM interactive session with 2 H100 GPUs
#   - Modified vLLM (with output_exact_entropy) installed
#   - verl installed from /fs/ess/PAS2836/pqd/e-rl/mode_collapse/diverlsity/
#   - Data files in parquet format (GSM8K example below)
# ============================================================================
set -x

# ---- Environment ----
# Some cluster modules export ROCm visibility vars even on CUDA runs.
# Clear them so verl workers see a single device-selection mechanism.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true

# ---- Weights & Biases ----
# export WANDB_API_KEY="your-key-here"     # or run `wandb login` beforehand
# export WANDB_ENTITY="your-entity"        # optional: set your W&B team/username

# ---- Paths (EDIT THESE) ----
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"             # HF hub name or local path
TRAIN_DATA="$HOME/data/gsm8k/train.parquet"
VAL_DATA="$HOME/data/gsm8k/test.parquet"

python3 -m verl.trainer.main_ppo \
    \
    `# ---- Algorithm: GRPO + Omega Escort ----` \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.omega_escort_alpha=0.5 \
    algorithm.omega_escort_block_size=64 \
    algorithm.omega_escort_log_clip=3.0 \
    \
    `# ---- Data ----` \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    `# ---- Model ----` \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    `# ---- Actor (training) ----` \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    `# ---- Rollout (vLLM) ----` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    \
    `# ---- Reference Policy ----` \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    `# ---- Trainer ----` \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='omega_escort_grpo' \
    trainer.experiment_name='qwen2.5_3b_omega_alpha0.5' \
    "$@"
