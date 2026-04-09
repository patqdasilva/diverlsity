#!/bin/bash
# Recipes:
#   Omega only:
#     keep algorithm.omega_escort_alpha>0 and actor_rollout_ref.actor.entropy_coeff=0
#   Tsallis only:
#     set algorithm.omega_escort_alpha=0 actor_rollout_ref.actor.entropy_type=tsallis actor_rollout_ref.actor.entropy_coeff>0
#   Omega + Tsallis:
#     keep algorithm.omega_escort_alpha>0 and set actor_rollout_ref.actor.entropy_type=tsallis actor_rollout_ref.actor.entropy_coeff>0
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-3B-Instruct"}
ERDOS_MODULE="examples/erdos_min_overlap/erdos_puct.py"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.omega_escort_alpha=0.5 \
    algorithm.omega_escort_block_size=64 \
    algorithm.omega_escort_log_clip=3.0 \
    data.train_files=erdos_train \
    data.val_files=erdos_val \
    data.custom_cls.path=${ERDOS_MODULE} \
    data.custom_cls.name=ErdosArchiveDataset \
    data.sampler.class_path=${ERDOS_MODULE} \
    data.sampler.class_name=ErdosPUCTSampler \
    data.dataloader_num_workers=0 \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.train_epoch_steps=32 \
    data.train_initial_state_count=64 \
    data.val_initial_state_count=16 \
    data.validation_puct_steps=4 \
    data.archive_max_size=256 \
    data.puct_c=1.0 \
    data.topk_children=2 \
    data.train_budget_s=10 \
    data.val_budget_s=30 \
    data.num_cpus_per_task=1 \
    data.max_construction_len=1000 \
    data.return_raw_chat=True \
    data.max_prompt_length=2048 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=False \
    data.validation_shuffle=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.reward_manager.name=naive \
    reward.num_workers=8 \
    reward.custom_reward_function.path=${ERDOS_MODULE} \
    reward.custom_reward_function.name=compute_score_erdos \
    +reward.custom_reward_function.reward_kwargs.train_budget_s=10 \
    +reward.custom_reward_function.reward_kwargs.val_budget_s=30 \
    +reward.custom_reward_function.reward_kwargs.num_cpus_per_task=1 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='omega_erdos_puct' \
    trainer.experiment_name='qwen2.5_3b_omega_erdos_puct' \
    "$@"
