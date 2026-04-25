import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import TSALLIS_STOCHASTIC_Q2
from verl.utils import tensordict_utils as tu
from verl.workers.config import FSDPActorConfig, OptimizerConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss


def _make_tsallis_actor_config():
    return FSDPActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=2,
        ppo_epochs=1,
        use_dynamic_bsz=False,
        use_torch_compile=False,
        optim=OptimizerConfig(lr=1e-4),
        policy_loss=PolicyLossConfig(
            loss_mode=TSALLIS_STOCHASTIC_Q2,
            tsallis_alpha=0.5,
            tsallis_chunk_rows=1,
            tsallis_prob_floor=1e-8,
        ),
    )


def _make_no_padding_batch():
    prompts = torch.nested.as_nested_tensor(
        [torch.tensor([10]), torch.tensor([20, 21])],
        layout=torch.jagged,
    )
    responses = torch.nested.as_nested_tensor(
        [torch.tensor([0, 1]), torch.tensor([2, 0])],
        layout=torch.jagged,
    )
    log_probs = torch.nested.as_nested_tensor(
        [torch.tensor([0.0, -0.2, -0.1]), torch.tensor([0.0, 0.0, -0.4, -0.3])],
        layout=torch.jagged,
    )
    response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
    data = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "old_log_probs": torch.zeros(2, 2),
            "advantages": torch.tensor([[1.0, -0.5], [0.25, 0.0]], dtype=torch.float32),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(response_mask.sum().item()), global_batch_size=2)
    return data, log_probs


def test_ppo_loss_passes_flattened_logits_and_action_ids_to_tsallis():
    config = _make_tsallis_actor_config()
    data, log_probs = _make_no_padding_batch()
    response_logits = torch.tensor(
        [
            [0.3, -0.1, 0.0],
            [0.2, 0.4, -0.2],
            [-0.5, 0.1, 0.6],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    model_output = {
        "log_probs": log_probs,
        "response_logits": response_logits,
        "action_ids": torch.tensor([0, 1, 2], dtype=torch.long),
    }

    loss, metrics = ppo_loss(config=config, model_output=model_output, data=data)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(response_logits.grad))
    assert "actor/pg_loss" in metrics
    assert "actor/ppo_kl" in metrics
