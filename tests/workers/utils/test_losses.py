# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from tensordict import TensorDict

from verl.workers.config import FSDPActorConfig, OptimizerConfig
from verl.workers.utils.losses import ppo_loss


def _make_actor_config(*, entropy_coeff: float = 0.0, entropy_type: str = "shannon", tsallis_q: float = 2.0):
    return FSDPActorConfig(
        strategy="fsdp",
        ppo_mini_batch_size=1,
        ppo_micro_batch_size_per_gpu=1,
        ppo_epochs=1,
        use_dynamic_bsz=False,
        entropy_coeff=entropy_coeff,
        entropy_type=entropy_type,
        tsallis_q=tsallis_q,
        rollout_n=1,
        optim=OptimizerConfig(lr=1e-6),
    )


def _make_loss_inputs(*, omega_log_weights: torch.Tensor | None = None):
    data = TensorDict(
        {
            "prompts": torch.tensor([[11]], dtype=torch.long),
            "responses": torch.tensor([[21, 22]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "old_log_probs": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            "advantages": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            "ref_log_prob": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            "dp_size": torch.tensor([1]),
            "batch_num_tokens": torch.tensor([2.0]),
            "global_batch_size": torch.tensor([1]),
        },
        batch_size=[1],
    )
    if omega_log_weights is not None:
        data["omega_log_weights"] = omega_log_weights

    model_output = {
        "log_probs": torch.tensor([0.1, 0.2, 0.0], dtype=torch.float32),
        "entropy": torch.tensor([0.6, 0.4, 0.0], dtype=torch.float32),
    }
    return model_output, data


def test_ppo_loss_omega_only_changes_policy_loss():
    config = _make_actor_config(entropy_coeff=0.0)
    model_output, data = _make_loss_inputs()
    omega_model_output, omega_data = _make_loss_inputs(omega_log_weights=torch.tensor([[0.4, -0.2]], dtype=torch.float32))

    base_loss, _ = ppo_loss(config=config, model_output=model_output, data=data)
    omega_loss, _ = ppo_loss(config=config, model_output=omega_model_output, data=omega_data)

    assert not torch.allclose(base_loss, omega_loss)


def test_ppo_loss_tsallis_only_reports_q_and_entropy_term():
    config = _make_actor_config(entropy_coeff=0.5, entropy_type="tsallis", tsallis_q=2.0)
    model_output, data = _make_loss_inputs()

    loss, metrics = ppo_loss(config=config, model_output=model_output, data=data)

    assert torch.isfinite(loss)
    assert "actor/entropy_loss" in metrics
    assert metrics["actor/tsallis_q"] == 2.0


def test_ppo_loss_combined_mode_applies_omega_and_tsallis():
    config = _make_actor_config(entropy_coeff=0.5, entropy_type="tsallis", tsallis_q=2.0)
    model_output, data = _make_loss_inputs()
    omega_model_output, omega_data = _make_loss_inputs(omega_log_weights=torch.tensor([[0.4, -0.2]], dtype=torch.float32))

    tsallis_loss, tsallis_metrics = ppo_loss(config=config, model_output=model_output, data=data)
    combined_loss, combined_metrics = ppo_loss(config=config, model_output=omega_model_output, data=omega_data)

    assert not torch.allclose(tsallis_loss, combined_loss)
    assert combined_metrics["actor/tsallis_q"] == 2.0
    assert "actor/entropy_loss" in tsallis_metrics
