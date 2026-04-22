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

import random
import unittest

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    TSALLIS_STOCHASTIC_Q2,
    _build_tsallis_q2_target,
    _simplex_project_q2,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_policy_loss_tsallis_stochastic_q2,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    register_adv_est,
)
from verl.workers.config import ActorConfig, FSDPActorConfig, OptimizerConfig, PolicyLossConfig


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()


def _make_tsallis_fsdp_config(**overrides):
    policy_overrides = overrides.pop("policy_loss_overrides", {})
    policy_loss = PolicyLossConfig(loss_mode=TSALLIS_STOCHASTIC_Q2, **policy_overrides)
    return FSDPActorConfig(
        strategy="fsdp2",
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=2,
        ppo_epochs=1,
        use_dynamic_bsz=False,
        use_torch_compile=False,
        optim=OptimizerConfig(lr=1e-4),
        policy_loss=policy_loss,
        **overrides,
    )


def _make_tsallis_loss_inputs():
    response_logits = torch.tensor(
        [
            [[0.2, -0.4, 0.1], [0.1, 0.3, -0.2]],
            [[-0.5, 0.7, 0.2], [0.4, -0.1, 0.0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    action_ids = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
    log_prob = torch.log_softmax(response_logits, dim=-1).gather(-1, action_ids.unsqueeze(-1)).squeeze(-1)
    advantages = torch.tensor([[1.0, -0.75], [0.5, 0.0]], dtype=torch.float32)
    old_log_prob = log_prob.detach()
    return old_log_prob, log_prob, advantages, response_mask, response_logits, action_ids


def test_simplex_project_q2_returns_valid_simplex_points():
    values = torch.tensor([[1.5, -0.2, 0.4], [0.2, 0.3, 0.5]], dtype=torch.float32)
    projected = _simplex_project_q2(values)

    assert torch.all(projected >= 0)
    assert torch.allclose(projected.sum(dim=-1), torch.ones(projected.shape[0]), atol=1e-6)
    assert torch.allclose(projected[1], values[1], atol=1e-6)


def test_tsallis_q2_target_moves_mass_with_advantage_sign():
    logits = torch.zeros(2, 3, dtype=torch.float32)
    action_ids = torch.tensor([0, 1], dtype=torch.long)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)

    targets = _build_tsallis_q2_target(
        logits_chunk=logits,
        action_ids_chunk=action_ids,
        advantages_chunk=advantages,
        alpha=0.5,
        prob_floor=1e-8,
    )

    base_prob = 1 / 3
    assert torch.allclose(targets.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert targets[0, 0] > base_prob
    assert targets[1, 1] < base_prob


def test_tsallis_policy_loss_is_chunk_invariant():
    old_log_prob, log_prob, advantages, response_mask, response_logits, action_ids = _make_tsallis_loss_inputs()
    config_small = _make_tsallis_fsdp_config(policy_loss_overrides={"tsallis_chunk_rows": 1})
    config_large = _make_tsallis_fsdp_config(policy_loss_overrides={"tsallis_chunk_rows": 16})

    loss_small, clip_small, kl_small, clip_lower_small = compute_policy_loss_tsallis_stochastic_q2(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config_small,
        response_logits=response_logits,
        action_ids=action_ids,
    )
    loss_large, clip_large, kl_large, clip_lower_large = compute_policy_loss_tsallis_stochastic_q2(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config_large,
        response_logits=response_logits,
        action_ids=action_ids,
    )

    assert torch.allclose(loss_small, loss_large, atol=1e-6)
    assert torch.allclose(clip_small, clip_large, atol=1e-6)
    assert torch.allclose(kl_small, kl_large, atol=1e-6)
    assert torch.allclose(clip_lower_small, clip_lower_large, atol=1e-6)


def test_tsallis_policy_loss_stays_finite_near_probability_floor():
    response_logits = torch.tensor([[[-40.0, 0.0, 0.0]]], dtype=torch.float32, requires_grad=True)
    action_ids = torch.tensor([[0]], dtype=torch.long)
    response_mask = torch.tensor([[1.0]], dtype=torch.float32)
    advantages = torch.tensor([[1.25]], dtype=torch.float32)
    log_prob = torch.log_softmax(response_logits, dim=-1).gather(-1, action_ids.unsqueeze(-1)).squeeze(-1)
    config = _make_tsallis_fsdp_config(policy_loss_overrides={"tsallis_prob_floor": 1e-6})

    loss, *_ = compute_policy_loss_tsallis_stochastic_q2(
        old_log_prob=log_prob.detach(),
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        response_logits=response_logits,
        action_ids=action_ids,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(response_logits.grad))


def test_tsallis_policy_loss_requires_logits_and_action_ids():
    old_log_prob, log_prob, advantages, response_mask, _, _ = _make_tsallis_loss_inputs()
    config = _make_tsallis_fsdp_config()

    with pytest.raises(ValueError, match="requires both response_logits and action_ids"):
        compute_policy_loss_tsallis_stochastic_q2(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=config,
        )


def test_tsallis_policy_loss_rejects_unsupported_modes():
    old_log_prob, log_prob, advantages, response_mask, response_logits, action_ids = _make_tsallis_loss_inputs()

    megatron_config = ActorConfig(
        strategy="megatron",
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=2,
        ppo_epochs=1,
        use_dynamic_bsz=False,
        optim=OptimizerConfig(lr=1e-4),
        policy_loss=PolicyLossConfig(loss_mode=TSALLIS_STOCHASTIC_Q2),
    )
    with pytest.raises(ValueError, match="only supported for FSDP actors"):
        compute_policy_loss_tsallis_stochastic_q2(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=megatron_config,
            response_logits=response_logits,
            action_ids=action_ids,
        )

    fused_config = _make_tsallis_fsdp_config(use_fused_kernels=True)
    with pytest.raises(ValueError, match="use_fused_kernels=True"):
        compute_policy_loss_tsallis_stochastic_q2(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=fused_config,
            response_logits=response_logits,
            action_ids=action_ids,
        )

    multi_epoch_config = _make_tsallis_fsdp_config(ppo_epochs=2)
    with pytest.raises(ValueError, match="requires ppo_epochs=1"):
        compute_policy_loss_tsallis_stochastic_q2(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=multi_epoch_config,
            response_logits=response_logits,
            action_ids=action_ids,
        )
