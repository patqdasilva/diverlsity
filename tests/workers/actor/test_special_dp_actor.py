# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import unittest

import torch
import torch.nn as nn
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, Qwen3Config

from verl import DataProto
from verl.utils.device import get_device_name
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import FSDPActorConfig, OptimizerConfig


class MockTransformerModel(nn.Module):
    """Mock transformer model for testing DataParallelPPOActor"""

    def __init__(self, vocab_size=1000, hidden_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True), num_layers=2
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, **kwargs):
        batch_size, seq_len = input_ids.shape

        embeddings = self.embedding(input_ids)
        hidden_states = self.transformer(embeddings)
        logits = self.lm_head(hidden_states)

        class MockOutput:
            def __init__(self, logits):
                self.logits = logits

        return MockOutput(logits)


class TestDataParallelPPOActor(unittest.TestCase):
    """Test DataParallelPPOActor compute_log_prob and update_policy methods"""

    @classmethod
    def setUpClass(cls):
        """Set up distributed environment"""
        if get_device_name() == "cuda":
            backend_name = "nccl"
        elif get_device_name() == "npu":
            backend_name = "hccl"
        else:
            backend_name = "gloo"

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend_name, init_method="env://")

        cls.rank = torch.distributed.get_rank()
        cls.world_size = torch.distributed.get_world_size()

        if get_device_name() == "cuda":
            torch.cuda.set_device(cls.rank)
            cls.device = torch.device(f"cuda:{cls.rank}")
        elif get_device_name() == "npu":
            torch.npu.set_device(cls.rank)
            cls.device = torch.device(f"npu:{cls.rank}")
        else:
            cls.device = torch.device("cpu")

    def setUp(self):
        """Set up test fixtures"""
        self.config = FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            clip_ratio=0.2,
            entropy_coeff=0.01,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            use_torch_compile=False,  # Disable torch.compile for testing
            ulysses_sequence_parallel_size=1,
            optim=OptimizerConfig(lr=1e-6),
            rollout_n=1,
        )

        self.mock_model = MockTransformerModel(vocab_size=1000, hidden_size=64).to(self.device)
        self.mock_optimizer = torch.optim.Adam(self.mock_model.parameters(), lr=1e-4)

        self.actor = DataParallelPPOActor(
            config=self.config, actor_module=self.mock_model, actor_optimizer=self.mock_optimizer
        )

    def _build_actor(self, *, entropy_coeff=None, entropy_type=None, tsallis_q=None):
        config = FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            clip_ratio=0.2,
            entropy_coeff=self.config.entropy_coeff if entropy_coeff is None else entropy_coeff,
            entropy_type=self.config.entropy_type if entropy_type is None else entropy_type,
            tsallis_q=self.config.tsallis_q if tsallis_q is None else tsallis_q,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            optim=OptimizerConfig(lr=1e-6),
            rollout_n=1,
        )
        model = MockTransformerModel(vocab_size=1000, hidden_size=64).to(self.device)
        model.load_state_dict(self.mock_model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        return DataParallelPPOActor(config=config, actor_module=model, actor_optimizer=optimizer)

    def _build_ref_policy(self, *, entropy_coeff=None, entropy_type=None, tsallis_q=None, use_fused_kernels=False):
        config = FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            clip_ratio=0.2,
            entropy_coeff=self.config.entropy_coeff if entropy_coeff is None else entropy_coeff,
            entropy_type=self.config.entropy_type if entropy_type is None else entropy_type,
            tsallis_q=self.config.tsallis_q if tsallis_q is None else tsallis_q,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            use_fused_kernels=use_fused_kernels,
            optim=OptimizerConfig(lr=1e-6),
            rollout_n=1,
        )
        model = MockTransformerModel(vocab_size=1000, hidden_size=64).to(self.device)
        model.load_state_dict(self.mock_model.state_dict())
        return DataParallelPPOActor(config=config, actor_module=model, actor_optimizer=None)

    @classmethod
    def tearDownClass(cls):
        """Clean up distributed environment"""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def _create_test_data_for_compute_log_prob(self):
        """Create test DataProto for compute_log_prob method"""
        batch_size = 2
        prompt_length = 8
        response_length = 4
        total_length = prompt_length + response_length
        vocab_size = 1000

        input_ids = torch.randint(0, vocab_size, (batch_size, total_length)).to(self.device)
        attention_mask = torch.ones(batch_size, total_length).to(self.device)
        position_ids = torch.arange(total_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
        responses = input_ids[:, -response_length:]  # Last part is the response

        tensor_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
            },
            batch_size=[batch_size],
        )

        meta_info = {"micro_batch_size": batch_size, "temperature": 1.0, "use_dynamic_bsz": False}

        return DataProto(batch=tensor_dict, meta_info=meta_info)

    def _create_test_data_for_update_policy(self, *, omega_log_weights=None, advantages=None, old_log_probs=None):
        """Create test DataProto for update_policy method"""
        batch_size = 4  # Must match ppo_mini_batch_size
        prompt_length = 8
        response_length = 4
        total_length = prompt_length + response_length
        vocab_size = 1000

        input_ids = torch.randint(0, vocab_size, (batch_size, total_length)).to(self.device)
        attention_mask = torch.ones(batch_size, total_length).to(self.device)
        position_ids = torch.arange(total_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
        responses = input_ids[:, -response_length:]
        response_mask = torch.ones(batch_size, response_length).to(self.device)
        if old_log_probs is None:
            old_log_probs = torch.randn(batch_size, response_length).to(self.device) * 0.1  # Small values
        if advantages is None:
            advantages = torch.randn(batch_size, response_length).to(self.device) * 0.5

        tensor_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
        }
        if omega_log_weights is not None:
            tensor_data["omega_log_weights"] = omega_log_weights

        tensor_dict = TensorDict(tensor_data, batch_size=[batch_size])

        meta_info = {"temperature": 1.0}

        return DataProto(batch=tensor_dict, meta_info=meta_info)

    def test_compute_log_prob(self):
        """Test compute_log_prob method"""
        data = self._create_test_data_for_compute_log_prob()

        outputs = self.actor.compute_log_prob(data, calculate_entropy=True)
        log_probs = outputs["log_probs"]
        entropys = outputs["entropys"]

        batch_size = data.batch["responses"].shape[0]
        response_length = data.batch["responses"].shape[1]

        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(log_probs)))

        self.assertIsInstance(entropys, torch.Tensor)
        self.assertEqual(entropys.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(entropys)))
        self.assertTrue(torch.all(entropys >= 0))  # Entropy should be non-negative

    def test_compute_log_prob_without_entropy(self):
        """Test compute_log_prob method without entropy calculation"""
        data = self._create_test_data_for_compute_log_prob()

        outputs = self.actor.compute_log_prob(data, calculate_entropy=False)
        log_probs = outputs["log_probs"]
        entropys = outputs.get("entropys", None)

        batch_size = data.batch["responses"].shape[0]
        response_length = data.batch["responses"].shape[1]

        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(log_probs)))
        self.assertIsNone(entropys)

    def test_compute_log_prob_with_tsallis_entropy(self):
        """Test Tsallis entropy differs from Shannon while staying finite and non-negative."""
        tsallis_config = FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            clip_ratio=0.2,
            entropy_coeff=0.01,
            entropy_type="tsallis",
            tsallis_q=2.0,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            optim=OptimizerConfig(lr=1e-6),
            rollout_n=1,
        )
        tsallis_model = MockTransformerModel(vocab_size=1000, hidden_size=64).to(self.device)
        tsallis_model.load_state_dict(self.mock_model.state_dict())
        tsallis_optimizer = torch.optim.Adam(tsallis_model.parameters(), lr=1e-4)
        tsallis_actor = DataParallelPPOActor(
            config=tsallis_config, actor_module=tsallis_model, actor_optimizer=tsallis_optimizer
        )

        data = self._create_test_data_for_compute_log_prob()

        shannon_outputs = self.actor.compute_log_prob(data, calculate_entropy=True)
        tsallis_outputs = tsallis_actor.compute_log_prob(data, calculate_entropy=True)

        shannon_entropy = shannon_outputs["entropys"]
        tsallis_entropy = tsallis_outputs["entropys"]

        self.assertEqual(tsallis_entropy.shape, shannon_entropy.shape)
        self.assertTrue(torch.all(torch.isfinite(tsallis_entropy)))
        self.assertTrue(torch.all(tsallis_entropy >= 0))
        self.assertTrue(torch.allclose(tsallis_outputs["log_probs"], shannon_outputs["log_probs"]))
        self.assertFalse(torch.allclose(tsallis_entropy, shannon_entropy))

    def test_ref_initialization_ignores_mirrored_tsallis_entropy(self):
        """Ref policies should stay forward-only even when config mirrors Tsallis settings."""
        ref_policy = self._build_ref_policy(
            entropy_coeff=2.0,
            entropy_type="tsallis",
            tsallis_q=2.0,
            use_fused_kernels=True,
        )

        self.assertEqual(ref_policy.entropy_type, "shannon")
        self.assertEqual(ref_policy.tsallis_q, 2.0)

        data = self._create_test_data_for_compute_log_prob()
        outputs = ref_policy.compute_log_prob(data, calculate_entropy=False)

        self.assertIn("log_probs", outputs)
        self.assertNotIn("entropys", outputs)

    def test_update_policy(self):
        """Test update_policy method"""
        data = self._create_test_data_for_update_policy()

        metrics = self.actor.update_policy(data)

        self.assertIsInstance(metrics, dict)

        expected_metric_keys = [
            "actor/pg_loss",
            "actor/pg_clipfrac",
            "actor/ppo_kl",
            "actor/pg_clipfrac_lower",
            "actor/grad_norm",
        ]

        for key in expected_metric_keys:
            self.assertIn(key, metrics)

    def test_update_policy_consumes_omega_log_weights(self):
        baseline_actor = self._build_actor(entropy_coeff=0.0)
        omega_actor = self._build_actor(entropy_coeff=0.0)

        advantages = torch.ones(4, 4, device=self.device)
        old_log_probs = torch.zeros(4, 4, device=self.device)
        baseline_data = self._create_test_data_for_update_policy(advantages=advantages, old_log_probs=old_log_probs)
        omega_data = self._create_test_data_for_update_policy(
            advantages=advantages,
            old_log_probs=old_log_probs,
            omega_log_weights=torch.full((4, 4), 0.35, device=self.device),
        )

        baseline_metrics = baseline_actor.update_policy(baseline_data)
        omega_metrics = omega_actor.update_policy(omega_data)

        self.assertNotEqual(baseline_metrics["actor/pg_loss"], omega_metrics["actor/pg_loss"])

    def test_update_policy_combined_omega_and_tsallis(self):
        combined_actor = self._build_actor(entropy_coeff=0.01, entropy_type="tsallis", tsallis_q=2.0)
        advantages = torch.ones(4, 4, device=self.device)
        old_log_probs = torch.zeros(4, 4, device=self.device)
        combined_data = self._create_test_data_for_update_policy(
            advantages=advantages,
            old_log_probs=old_log_probs,
            omega_log_weights=torch.full((4, 4), 0.25, device=self.device),
        )

        metrics = combined_actor.update_policy(combined_data)

        self.assertIn("actor/entropy", metrics)
        self.assertIn("actor/tsallis_q", metrics)
        self.assertEqual(metrics["actor/tsallis_q"][0], 2.0)

    def test_dataparallelppoactor_initialization(self):
        """Test DataParallelPPOActor initialization"""
        self.assertIsNotNone(self.actor.actor_module)
        self.assertIsNotNone(self.actor.actor_optimizer)
        self.assertEqual(self.actor.config, self.config)

        self.assertEqual(self.actor.config.strategy, "fsdp2")
        self.assertEqual(self.actor.config.ppo_mini_batch_size, 4)
        self.assertEqual(self.actor.config.clip_ratio, 0.2)

    def test_dataparallelppoactor_with_qwen3_model(self):
        """Test DataParallelPPOActor with real Qwen3ForCausalLM model"""
        qwen_config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            torch_dtype=torch.float32,
            use_cache=False,
        )

        with torch.device(self.device):
            qwen_model = AutoModelForCausalLM.from_config(config=qwen_config, torch_dtype=torch.float32).to(self.device)

        qwen_optimizer = torch.optim.Adam(qwen_model.parameters(), lr=1e-4)

        qwen_actor = DataParallelPPOActor(config=self.config, actor_module=qwen_model, actor_optimizer=qwen_optimizer)

        data = self._create_test_data_for_compute_log_prob()
        outputs = qwen_actor.compute_log_prob(data, calculate_entropy=True)
        log_probs = outputs["log_probs"]
        entropys = outputs["entropys"]

        batch_size = data.batch["responses"].shape[0]
        response_length = data.batch["responses"].shape[1]

        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(log_probs)))

        self.assertIsInstance(entropys, torch.Tensor)
        self.assertEqual(entropys.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(entropys)))
        self.assertTrue(torch.all(entropys >= 0))

        policy_data = self._create_test_data_for_update_policy()
        metrics = qwen_actor.update_policy(policy_data)

        self.assertIsInstance(metrics, dict)

        expected_metric_keys = [
            "actor/pg_loss",
            "actor/pg_clipfrac",
            "actor/ppo_kl",
            "actor/pg_clipfrac_lower",
            "actor/grad_norm",
        ]

        for key in expected_metric_keys:
            self.assertIn(key, metrics)
            if isinstance(metrics[key], list):
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics[key]))
            else:
                self.assertIsInstance(metrics[key], (float, int))
                self.assertTrue(torch.isfinite(torch.tensor(metrics[key])))


if __name__ == "__main__":
    unittest.main()
