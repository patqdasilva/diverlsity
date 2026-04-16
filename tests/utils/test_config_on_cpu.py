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

import unittest
from dataclasses import dataclass, field

from omegaconf import OmegaConf

from verl.base_config import BaseConfig
from verl.utils import omega_conf_to_dataclass


@dataclass
class TestDataclass(BaseConfig):
    hidden_size: int = 0
    activation: str = "relu"


@dataclass
class TestTrainConfig(BaseConfig):
    batch_size: int = 0
    model: TestDataclass = field(default_factory=TestDataclass)
    override_config: dict = field(default_factory=dict)


_cfg_str = """train_config:
  _target_: tests.utils.test_config_on_cpu.TestTrainConfig
  batch_size: 32
  model:
    hidden_size: 768
    activation: relu
  override_config: {}"""


class TestConfigOnCPU(unittest.TestCase):
    """Test cases for configuration utilities on CPU.

    Test Plan:
    1. Test basic OmegaConf to dataclass conversion for simple nested structures
    2. Test nested OmegaConf to dataclass conversion for complex hierarchical configurations
    3. Verify all configuration values are correctly converted and accessible
    """

    def setUp(self):
        self.config = OmegaConf.create(_cfg_str)

    def test_omega_conf_to_dataclass(self):
        sub_cfg = self.config.train_config.model
        cfg = omega_conf_to_dataclass(sub_cfg, TestDataclass)
        self.assertEqual(cfg.hidden_size, 768)
        self.assertEqual(cfg.activation, "relu")
        assert isinstance(cfg, TestDataclass)

    def test_nested_omega_conf_to_dataclass(self):
        cfg = omega_conf_to_dataclass(self.config.train_config, TestTrainConfig)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.model.hidden_size, 768)
        self.assertEqual(cfg.model.activation, "relu")
        assert isinstance(cfg, TestTrainConfig)
        assert isinstance(cfg.model, TestDataclass)


class TestPrintCfgCommand(unittest.TestCase):
    """Test suite for the print_cfg.py command-line tool."""

    def test_command_with_override(self):
        """Hydra config should still render cleanly with the default trainer config."""
        import os

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config"), version_base=None):
            cfg = compose(config_name="ppo_trainer")

        rendered = OmegaConf.to_yaml(cfg, resolve=True)

        self.assertIn("critic:", rendered)
        self.assertIn("profiler:", rendered)

    def test_command_mirrors_actor_entropy_fields_into_ref(self):
        """Actor entropy overrides should appear on the ref config surface as well."""
        import os

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config"), version_base=None):
            cfg = compose(
                config_name="ppo_trainer",
                overrides=[
                    "actor_rollout_ref.actor.entropy_type=tsallis",
                    "actor_rollout_ref.actor.entropy_coeff=2",
                ],
            )

        self.assertEqual(cfg.actor_rollout_ref.ref.entropy_coeff, cfg.actor_rollout_ref.actor.entropy_coeff)
        self.assertEqual(cfg.actor_rollout_ref.ref.entropy_type, cfg.actor_rollout_ref.actor.entropy_type)
        self.assertEqual(cfg.actor_rollout_ref.ref.tsallis_q, cfg.actor_rollout_ref.actor.tsallis_q)

        rendered = OmegaConf.to_yaml(cfg, resolve=True)
        self.assertIn("entropy_coeff: 2", rendered)
        self.assertIn("entropy_type: tsallis", rendered)
        self.assertIn("tsallis_q: 2.0", rendered)


if __name__ == "__main__":
    unittest.main()
