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

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import validate_geometry_feature_config


def _make_config(*, actor_strategy="fsdp", rollout_name="vllm", entropy_type="shannon", entropy_coeff=0.0, use_fused_kernels=False, omega_alpha=0.0):
    return OmegaConf.create(
        {
            "algorithm": {"omega_escort_alpha": omega_alpha},
            "actor_rollout_ref": {
                "model": {"use_fused_kernels": use_fused_kernels},
                "actor": {
                    "strategy": actor_strategy,
                    "entropy_type": entropy_type,
                    "entropy_coeff": entropy_coeff,
                    "use_fused_kernels": use_fused_kernels,
                },
                "rollout": {"name": rollout_name},
            },
        }
    )


def test_geometry_validation_allows_fsdp_vllm_modes():
    validate_geometry_feature_config(_make_config(omega_alpha=0.5))
    validate_geometry_feature_config(_make_config(entropy_type="tsallis", entropy_coeff=0.1))
    validate_geometry_feature_config(_make_config(entropy_type="tsallis", entropy_coeff=0.1, omega_alpha=0.5))


def test_geometry_validation_rejects_non_vllm_omega():
    config = _make_config(rollout_name="sglang", omega_alpha=0.5)
    with pytest.raises(ValueError, match="requires actor_rollout_ref.rollout.name=vllm"):
        validate_geometry_feature_config(config)


def test_geometry_validation_rejects_non_fsdp_tsallis():
    config = _make_config(actor_strategy="megatron", entropy_type="tsallis", entropy_coeff=0.1)
    with pytest.raises(ValueError, match="only implemented for FSDP/FSDP2"):
        validate_geometry_feature_config(config)


def test_geometry_validation_rejects_tsallis_with_fused_kernels():
    config = _make_config(entropy_type="tsallis", entropy_coeff=0.1, use_fused_kernels=True)
    with pytest.raises(ValueError, match="not supported when use_fused_kernels=True"):
        validate_geometry_feature_config(config)
