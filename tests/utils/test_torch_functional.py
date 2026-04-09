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

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.trainer.ppo.omega_escort import compute_vc_bte_vectorized
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.torch_functional import (
    distributed_masked_mean,
    distributed_mean_max_min_std,
    entropy_from_logits,
    expand_as_nested,
    masked_mean,
    tsallis_entropy_from_logits,
    tsallis_entropy_from_logits_with_chunking,
)


def _worker_mean(rank: int, world_size: int, rendezvous_file: str):
    # 1) set GPU and init NCCL
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    # each rank holds tensor [rank+1]
    local = torch.tensor([float(rank + 1)], device=f"{get_device_name()}:{rank}")
    mean, gmax, gmin, gstd = distributed_mean_max_min_std(local, True, True, True)

    values = [float(i + 1) for i in range(world_size)]
    exp_mean = sum(values) / len(values)
    exp_max = max(values)
    exp_min = min(values)
    var = sum((x - exp_mean) ** 2 for x in values) / (len(values) - 1)
    exp_std = var**0.5

    # all ranks should see the same result
    assert torch.allclose(mean.cpu(), torch.tensor(exp_mean)), f"mean@{rank}"
    assert torch.allclose(gmax.cpu(), torch.tensor(exp_max)), f"max@{rank}"
    assert torch.allclose(gmin.cpu(), torch.tensor(exp_min)), f"min@{rank}"
    assert torch.allclose(gstd.cpu(), torch.tensor(exp_std)), f"std@{rank}"

    dist.destroy_process_group()


@pytest.mark.parametrize(
    "value,mask,gt",
    [
        ([1.0, 2.0, 3.0, 4.0], [1, 0, 0, 1], 2.5),
        ([1.0, 2.0, float("nan"), 4.0], [1, 0, 0, 1], 2.5),
        ([1.0, 2.0, float("nan"), 4.0], [1, 0, 1, 0], float("nan")),
    ],
)
def test_masked_mean(value, mask, gt):
    res = masked_mean(torch.tensor(value), torch.tensor(mask))
    gt = torch.tensor(gt)
    assert torch.allclose(res, gt) or (torch.isnan(res) and torch.isnan(gt))


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_mean_max_min_std(world_size, tmp_path):
    rendezvous_file = str(tmp_path / "rdzv_mean")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    mp.spawn(
        fn=_worker_mean,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


def _worker_mask(rank: int, world_size: int, rendezvous_file: str):
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    # build per‐rank tensor and mask
    local_tensor = torch.tensor([rank * 2 + 1.0, rank * 2 + 2.0], device=f"{get_device_name()}:{rank}")
    if rank == 0:
        mask = torch.tensor([1, 0], device=f"{get_device_name()}:{rank}", dtype=torch.float32)
    else:
        mask = torch.tensor([0, 1], device=f"{get_device_name()}:{rank}", dtype=torch.float32)

    gmean = distributed_masked_mean(local_tensor, mask)

    valid_values = [1.0] + [2 * i + 2.0 for i in range(1, world_size)]
    expected_mean = sum(valid_values) / len(valid_values)
    assert torch.allclose(gmean.cpu(), torch.tensor(expected_mean)), f"masked_mean@{rank}"

    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_masked_mean(world_size, tmp_path):
    rendezvous_file = str(tmp_path / "rdzv_mask")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    mp.spawn(
        fn=_worker_mask,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


def test_expand_as_nested():
    a = torch.randn(2)
    b = torch.randn(3)
    c = torch.randn(4)
    nested_tensor = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
    tensor = torch.tensor([1, 2, 3])

    output = expand_as_nested(tensor, nested_tensor)

    assert output.values().tolist() == [1, 1, 2, 2, 2, 3, 3, 3, 3]
    assert torch.all(output.offsets() == nested_tensor.offsets()).item()

    # test exceptions
    with pytest.raises(AssertionError):
        expand_as_nested(tensor, tensor)

    other_tensor = torch.tensor([1, 2, 3, 4])

    with pytest.raises(AssertionError):
        expand_as_nested(other_tensor, nested_tensor)

    other_tensor = torch.tensor([[1, 2, 3]])

    with pytest.raises(AssertionError):
        expand_as_nested(other_tensor, nested_tensor)

    with pytest.raises(AssertionError):
        expand_as_nested(tensor, nested_tensor.unsqueeze(-1))


def test_tsallis_entropy_from_logits_matches_closed_form():
    logits = torch.log(torch.tensor([[3.0, 1.0], [1.0, 1.0]], dtype=torch.float32))

    entropy = tsallis_entropy_from_logits(logits, q=2.0)
    expected = torch.tensor([0.375, 0.5], dtype=torch.float32)

    assert torch.allclose(entropy, expected, atol=1e-6)


def test_tsallis_entropy_from_logits_with_chunking_matches_non_chunked():
    torch.manual_seed(0)
    logits = torch.randn(17, 13, dtype=torch.float32)

    entropy = tsallis_entropy_from_logits(logits, q=2.0)
    entropy_chunked = tsallis_entropy_from_logits_with_chunking(logits, q=2.0, chunk_size=4)

    assert torch.allclose(entropy_chunked, entropy, atol=1e-6)


def test_tsallis_entropy_from_logits_falls_back_to_shannon_near_q_one():
    torch.manual_seed(0)
    logits = torch.randn(5, 7, dtype=torch.float32)

    shannon_entropy = entropy_from_logits(logits)
    near_limit_entropy = tsallis_entropy_from_logits(logits, q=1.0 + 1e-6)
    exact_limit_entropy = tsallis_entropy_from_logits(logits, q=1.0)

    assert torch.allclose(near_limit_entropy, shannon_entropy, atol=1e-6)
    assert torch.allclose(exact_limit_entropy, shannon_entropy, atol=1e-6)


def test_vc_bte_positive_alpha_upweights_rarer_blocks():
    logprobs = torch.tensor([[-3.0, -3.0], [-0.2, -0.2]], dtype=torch.float32)
    entropies = torch.tensor([[1.0, 1.0], [0.2, 0.2]], dtype=torch.float32)
    variances = torch.zeros_like(logprobs)
    mask = torch.ones_like(logprobs, dtype=torch.bool)

    omega = compute_vc_bte_vectorized(
        logprobs=logprobs,
        entropies=entropies,
        variances=variances,
        mask=mask,
        alpha=1.0,
        block_size=2,
        log_omega_clip=10.0,
    )

    assert omega["raw_log_omega"][0, 0] > omega["raw_log_omega"][1, 0]
    assert torch.all(omega["omega_t_renorm"][0] > 1.0)
    assert torch.all(omega["omega_t_renorm"][1] < 1.0)


def test_vc_bte_batch_renorm_has_mean_one_per_timestep():
    torch.manual_seed(0)
    logprobs = torch.randn(3, 5, dtype=torch.float32)
    entropies = torch.rand(3, 5, dtype=torch.float32)
    variances = torch.rand(3, 5, dtype=torch.float32)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    omega = compute_vc_bte_vectorized(
        logprobs=logprobs,
        entropies=entropies,
        variances=variances,
        mask=mask,
        alpha=0.7,
        block_size=2,
        log_omega_clip=10.0,
    )

    mask_float = mask.float()
    per_timestep_mean = (omega["omega_t_renorm"] * mask_float).sum(dim=0) / mask_float.sum(dim=0).clamp(min=1.0)
    valid_timesteps = mask.any(dim=0)
    assert torch.allclose(per_timestep_mean[valid_timesteps], torch.ones_like(per_timestep_mean[valid_timesteps]))


def test_vc_bte_clips_large_log_weights():
    logprobs = torch.tensor([[-10.0, -10.0], [-0.1, -0.1]], dtype=torch.float32)
    entropies = torch.zeros_like(logprobs)
    variances = torch.zeros_like(logprobs)
    mask = torch.ones_like(logprobs, dtype=torch.bool)

    omega = compute_vc_bte_vectorized(
        logprobs=logprobs,
        entropies=entropies,
        variances=variances,
        mask=mask,
        alpha=1.0,
        block_size=2,
        log_omega_clip=0.5,
    )

    assert omega["raw_log_omega"][0, 0] > 0.5
    assert torch.all(omega["clipped_log_omega"] <= 0.5)
    assert torch.all(omega["clipped_log_omega"] >= -0.5)
