from __future__ import annotations

import ast
import copy
import os
import pickle
import re
import signal
import subprocess
import sys
import tempfile
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch.utils.data import Sampler

try:
    from verl.experimental.dataset.sampler import AbstractCurriculumSampler
except ModuleNotFoundError:
    class AbstractCurriculumSampler(Sampler[int]):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def update(self, batch) -> None:
            raise NotImplementedError

if TYPE_CHECKING:
    from verl import DataProto

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
DEFAULT_TARGET_C5 = 0.38080
DEFAULT_CURRENT_RECORD_C5 = 0.38092
DEFAULT_MAX_CONSTRUCTION_LEN = 1000
STDOUT_TRUNCATION_CHARS = 500


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Cannot convert h_values to numpy array: {exc}") from exc

    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")

    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")

    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    n = n_points
    target_sum = n / 2.0
    current_sum = float(np.sum(h_values))
    if current_sum == 0:
        raise ValueError("h_values sum to zero")

    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    dx = 2.0 / n_points
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = float(np.max(correlation))

    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")

    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")

    return computed_c5


def evaluate_erdos_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> float:
    verify_c5_solution(h_values, c5_bound, n_points)
    return float(c5_bound)


def verify_erdos_solution(result: tuple[np.ndarray, float, int]) -> bool:
    try:
        h_values, c5_bound, n_points = result
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
            return False
    except Exception:
        return False
    return True


def _infer_split(data_files: str | list[str]) -> str:
    if isinstance(data_files, list):
        text = " ".join(str(item) for item in data_files)
    else:
        text = str(data_files)
    lowered = text.lower()
    if "val" in lowered or "test" in lowered or "eval" in lowered:
        return VAL_SPLIT
    return TRAIN_SPLIT


def _prompt_budget_s(config: DictConfig, split: str) -> float:
    if split == VAL_SPLIT:
        return float(config.get("val_budget_s", 30))
    return float(config.get("train_budget_s", 10))


def _to_float_list(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values]


def _construction_key(construction: list[float] | None) -> tuple[float, ...] | None:
    if not construction:
        return None
    return tuple(round(float(value), 12) for value in construction)


def _as_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


def _state_value_to_raw_score(value: float) -> float:
    return float(-value)


@dataclass
class ErdosStateRecord:
    state_id: str
    timestep: int
    value: float
    raw_score: float
    construction: list[float]
    code: str
    stdout: str = ""
    split: str = TRAIN_SPLIT
    parent_state_id: Optional[str] = None
    parent_ids: list[str] = field(default_factory=list)
    parent_values: list[float] = field(default_factory=list)
    seed: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "timestep": int(self.timestep),
            "value": float(self.value),
            "raw_score": float(self.raw_score),
            "construction": list(self.construction),
            "code": self.code,
            "stdout": self.stdout,
            "split": self.split,
            "parent_state_id": self.parent_state_id,
            "parent_ids": list(self.parent_ids),
            "parent_values": list(self.parent_values),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ErdosStateRecord":
        return cls(
            state_id=str(data["state_id"]),
            timestep=int(data["timestep"]),
            value=float(data["value"]),
            raw_score=float(data.get("raw_score", _state_value_to_raw_score(float(data["value"])))),
            construction=_to_float_list(data.get("construction", [])),
            code=str(data.get("code", "")),
            stdout=str(data.get("stdout", "")),
            split=str(data.get("split", TRAIN_SPLIT)),
            parent_state_id=data.get("parent_state_id"),
            parent_ids=[str(item) for item in data.get("parent_ids", [])],
            parent_values=[float(item) for item in data.get("parent_values", [])],
            seed=data.get("seed"),
        )


def _create_initial_state(seed: int, split: str) -> ErdosStateRecord:
    rng = np.random.default_rng(seed)
    n_points = int(rng.integers(40, 100))
    construction = np.ones(n_points, dtype=np.float64) * 0.5
    perturbation = rng.uniform(-0.4, 0.4, n_points)
    perturbation = perturbation - np.mean(perturbation)
    construction = construction + perturbation
    dx = 2.0 / n_points
    correlation = np.correlate(construction, 1 - construction, mode="full") * dx
    c5_bound = float(np.max(correlation))
    return ErdosStateRecord(
        state_id=f"{split}-seed-{seed}",
        timestep=-1,
        value=-c5_bound,
        raw_score=c5_bound,
        construction=_to_float_list(construction),
        code="",
        stdout="",
        split=split,
        seed=seed,
    )


def _truncate_stdout(stdout: str) -> str:
    stdout = stdout.strip()
    if len(stdout) <= STDOUT_TRUNCATION_CHARS:
        return stdout
    return "\n\n...(TRUNCATED)...\n" + stdout[-STDOUT_TRUNCATION_CHARS:]


def _state_prompt_context(
    state: ErdosStateRecord,
    *,
    target_c5: float,
    metric_name: str = "C5 bound",
) -> str:
    value_ctx = f"You are iteratively optimizing {metric_name}."
    if state.code.strip():
        value_ctx += "\nHere is the last code we ran:\n```python\n" + state.code + "\n```"
    else:
        value_ctx += "\nNo previous code available."

    if state.parent_values and state.construction:
        before_value = _state_value_to_raw_score(state.parent_values[0])
        after_value = state.raw_score
        current_gap = after_value - target_c5
        value_ctx += (
            f"\nHere is the {metric_name} before and after running the code above (lower is better): "
            f"{before_value:.6f} -> {after_value:.6f}"
        )
        value_ctx += (
            f"\nTarget: {target_c5:.5f}. Current gap: {current_gap:.6f}. "
            "Further improvements will also be generously rewarded."
        )
    else:
        after_value = state.raw_score
        current_gap = after_value - target_c5
        value_ctx += f"\nCurrent {metric_name} (lower is better): {after_value:.6f}"
        value_ctx += (
            f"\nTarget: {target_c5:.5f}. Current gap: {current_gap:.6f}. "
            "Further improvements will also be generously rewarded."
        )

    if state.stdout.strip():
        value_ctx += f"\n\n--- Previous Program Output ---\n{_truncate_stdout(state.stdout)}\n--- End Output ---"

    return value_ctx


def build_erdos_prompt(
    state: ErdosStateRecord,
    *,
    budget_s: float,
    target_c5: float = DEFAULT_TARGET_C5,
    current_record_c5: float = DEFAULT_CURRENT_RECORD_C5,
    num_cpus_per_task: int = 1,
) -> str:
    state_ctx = _state_prompt_context(state, target_c5=target_c5)
    construction_section = (
        "\nYou may want to start your search from the current construction, which you can access through "
        f"the `initial_h_values` global variable (n={len(state.construction)} samples).\n"
        "You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.\n"
    )

    if state.code.strip():
        code_section = (
            "Reason about how you could further improve this construction.\n"
            "Ideally, try to do something different than the above algorithm. This could be a different algorithmic idea, "
            "a better heuristic, or a better parameter sweep.\n"
            "Unless you make a meaningful improvement, you will not be rewarded."
        )
    else:
        code_section = "Write code to optimize this construction."

    scaffold = f"""Recommended scaffold:
```python
import numpy as np
import time

def run(seed=42, budget_s={budget_s}, **kwargs):
    start = time.time()
    if "initial_h_values" in globals():
        h = np.asarray(initial_h_values, dtype=np.float64).copy()
    else:
        n_points = int(kwargs.get("n_points", 200))
        h = np.full(n_points, 0.5, dtype=np.float64)
    n_points = int(h.size)
    c5_bound = float(np.max(np.correlate(h, 1.0 - h, mode="full") * (2.0 / n_points)))
    return h, c5_bound, n_points
```
"""

    return f"""You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdos minimum overlap problem constant C5.

## Problem

Find a step function h: [0, 2] -> [0, 1] that minimizes the overlap integral:

C5 = max_k integral h(x)(1 - h(x+k)) dx

Constraints:
1. h(x) must stay in [0, 1]
2. integral_0^2 h(x) dx = 1

Discretization:
- Represent h as n_points samples over [0, 2]
- Let dx = 2.0 / n_points
- 0 <= h[i] <= 1 for all i
- sum(h) * dx = 1, equivalently sum(h) == n_points / 2 exactly

Evaluation:
- The evaluator computes C5 = max(np.correlate(h, 1 - h, mode="full") * dx)
- Lower C5 values are better
- Smaller sequences with fewer than 1k samples are preferred because they are faster to optimize and evaluate

## Budget and Resources
- Time budget: {budget_s}s for your code to run
- CPUs: {num_cpus_per_task} available

## Rules
- Return exactly one fenced `python` code block and nothing else
- Define `run(seed=42, budget_s={budget_s}, **kwargs)` and return `(h_values, c5_bound, n_points)`
- Keep helper functions top level
- Avoid filesystem or network IO
- `evaluate_erdos_solution()` is pre-imported at execution time
- `initial_h_values` is pre-imported at execution time when a prior construction is available
- Your function must finish within `budget_s` and return the best solution found

Lower is better. Current record: C5 <= {current_record_c5:.5f}. Our goal is to find a construction that shows C5 <= {target_c5:.5f}.

{state_ctx}
{construction_section}
{scaffold}
{code_section}
"""


class ErdosArchiveDataset(Dataset):
    use_sampler_for_validation = True

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer,
        config: DictConfig,
        processor=None,
        max_samples: int = -1,
        is_train: Optional[bool] = None,
    ):
        del tokenizer, processor, max_samples, is_train
        self.config = config
        self.split = _infer_split(data_files)
        self.target_c5 = float(config.get("target_c5", DEFAULT_TARGET_C5))
        self.current_record_c5 = float(config.get("current_record_c5", DEFAULT_CURRENT_RECORD_C5))
        self.num_cpus_per_task = int(config.get("num_cpus_per_task", 1))
        self.data_source = "erdos_min_overlap"
        self._sampler: Optional[ErdosPUCTSampler] = None

        if self.split == VAL_SPLIT:
            count = int(config.get("val_initial_state_count", 32))
            seed_offset = 10000
        else:
            count = int(config.get("train_initial_state_count", max(1, int(config.get("train_batch_size", 1)))))
            seed_offset = 0

        self.initial_states = [_create_initial_state(seed_offset + index, self.split) for index in range(count)]
        self._states = [copy.deepcopy(state) for state in self.initial_states]
        self._rows: list[dict[str, Any]] = []
        self._sync_rows()

    def attach_sampler(self, sampler: "ErdosPUCTSampler") -> None:
        self._sampler = sampler

    def get_states(self) -> list[ErdosStateRecord]:
        return [copy.deepcopy(state) for state in self._states]

    def reset_to_initial_archive(self) -> None:
        self.sync_from_states(self.initial_states)

    def sync_from_states(self, states: Iterable[ErdosStateRecord]) -> None:
        self._states = [copy.deepcopy(state) for state in states]
        self._sync_rows()

    def _sync_rows(self) -> None:
        budget_s = _prompt_budget_s(self.config, self.split)
        self._rows = [
            {
                "data_source": self.data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": build_erdos_prompt(
                            state,
                            budget_s=budget_s,
                            target_c5=self.target_c5,
                            current_record_c5=self.current_record_c5,
                            num_cpus_per_task=self.num_cpus_per_task,
                        ),
                    }
                ],
                "ability": "code_math",
                "reward_model": {"style": "rule", "ground_truth": str(state.raw_score)},
                "extra_info": {
                    "split": self.split,
                    "index": index,
                    "state_id": state.state_id,
                    "parent_ids": list(state.parent_ids),
                    "parent_value": float(state.value),
                    "current_raw_score": float(state.raw_score),
                    "timestep": int(state.timestep),
                    "n_points": int(len(state.construction)),
                    "initial_h_values": list(state.construction),
                },
            }
            for index, state in enumerate(self._states)
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = copy.deepcopy(self._rows[item])
        row["raw_prompt"] = copy.deepcopy(row["prompt"])
        row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
        if "extra_info" not in row or row["extra_info"] is None:
            row["extra_info"] = {}
        row["index"] = row["extra_info"].get("index", item)
        row["tools_kwargs"] = row["extra_info"].get("tools_kwargs", {})
        row["interaction_kwargs"] = row["extra_info"].get("interaction_kwargs", {})
        return row

    def on_batch_end(self, batch: DataProto, validate: bool = False) -> None:
        del batch, validate
        if self._sampler is None:
            return
        self._sampler.commit_pending_children()
        self.sync_from_states(self._sampler.get_states())

    def reset_for_validation(self) -> None:
        if self.split != VAL_SPLIT:
            return
        if self._sampler is not None:
            self._sampler.reset_for_validation()
        else:
            self.reset_to_initial_archive()


class ErdosPUCTSampler(AbstractCurriculumSampler):
    def __init__(self, data_source: Dataset, data_config: DictConfig):
        self.dataset = data_source
        if not isinstance(self.dataset, ErdosArchiveDataset):
            raise TypeError("ErdosPUCTSampler requires an ErdosArchiveDataset")

        self.dataset.attach_sampler(self)
        self.config = data_config
        self.split = self.dataset.split
        self.max_buffer_size = int(data_config.get("archive_max_size", 1000))
        self.puct_c = float(data_config.get("puct_c", 1.0))
        self.topk_children = int(data_config.get("topk_children", 2))
        self.max_construction_len = int(data_config.get("max_construction_len", DEFAULT_MAX_CONSTRUCTION_LEN))

        if self.split == VAL_SPLIT:
            val_batch_size = data_config.get("val_batch_size", None)
            self.batch_size = int(val_batch_size if val_batch_size is not None else len(self.dataset))
            self.epoch_steps = int(data_config.get("validation_puct_steps", 1))
        else:
            self.batch_size = int(data_config.get("gen_batch_size", data_config.train_batch_size))
            self.epoch_steps = int(data_config.get("train_epoch_steps", 1))

        self._initial_states = [copy.deepcopy(state) for state in self.dataset.initial_states]
        self._states = self.dataset.get_states()
        self._state_by_id: dict[str, ErdosStateRecord] = {}
        self._n: dict[str, int] = {}
        self._m: dict[str, float] = {}
        self._T = 0
        self._last_scale = 1.0
        self._pending_children: list[ErdosStateRecord] = []
        self._pending_yield_indices: list[int] = []
        self._cursor = 0
        self._rebuild_state_index()

    def _rebuild_state_index(self) -> None:
        self._state_by_id = {state.state_id: state for state in self._states}

    def get_states(self) -> list[ErdosStateRecord]:
        return [copy.deepcopy(state) for state in self._states]

    def __len__(self) -> int:
        return self.batch_size * self.epoch_steps

    def __iter__(self):
        if self._cursor >= len(self):
            self._cursor = 0
            self._pending_yield_indices = []

        while self._cursor < len(self):
            if not self._pending_yield_indices:
                remaining = len(self) - self._cursor
                next_batch_size = min(self.batch_size, remaining)
                self._pending_yield_indices.extend(self._sample_state_indices(next_batch_size))
            next_index = self._pending_yield_indices.pop(0)
            self._cursor += 1
            yield next_index

        self._cursor = 0
        self._pending_yield_indices = []

    def state_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "states": [state.to_dict() for state in self._states],
            "initial_states": [state.to_dict() for state in self._initial_states],
            "puct_n": dict(self._n),
            "puct_m": dict(self._m),
            "puct_T": int(self._T),
            "last_scale": float(self._last_scale),
            "pending_children": [state.to_dict() for state in self._pending_children],
            "pending_yield_indices": list(self._pending_yield_indices),
            "cursor": int(self._cursor),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._states = [ErdosStateRecord.from_dict(data) for data in state_dict.get("states", [])]
        self._initial_states = [ErdosStateRecord.from_dict(data) for data in state_dict.get("initial_states", [])]
        self._n = {str(key): int(value) for key, value in state_dict.get("puct_n", {}).items()}
        self._m = {str(key): float(value) for key, value in state_dict.get("puct_m", {}).items()}
        self._T = int(state_dict.get("puct_T", 0))
        self._last_scale = float(state_dict.get("last_scale", 1.0))
        self._pending_children = [
            ErdosStateRecord.from_dict(data) for data in state_dict.get("pending_children", [])
        ]
        self._pending_yield_indices = [int(index) for index in state_dict.get("pending_yield_indices", [])]
        self._cursor = int(state_dict.get("cursor", 0))
        self._rebuild_state_index()
        self.dataset.initial_states = [copy.deepcopy(state) for state in self._initial_states]
        self.dataset.sync_from_states(self._states)

    def reset_for_validation(self) -> None:
        self._states = [copy.deepcopy(state) for state in self._initial_states]
        self._n = {}
        self._m = {}
        self._T = 0
        self._last_scale = 1.0
        self._pending_children = []
        self._pending_yield_indices = []
        self._cursor = 0
        self._rebuild_state_index()
        self.dataset.sync_from_states(self._states)

    def _compute_scale(self, values: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        if values.size == 0:
            return 1.0
        used_values = values[mask] if mask is not None else values
        if used_values.size == 0:
            return 1.0
        return float(max(np.max(used_values) - np.min(used_values), 1e-6))

    def _compute_prior(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.array([])
        count = len(values)
        ranks = np.argsort(np.argsort(-values))
        weights = (count - ranks).astype(np.float64)
        return weights / weights.sum()

    def _build_children_map(self) -> dict[str, set[str]]:
        children: dict[str, set[str]] = {}
        for state in self._states:
            if state.parent_state_id:
                children.setdefault(state.parent_state_id, set()).add(state.state_id)
        return children

    def _get_full_lineage(self, state: ErdosStateRecord, children_map: dict[str, set[str]]) -> set[str]:
        lineage = {state.state_id, *state.parent_ids}
        queue = [state.state_id]
        seen = {state.state_id}
        while queue:
            current = queue.pop(0)
            for child_id in children_map.get(current, set()):
                if child_id in seen:
                    continue
                seen.add(child_id)
                lineage.add(child_id)
                queue.append(child_id)
        return lineage

    def _sample_state_indices(self, num_states: int) -> list[int]:
        if len(self._states) < num_states:
            raise ValueError(
                f"Archive only has {len(self._states)} states, but batch_size requires {num_states}. "
                "Increase the initial state count or lower the batch size."
            )

        initial_ids = {state.state_id for state in self._initial_states}
        values = np.array([float(state.value) for state in self._states], dtype=np.float64)
        non_initial_mask = np.array([state.state_id not in initial_ids for state in self._states], dtype=bool)
        scale = self._compute_scale(values, non_initial_mask if non_initial_mask.any() else None)
        self._last_scale = scale
        priors = self._compute_prior(values)
        sqrt_t = np.sqrt(1.0 + self._T)

        scored: list[tuple[float, float, int, float]] = []
        for index, state in enumerate(self._states):
            state_visits = self._n.get(state.state_id, 0)
            q_value = self._m.get(state.state_id, state.value) if state_visits > 0 else state.value
            bonus = self.puct_c * scale * priors[index] * sqrt_t / (1.0 + state_visits)
            scored.append((q_value + bonus, state.value, index, bonus))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if num_states == 1:
            return [scored[0][2]]

        children_map = self._build_children_map()
        picked: list[int] = []
        blocked_ids: set[str] = set()
        for _, _, index, _ in scored:
            state = self._states[index]
            if state.state_id in blocked_ids:
                continue
            picked.append(index)
            blocked_ids.update(self._get_full_lineage(state, children_map))
            if len(picked) >= num_states:
                break

        if len(picked) < num_states:
            for _, _, index, _ in scored:
                if index in picked:
                    continue
                picked.append(index)
                if len(picked) >= num_states:
                    break

        if len(picked) != num_states:
            raise RuntimeError(f"Failed to sample {num_states} states from archive of size {len(self._states)}")

        return picked

    def _record_visit(self, parent: ErdosStateRecord) -> None:
        for state_id in [parent.state_id, *parent.parent_ids]:
            self._n[state_id] = self._n.get(state_id, 0) + 1
        self._T += 1

    def _materialize_child_state(
        self,
        parent: ErdosStateRecord,
        child_state_data: dict[str, Any],
        global_step: int,
    ) -> ErdosStateRecord:
        raw_score = float(child_state_data["raw_score"])
        child_code = str(child_state_data.get("code", ""))
        child_stdout = str(child_state_data.get("stdout", ""))
        construction = _to_float_list(child_state_data.get("construction", []))
        return ErdosStateRecord(
            state_id=str(uuid.uuid4()),
            timestep=int(global_step),
            value=-raw_score,
            raw_score=raw_score,
            construction=construction,
            code=child_code,
            stdout=child_stdout,
            split=self.split,
            parent_state_id=parent.state_id,
            parent_ids=[parent.state_id, *parent.parent_ids],
            parent_values=[parent.value, *parent.parent_values],
        )

    def update(self, batch: DataProto) -> None:
        parent_ids = [str(value) for value in _as_list(batch.non_tensor_batch.get("parent_state_id"))]
        if not parent_ids:
            return

        valids = [bool(value) for value in _as_list(batch.non_tensor_batch.get("valid"))]
        raw_scores = _as_list(batch.non_tensor_batch.get("raw_score"))
        child_states = _as_list(batch.non_tensor_batch.get("child_state"))
        grouped: dict[str, tuple[float, dict[str, Any]]] = {}
        visited_parents = set()
        global_step = int(batch.meta_info.get("global_steps", 0))

        for parent_id, valid, raw_score, child_state in zip(parent_ids, valids, raw_scores, child_states):
            visited_parents.add(parent_id)
            if not valid or not isinstance(child_state, dict):
                continue
            score_value = float(raw_score)
            best = grouped.get(parent_id)
            if best is None or score_value < best[0]:
                grouped[parent_id] = (score_value, child_state)

        for parent_id in visited_parents:
            parent = self._state_by_id.get(parent_id)
            if parent is None:
                continue
            best = grouped.get(parent_id)
            if best is None:
                self._record_visit(parent)
                continue

            child_state = self._materialize_child_state(parent, best[1], global_step=global_step)
            self._m[parent_id] = max(self._m.get(parent_id, child_state.value), child_state.value)
            self._record_visit(parent)
            self._pending_children.append(child_state)

    def _prune_archive(self) -> None:
        initial_ids = {state.state_id for state in self._initial_states}

        if self.topk_children > 0:
            by_parent: dict[str, list[ErdosStateRecord]] = defaultdict(list)
            initial_states = []
            for state in self._states:
                if state.state_id in initial_ids:
                    initial_states.append(state)
                elif state.parent_state_id:
                    by_parent[state.parent_state_id].append(state)

            filtered_states = list(initial_states)
            for children in by_parent.values():
                children.sort(key=lambda item: item.value, reverse=True)
                filtered_states.extend(children[: self.topk_children])
            self._states = filtered_states

        if len(self._states) <= self.max_buffer_size:
            return

        initial_indices = [index for index, state in enumerate(self._states) if state.state_id in initial_ids]
        keep = set(initial_indices)
        value_order = sorted(
            range(len(self._states)),
            key=lambda index: self._states[index].value,
            reverse=True,
        )
        for index in value_order:
            if len(keep) >= self.max_buffer_size:
                break
            keep.add(index)
        self._states = [self._states[index] for index in sorted(keep)]

    def commit_pending_children(self) -> None:
        if not self._pending_children:
            return

        existing = {_construction_key(state.construction) for state in self._states}
        existing.discard(None)
        new_states: list[ErdosStateRecord] = []
        for state in self._pending_children:
            if state.value is None or not state.construction:
                continue
            if len(state.construction) > self.max_construction_len:
                continue
            key = _construction_key(state.construction)
            if key is None or key in existing:
                continue
            new_states.append(state)
            existing.add(key)

        self._pending_children = []
        if not new_states:
            return

        self._states.extend(new_states)
        self._prune_archive()
        self._rebuild_state_index()


def _extract_python_code(response: str) -> Optional[str]:
    match = re.search(r"```python\s+([\s\S]*?)\s*```", response)
    if match is None:
        return None
    code = match.group(1).strip()
    return code or None


def _validate_program_entrypoint(code: str, entrypoint: str = "run") -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno is not None else "unknown location"
        return f"Invalid python code: {exc.msg} ({location})."

    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entrypoint:
            return None
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == entrypoint for target in node.targets):
                return None
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == entrypoint:
                return None

    return f"Missing required top-level entrypoint `{entrypoint}`."


def _build_program(code: str, initial_h_values: list[float]) -> str:
    helper_source = (
        "import numpy as np\n\n"
        f"{inspect_source(verify_c5_solution)}\n\n"
        f"{inspect_source(evaluate_erdos_solution)}\n\n"
        f"{inspect_source(verify_erdos_solution)}\n\n"
        f"initial_h_values = np.array({list(initial_h_values)!r}, dtype=np.float64)\n\n"
    )
    return helper_source + code


def inspect_source(fn) -> str:
    import inspect

    return inspect.getsource(fn)


def _ensure_future_annotations(program_code: str) -> str:
    future_import = "from __future__ import annotations"
    if future_import in program_code:
        return program_code
    return future_import + "\n\n" + program_code


def _run_code_in_subprocess(
    code: str,
    *,
    timeout_seconds: float,
    num_cpus: int,
) -> tuple[Any, str]:
    program_code = _ensure_future_annotations(code)
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as program_file:
        program_file.write(program_code)
        program_path = program_file.name

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as runner_file:
        runner_path = runner_file.name

    results_path = runner_path + ".pkl"
    stdout_path = runner_path + ".stdout"
    stderr_path = runner_path + ".stderr"
    runner_script = f"""
import ast
import builtins
import importlib.util
import os
import pickle
import socket
import sys
import traceback

PROGRAM_PATH = {program_path!r}
RESULTS_PATH = {results_path!r}
STDOUT_PATH = {stdout_path!r}
STDERR_PATH = {stderr_path!r}
ALLOWED_WRITE_PATHS = {{
    os.path.abspath(RESULTS_PATH),
    os.path.abspath(STDOUT_PATH),
    os.path.abspath(STDERR_PATH),
}}

def _blocked(*args, **kwargs):
    raise PermissionError("Filesystem mutation disabled in Erdos sandbox")

def _disable_mutations():
    orig_open = builtins.open
    def _ro_open(file, mode="r", *args, **kwargs):
        file_path = os.path.abspath(file) if isinstance(file, (str, os.PathLike)) else None
        if any(flag in mode for flag in ("w", "a", "x", "+")) and file_path not in ALLOWED_WRITE_PATHS:
            raise PermissionError("File writes are disabled in Erdos sandbox")
        return orig_open(file, mode, *args, **kwargs)
    builtins.open = _ro_open
    for name in ("remove", "unlink", "rename", "replace", "rmdir", "mkdir", "makedirs", "chmod", "chown"):
        if hasattr(os, name):
            setattr(os, name, _blocked)
    orig_connect = socket.socket.connect
    def _blocked_connect(self, address):
        raise PermissionError("Network access is disabled in Erdos sandbox")
    socket.socket.connect = _blocked_connect
    return orig_connect

def _filter_call_kwargs(func, kwargs):
    import inspect
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    filtered = {{}}
    for name, value in kwargs.items():
        param = params.get(name)
        if param is None:
            continue
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            filtered[name] = value
    return filtered

def main():
    stdout_fh = open(STDOUT_PATH, "w", encoding="utf-8")
    stderr_fh = open(STDERR_PATH, "w", encoding="utf-8")
    sys.stdout = stdout_fh
    sys.stderr = stderr_fh
    try:
        _disable_mutations()
        spec = importlib.util.spec_from_file_location("erdos_program", PROGRAM_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        func = getattr(module, "run")
        kwargs = _filter_call_kwargs(func, {{"seed": 42, "budget_s": float({timeout_seconds!r})}})
        result = func(**kwargs)
        with open(RESULTS_PATH, "wb") as handle:
            pickle.dump({{"result": result}}, handle)
    except Exception as exc:
        traceback.print_exc()
        with open(RESULTS_PATH, "wb") as handle:
            pickle.dump({{"error": str(exc)}}, handle)
    finally:
        stdout_fh.flush()
        stderr_fh.flush()
        stdout_fh.close()
        stderr_fh.close()

if __name__ == "__main__":
    main()
"""
    with open(runner_path, "w", encoding="utf-8") as handle:
        handle.write(runner_script)

    env = os.environ.copy()
    thread_cap = str(max(1, int(num_cpus)))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env.setdefault(variable, thread_cap)

    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, runner_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                process.wait(timeout=1.0)
            except Exception:
                pass
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                pass
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds") from exc

        if process.returncode != 0 and not os.path.exists(results_path):
            raise RuntimeError(f"Program exited with code {process.returncode}")

        stdout = ""
        if os.path.exists(stdout_path):
            with open(stdout_path, "r", encoding="utf-8", errors="ignore") as handle:
                stdout = handle.read()

        if not os.path.exists(results_path):
            raise RuntimeError("Results file not found")

        with open(results_path, "rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(payload["error"])
        return payload.get("result"), stdout
    finally:
        if process is not None:
            for stream in (process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
        for path in (program_path, runner_path, results_path, stdout_path, stderr_path):
            try:
                os.unlink(path)
            except OSError:
                pass


def compute_score_erdos(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    *,
    train_budget_s: float = 10,
    val_budget_s: float = 30,
    num_cpus_per_task: int = 1,
) -> dict[str, Any]:
    del data_source, ground_truth
    extra_info = dict(extra_info or {})
    split = extra_info.get("split", TRAIN_SPLIT)
    budget_s = float(val_budget_s if split == VAL_SPLIT else train_budget_s)
    parent_state_id = str(extra_info.get("state_id", ""))
    parent_value = float(extra_info.get("parent_value", 0.0))
    parent_raw_score = float(extra_info.get("current_raw_score", _state_value_to_raw_score(parent_value)))
    initial_h_values = _to_float_list(extra_info.get("initial_h_values", []))
    if not initial_h_values:
        n_points = int(extra_info.get("n_points", 200))
        initial_h_values = [0.5] * n_points

    result = {
        "score": 0.0,
        "acc": 0.0,
        "raw_score": parent_raw_score,
        "baseline_c5": parent_raw_score,
        "delta_c5": 0.0,
        "improved": False,
        "valid": False,
        "status": "invalid",
        "stdout": "",
        "n_points": int(len(initial_h_values)),
        "parent_state_id": parent_state_id,
        "parent_value": parent_value,
        "child_state": None,
    }

    code = _extract_python_code(solution_str)
    if code is None:
        result["status"] = "missing_code"
        return result

    entrypoint_error = _validate_program_entrypoint(code)
    if entrypoint_error is not None:
        result["status"] = "missing_entrypoint"
        result["stdout"] = entrypoint_error
        return result

    try:
        program = _build_program(code, initial_h_values)
        execution_result, stdout = _run_code_in_subprocess(
            program,
            timeout_seconds=budget_s,
            num_cpus=num_cpus_per_task,
        )
        result["stdout"] = stdout
        if not verify_erdos_solution(execution_result):
            result["status"] = "invalid_solution"
            return result

        h_values, c5_bound, n_points = execution_result
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        child_state = {
            "construction": _to_float_list(np.asarray(h_values, dtype=np.float64)),
            "raw_score": float(c5_bound),
            "stdout": stdout,
            "code": code,
        }
        result.update(
            {
                "score": float(1.0 / (1e-8 + c5_bound)),
                "acc": 1.0,
                "raw_score": float(c5_bound),
                "delta_c5": float(parent_raw_score - c5_bound),
                "improved": bool(c5_bound < parent_raw_score),
                "valid": True,
                "status": "ok",
                "n_points": int(n_points),
                "child_state": child_state,
            }
        )
        return result
    except TimeoutError:
        result["status"] = "timeout"
        return result
    except Exception as exc:
        result["status"] = "error"
        result["stdout"] = traceback.format_exc() + f"\n{exc}"
        return result
