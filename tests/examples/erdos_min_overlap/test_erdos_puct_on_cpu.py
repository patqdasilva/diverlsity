import copy
import unittest
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from examples.erdos_min_overlap.erdos_puct import (
    ErdosArchiveDataset,
    ErdosPUCTSampler,
    compute_score_erdos,
)


def _make_config(**overrides):
    config = {
        "train_batch_size": 2,
        "gen_batch_size": 2,
        "train_epoch_steps": 1,
        "train_initial_state_count": 3,
        "val_initial_state_count": 2,
        "val_batch_size": 2,
        "validation_puct_steps": 2,
        "archive_max_size": 8,
        "topk_children": 1,
        "puct_c": 1.0,
        "train_budget_s": 1,
        "val_budget_s": 1,
        "num_cpus_per_task": 1,
    }
    config.update(overrides)
    return OmegaConf.create(config)


def _make_batch(parent_state_ids, valids, raw_scores, child_states, *, global_steps=1):
    return SimpleNamespace(
        non_tensor_batch={
            "parent_state_id": np.array(parent_state_ids, dtype=object),
            "valid": np.array(valids, dtype=object),
            "raw_score": np.array(raw_scores, dtype=object),
            "child_state": np.array(child_states, dtype=object),
        },
        meta_info={"global_steps": global_steps},
    )


class ErdosPUCTSamplerTests(unittest.TestCase):
    def setUp(self):
        self.config = _make_config()
        self.dataset = ErdosArchiveDataset("erdos_train", tokenizer=None, config=self.config)
        self.sampler = ErdosPUCTSampler(self.dataset, self.config)

    def test_sampler_keeps_best_child_and_blocks_lineage(self):
        parent = self.dataset.get_states()[0]
        base_construction = list(parent.construction)
        worse_child = {
            "construction": [min(1.0, value + 1e-4) for value in base_construction],
            "raw_score": parent.raw_score - 0.001,
            "stdout": "worse",
            "code": "def run(): pass",
        }
        best_child = {
            "construction": [max(0.0, value - 1e-4) for value in base_construction],
            "raw_score": parent.raw_score - 0.002,
            "stdout": "best",
            "code": "def run(): pass",
        }
        batch = _make_batch(
            [parent.state_id, parent.state_id],
            [True, True],
            [worse_child["raw_score"], best_child["raw_score"]],
            [worse_child, best_child],
            global_steps=3,
        )

        self.sampler.update(batch)
        self.dataset.on_batch_end(batch=batch)

        states = self.dataset.get_states()
        children = [state for state in states if state.parent_state_id == parent.state_id]
        self.assertEqual(len(children), 1)
        self.assertAlmostEqual(children[0].raw_score, best_child["raw_score"])
        self.assertEqual(self.sampler._T, 1)
        self.assertEqual(self.sampler._n[parent.state_id], 1)

        better_child = {
            "construction": [max(0.0, value - 2e-4) for value in base_construction],
            "raw_score": parent.raw_score - 0.003,
            "stdout": "better",
            "code": "def run(): pass",
        }
        next_batch = _make_batch(
            [parent.state_id, parent.state_id],
            [True, False],
            [better_child["raw_score"], parent.raw_score],
            [better_child, None],
            global_steps=4,
        )
        self.sampler.update(next_batch)
        self.dataset.on_batch_end(batch=next_batch)

        states = self.dataset.get_states()
        children = [state for state in states if state.parent_state_id == parent.state_id]
        self.assertEqual(len(children), 1)
        self.assertAlmostEqual(children[0].raw_score, better_child["raw_score"])

        sampled_indices = self.sampler._sample_state_indices(2)
        sampled_ids = {self.sampler.get_states()[index].state_id for index in sampled_indices}
        self.assertFalse({parent.state_id, children[0].state_id}.issubset(sampled_ids))

    def test_failed_rollout_records_visit(self):
        parent = self.dataset.get_states()[1]
        batch = _make_batch(
            [parent.state_id, parent.state_id],
            [False, False],
            [parent.raw_score, parent.raw_score],
            [None, None],
            global_steps=2,
        )

        self.sampler.update(batch)
        self.dataset.on_batch_end(batch=batch)

        self.assertEqual(len(self.dataset), 3)
        self.assertEqual(self.sampler._T, 1)
        self.assertEqual(self.sampler._n[parent.state_id], 1)

    def test_state_dict_roundtrip_restores_archive_and_stats(self):
        parent = self.dataset.get_states()[0]
        child_state = {
            "construction": [min(1.0, value + 5e-5) for value in parent.construction],
            "raw_score": parent.raw_score - 0.0015,
            "stdout": "child",
            "code": "def run(): pass",
        }
        batch = _make_batch(
            [parent.state_id, parent.state_id],
            [True, False],
            [child_state["raw_score"], parent.raw_score],
            [child_state, None],
            global_steps=5,
        )
        self.sampler.update(batch)
        self.dataset.on_batch_end(batch=batch)

        saved_state = copy.deepcopy(self.sampler.state_dict())
        new_dataset = ErdosArchiveDataset("erdos_train", tokenizer=None, config=self.config)
        new_sampler = ErdosPUCTSampler(new_dataset, self.config)
        new_sampler.load_state_dict(saved_state)

        self.assertEqual(
            [state.state_id for state in self.dataset.get_states()],
            [state.state_id for state in new_dataset.get_states()],
        )
        self.assertEqual(self.sampler._n, new_sampler._n)
        self.assertEqual(self.sampler._m, new_sampler._m)
        self.assertEqual(self.sampler._T, new_sampler._T)


class ErdosRewardTests(unittest.TestCase):
    def test_reward_accepts_valid_program(self):
        initial_h_values = [0.5, 0.5, 0.5, 0.5]
        solution = """```python
import numpy as np

def run(seed=42, budget_s=1, **kwargs):
    h = np.asarray(initial_h_values, dtype=np.float64).copy()
    n_points = int(h.size)
    c5_bound = float(np.max(np.correlate(h, 1.0 - h, mode="full") * (2.0 / n_points)))
    return h, c5_bound, n_points
```"""
        result = compute_score_erdos(
            data_source="erdos_min_overlap",
            solution_str=solution,
            ground_truth="",
            extra_info={
                "split": "train",
                "state_id": "state-1",
                "parent_value": -0.5,
                "current_raw_score": 0.5,
                "initial_h_values": initial_h_values,
                "n_points": len(initial_h_values),
            },
            train_budget_s=1,
            val_budget_s=1,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["score"], 0.0)
        self.assertIsInstance(result["child_state"], dict)

    def test_reward_rejects_missing_code_block(self):
        result = compute_score_erdos(
            data_source="erdos_min_overlap",
            solution_str="not code",
            ground_truth="",
            extra_info={"split": "train", "state_id": "state-2", "parent_value": -0.5, "current_raw_score": 0.5},
            train_budget_s=1,
            val_budget_s=1,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "missing_code")

    def test_reward_rejects_missing_entrypoint(self):
        result = compute_score_erdos(
            data_source="erdos_min_overlap",
            solution_str="```python\nx = 1\n```",
            ground_truth="",
            extra_info={"split": "train", "state_id": "state-3", "parent_value": -0.5, "current_raw_score": 0.5},
            train_budget_s=1,
            val_budget_s=1,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "missing_entrypoint")

    def test_reward_rejects_invalid_solution(self):
        initial_h_values = [0.5, 0.5, 0.5, 0.5]
        solution = """```python
import numpy as np

def run(seed=42, budget_s=1, **kwargs):
    h = np.asarray(initial_h_values, dtype=np.float64).copy()
    n_points = int(h.size)
    return h, 0.123, n_points
```"""
        result = compute_score_erdos(
            data_source="erdos_min_overlap",
            solution_str=solution,
            ground_truth="",
            extra_info={
                "split": "train",
                "state_id": "state-4",
                "parent_value": -0.5,
                "current_raw_score": 0.5,
                "initial_h_values": initial_h_values,
                "n_points": len(initial_h_values),
            },
            train_budget_s=1,
            val_budget_s=1,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "invalid_solution")

    def test_reward_times_out(self):
        solution = """```python
import time

def run(seed=42, budget_s=1, **kwargs):
    while True:
        time.sleep(0.05)
```"""
        result = compute_score_erdos(
            data_source="erdos_min_overlap",
            solution_str=solution,
            ground_truth="",
            extra_info={"split": "train", "state_id": "state-5", "parent_value": -0.5, "current_raw_score": 0.5},
            train_budget_s=1,
            val_budget_s=1,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "timeout")


if __name__ == "__main__":
    unittest.main()
