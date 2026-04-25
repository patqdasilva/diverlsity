from __future__ import annotations

import pytest

from recipe.circle_packing import reward_function


VALID_26_CODE = """
```python
import numpy as np

def run_packing():
    centers = []
    xs = [0.08, 0.24, 0.40, 0.56, 0.72, 0.88]
    ys = [0.10, 0.30, 0.50, 0.70, 0.90]
    for y in ys:
        for x in xs:
            centers.append([x, y])
    centers = np.array(centers[:26], dtype=float)
    radii = np.full(26, 0.05, dtype=float)
    return centers, radii, 999.0
```
"""


def test_circle_packing_reward_valid_fenced_code() -> None:
    result = reward_function.compute_score(
        data_source="circle_packing_26",
        solution_str=VALID_26_CODE,
        ground_truth="",
        extra_info={"num_circles": 26},
        timeout_seconds=5,
    )

    assert result["valid"] == 1.0
    assert result["sum_radii"] == pytest.approx(1.3)
    assert result["score"] == pytest.approx(1.3)
    assert result["failure_reason"] == "valid"


def test_circle_packing_reward_extracts_last_code_block() -> None:
    completion = """
<strategy>
Try a simple grid to get a valid baseline.
</strategy>

```python
def wrong():
    return 0
```

Final answer:
```python
import numpy as np

def run_packing():
    centers = np.array([[0.25, 0.25], [0.75, 0.75]] * 13, dtype=float)
    radii = np.full(26, 0.01, dtype=float)
    return centers, radii, 0.0
```
"""
    result = reward_function.compute_score(
        data_source="circle_packing_26",
        solution_str=completion,
        ground_truth="",
        extra_info={"num_circles": 26},
        timeout_seconds=5,
    )

    assert result["score"] == 0.0
    assert result["failure_reason"] == "circles 0 and 2 overlap"


def test_circle_packing_reward_ignores_inline_backtick_mentions() -> None:
    completion = """
<think>
I should follow the instruction exactly and return the final program between ```python and ```.
</think>

<strategy>
Use a simple non-overlapping grid baseline.
</strategy>

```python
import numpy as np

def run_packing():
    centers = []
    xs = [0.08, 0.24, 0.40, 0.56, 0.72, 0.88]
    ys = [0.10, 0.30, 0.50, 0.70, 0.90]
    for y in ys:
        for x in xs:
            centers.append([x, y])
    centers = np.array(centers[:26], dtype=float)
    radii = np.full(26, 0.05, dtype=float)
    return centers, radii, 0.0
```
"""
    result = reward_function.compute_score(
        data_source="circle_packing_26",
        solution_str=completion,
        ground_truth="",
        extra_info={"num_circles": 26},
        timeout_seconds=5,
    )

    assert result["valid"] == 1.0
    assert result["score"] == pytest.approx(1.3)
    assert result["failure_reason"] == "valid"


def test_circle_packing_reward_rejects_overlap() -> None:
    result = reward_function.compute_score(
        data_source="circle_packing_26",
        solution_str="""
import numpy as np

def run_packing():
    centers = np.tile(np.array([[0.50, 0.50]], dtype=float), (26, 1))
    radii = np.full(26, 0.01, dtype=float)
    return centers, radii, 0.0
""",
        ground_truth="",
        extra_info={"num_circles": 26},
        timeout_seconds=5,
    )

    assert result["score"] == 0.0
    assert result["valid"] == 0.0
    assert "overlap" in result["failure_reason"]


def test_circle_packing_reward_rejects_out_of_bounds() -> None:
    result = reward_function.compute_score(
        data_source="circle_packing_26",
        solution_str="""
import numpy as np

def run_packing():
    centers = np.tile(np.array([[0.01, 0.50]], dtype=float), (26, 1))
    radii = np.full(26, 0.02, dtype=float)
    return centers, radii, 0.0
""",
        ground_truth="",
        extra_info={"num_circles": 26},
        timeout_seconds=5,
    )

    assert result["score"] == 0.0
    assert "outside the unit square" in result["failure_reason"]


def test_circle_packing_reward_rejects_wrong_shape() -> None:
    result = reward_function.compute_score(
        data_source="circle_packing_26",
        solution_str="""
import numpy as np

def run_packing():
    centers = np.zeros((25, 2), dtype=float)
    radii = np.full(26, 0.01, dtype=float)
    return centers, radii, 0.0
""",
        ground_truth="",
        extra_info={"num_circles": 26},
        timeout_seconds=5,
    )

    assert result["score"] == 0.0
    assert "centers shape" in result["failure_reason"]


def test_circle_packing_reward_rejects_missing_run_packing() -> None:
    result = reward_function.compute_score(
        data_source="circle_packing_26",
        solution_str="""
def helper():
    return 0
""",
        ground_truth="",
        extra_info={"num_circles": 26},
        timeout_seconds=5,
    )

    assert result["score"] == 0.0
    assert "run_packing" in result["failure_reason"]


def test_circle_packing_reward_handles_timeout() -> None:
    result = reward_function.compute_score(
        data_source="circle_packing_26",
        solution_str="""
def run_packing():
    while True:
        pass
""",
        ground_truth="",
        extra_info={"num_circles": 26},
        timeout_seconds=1,
    )

    assert result["score"] == 0.0
    assert "timed out" in result["failure_reason"]


def test_circle_packing_reward_batch_smoke() -> None:
    results = reward_function.compute_score_batch(
        data_sources=["circle_packing_26", "circle_packing_26"],
        solution_strs=[
            VALID_26_CODE,
            """
def run_packing():
    return [], [], 0.0
""",
        ],
        ground_truths=["", ""],
        extra_infos=[{"num_circles": 26}, {"num_circles": 26}],
        timeout_seconds=5,
        max_workers=2,
    )

    assert len(results) == 2
    assert results[0]["score"] == pytest.approx(1.3)
    assert results[1]["score"] == 0.0
