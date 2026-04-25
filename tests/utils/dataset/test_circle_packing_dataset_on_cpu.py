from __future__ import annotations

import pandas as pd

from recipe.circle_packing import create_dataset


def test_circle_packing_dataset_generation(tmp_path) -> None:
    output_dir = tmp_path / "circle_packing"
    specs = create_dataset.parse_variants("26,32")

    create_dataset.write_datasets(
        output_dir=str(output_dir),
        specs=specs,
        train_repeats=2,
        val_repeats=1,
    )

    train_df = pd.read_parquet(output_dir / "train.parquet")
    test_df = pd.read_parquet(output_dir / "test.parquet")

    expected_columns = {"prompt", "data_source", "ability", "reward_model", "extra_info"}
    assert expected_columns.issubset(train_df.columns)
    assert expected_columns.issubset(test_df.columns)

    assert len(train_df) == 4
    assert len(test_df) == 2
    assert set(train_df["data_source"]) == {"circle_packing_26", "circle_packing_32"}
    assert set(test_df["data_source"]) == {"circle_packing_26", "circle_packing_32"}

    for extra_info in train_df["extra_info"]:
        assert "num_circles" in extra_info
        assert "target" in extra_info
        assert "split" in extra_info
        assert "index" in extra_info
        assert "variant" in extra_info

    prompt_text = train_df.iloc[0]["prompt"][0]["content"]
    assert "run_packing" in prompt_text
    assert "validate_packing" in prompt_text
    assert "sum of radii" in prompt_text
