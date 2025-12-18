"""Feature engineering pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import polars as pl

from predictor.config import PredictorConfig


def build_features(config: PredictorConfig) -> pl.DataFrame:
    """Creates a deterministic feature view.

    The initial implementation operates on synthetic data so `knapsack` can begin
    integration immediately. Replace the placeholder logic with real ETL once data
    contracts are finalized.
    """

    synthetic_rows: Dict[str, Any] = {
        "account_id": [f"acct_{i}" for i in range(10)],
        "balance": [1000 + 25 * i for i in range(10)],
        "days_past_due": [i * 3 for i in range(10)],
        "strategy_id": ["default"] * 10,
    }
    feature_frame = pl.DataFrame(synthetic_rows)
    output_path = Path(config.data_paths.feature_store)
    output_path.mkdir(parents=True, exist_ok=True)
    feature_frame.write_parquet(output_path / "features.parquet")
    return feature_frame
