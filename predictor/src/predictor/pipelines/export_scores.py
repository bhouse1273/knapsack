"""Batch scoring pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import polars as pl

from predictor.config import PredictorConfig
from predictor.features.builder import build_features


def run(config: PredictorConfig, as_of: datetime) -> Path:
    """Exports an Arrow/Parquet dataset with expected values and hazard placeholders."""

    features = build_features(config)
    artifact_path = Path(config.data_paths.model_artifacts) / f"hurdle_{config.model.strategy_set}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact at {artifact_path}. Train models before exporting scores."
        )

    model = joblib.load(artifact_path)
    expected = model.predict_expected_value(features.to_pandas())
    scored = features.with_columns(
        predicted_value=pl.Series(expected),
        schema_version=pl.lit("1.0.0"),
        as_of_date=pl.lit(as_of.date().isoformat()),
    )

    batch_output_dir = Path(config.data_paths.batch_output)
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = batch_output_dir / f"scores_{as_of.date().isoformat()}.parquet"
    scored.write_parquet(output_path)
    return output_path
