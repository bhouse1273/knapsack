"""Training pipeline for baseline models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from predictor.config import PredictorConfig
from predictor.features.builder import build_features
from predictor.models.baseline import train_hurdle_model


def run(config: PredictorConfig) -> Path:
    """Trains models and writes artifacts to disk."""

    frame = build_features(config).to_pandas()
    labels = pd.DataFrame({"amount_collected": np.maximum(frame["balance"] * 0.05, 0)})
    model = train_hurdle_model(frame, labels, config)

    artifact_dir = Path(config.data_paths.model_artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"hurdle_{config.model.strategy_set}.joblib"
    joblib.dump(model, artifact_path)
    return artifact_path
