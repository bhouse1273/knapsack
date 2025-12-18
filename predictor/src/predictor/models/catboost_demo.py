"""CatBoost demonstration utilities for fixtures and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


def load_fixture(fixture_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a JSON fixture where `amount_collected` is the label column."""

    frame = pd.read_json(fixture_path)
    if "account_id" in frame.columns:
        frame = frame.drop(columns=["account_id"])
    labels = frame.pop("amount_collected")
    return frame, labels


def train_demo_regressor(
    features: pd.DataFrame,
    labels: pd.Series,
    categorical: Sequence[str],
    *,
    iterations: int = 40,
) -> CatBoostRegressor:
    """Train a small CatBoost regressor on tabular features."""

    pool = Pool(features, label=labels, cat_features=list(categorical))
    model = CatBoostRegressor(
        iterations=iterations,
        depth=4,
        learning_rate=0.15,
        loss_function="RMSE",
        random_seed=7,
        verbose=False,
        task_type="CPU",
    )
    model.fit(pool)
    return model


def predict_expected_value(
    model: CatBoostRegressor,
    features: pd.DataFrame,
    categorical: Sequence[str],
) -> np.ndarray:
    """Generate predictions while respecting categorical indices."""

    pool = Pool(features, cat_features=list(categorical))
    return model.predict(pool)
