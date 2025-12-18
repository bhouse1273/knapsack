from pathlib import Path

import numpy as np

from predictor.models import (
    load_fixture,
    predict_expected_value,
    train_demo_regressor,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "catboost_accounts.json"
CAT_FEATURES = ["state", "product", "strategy_id"]


def test_catboost_fixture_smoke() -> None:
    features, labels = load_fixture(FIXTURE)
    model = train_demo_regressor(features, labels, CAT_FEATURES, iterations=60)
    predictions = predict_expected_value(model, features, CAT_FEATURES)
    assert predictions.shape == (len(labels),)
    assert float(np.corrcoef(predictions, labels)[0, 1]) > 0.98


def test_catboost_respects_categories() -> None:
    features, labels = load_fixture(FIXTURE)
    model = train_demo_regressor(features, labels, CAT_FEATURES, iterations=45)

    sample = features.iloc[[0]].copy()
    baseline = predict_expected_value(model, sample, CAT_FEATURES)[0]

    alt_state = features.loc[features["state"] != sample.iloc[0]["state"], "state"].iloc[0]
    mutated = sample.copy()
    mutated.loc[:, "state"] = alt_state
    shifted = predict_expected_value(model, mutated, CAT_FEATURES)[0]

    assert abs(baseline - shifted) > 1.0
