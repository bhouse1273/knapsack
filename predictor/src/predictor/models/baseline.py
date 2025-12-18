"""Baseline gradient boosted + hurdle models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from predictor.config import PredictorConfig


@dataclass
class HurdleModel:
    classifier: GradientBoostingClassifier
    regressor: GradientBoostingRegressor

    def predict_expected_value(self, x) -> np.ndarray:  # type: ignore[no-untyped-def]
        pay_prob = self.classifier.predict_proba(x)[:, 1]
        conditional_amount = self.regressor.predict(x)
        return pay_prob * np.clip(conditional_amount, a_min=0.0, a_max=None)


def train_hurdle_model(features, labels, config: PredictorConfig) -> HurdleModel:
    """Trains the two-stage hurdle model using default hyper parameters."""

    clf = GradientBoostingClassifier(random_state=0)
    reg = GradientBoostingRegressor(random_state=0)

    # Placeholder logic: assumes labels is a dict containing the required columns.
    y_binary = labels[config.model.label] > 0
    y_amount = labels[config.model.label]

    clf.fit(features, y_binary)
    reg.fit(features[y_binary], y_amount[y_binary])
    return HurdleModel(classifier=clf, regressor=reg)
