"""Survival-model abstractions for time-to-payment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lifelines import CoxPHFitter

from predictor.config import PredictorConfig


@dataclass
class SurvivalModel:
    fitter: CoxPHFitter

    def hazard_curve(self, frame) -> Any:  # type: ignore[no-untyped-def]
        return self.fitter.predict_partial_hazard(frame)


def train_cox_model(frame, config: PredictorConfig) -> SurvivalModel:  # type: ignore[no-untyped-def]
    fitter = CoxPHFitter()
    fitter.fit(
        frame,
        duration_col=config.model.survival_label or "time_to_event",
        event_col="event",
    )
    return SurvivalModel(fitter=fitter)
