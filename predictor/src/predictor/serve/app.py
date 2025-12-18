"""FastAPI app exposing online scoring."""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from predictor.config import PredictorConfig
from predictor.features.builder import build_features


class ScoreRequest(BaseModel):
    account_ids: List[str]


app = FastAPI(title="Knapsack Predictor")
_config: PredictorConfig | None = None
_model = None


def load_app(config: PredictorConfig) -> FastAPI:
    global _config, _model
    _config = config
    artifact_path = Path(config.data_paths.model_artifacts) / f"hurdle_{config.model.strategy_set}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError("Model artifact missing; run training first.")
    _model = joblib.load(artifact_path)
    return app


@app.post("/score")
async def score(req: ScoreRequest):
    if _config is None or _model is None:
        raise RuntimeError("App not initialized; call load_app before serving requests.")

    features = build_features(_config)
    filtered = features.filter(features["account_id"].is_in(req.account_ids)).to_pandas()
    predictions = _model.predict_expected_value(filtered)
    return {
        "schema_version": "1.0.0",
        "scores": [
            {
                "account_id": acct,
                "expected_value": float(pred),
            }
            for acct, pred in zip(filtered["account_id"], predictions, strict=False)
        ],
    }
