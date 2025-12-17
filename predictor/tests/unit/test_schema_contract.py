from pathlib import Path

from predictor.io.schemas import PredictorSchema, SCHEMA_PATH


def test_schema_loads() -> None:
    schema = PredictorSchema.load(SCHEMA_PATH)
    assert schema.name == "predictor_item_scores"
    required = {field["name"] for field in schema.fields if field.get("required")}
    assert {"account_id", "strategy_id", "predicted_value"}.issubset(required)
