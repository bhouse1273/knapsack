"""Schema helpers shared between predictor and knapsack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json


@dataclass(frozen=True)
class PredictorSchema:
    name: str
    version: str
    fields: List[Dict[str, str]]

    @staticmethod
    def load(schema_path: Path) -> "PredictorSchema":
        content = json.loads(schema_path.read_text())
        return PredictorSchema(
            name=content["name"],
            version=content["version"],
            fields=content["fields"],
        )


def _discover_schema_path() -> Path:
    """Walk upwards until the predictor schema directory is located."""

    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        schema_dir = candidate / "shared" / "schemas" / "predictor"
        schema_file = schema_dir / "item_scores.schema.json"
        if schema_file.exists():
            return schema_file
    raise FileNotFoundError(
        "Unable to locate predictor schema. Ensure shared/schemas/predictor exists in the repo tree."
    )


SCHEMA_PATH = _discover_schema_path()
