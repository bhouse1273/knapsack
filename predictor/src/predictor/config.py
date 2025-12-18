"""Shared configuration objects for predictor workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(slots=True)
class DataPaths:
    raw_data: Path
    feature_store: Path
    model_artifacts: Path
    batch_output: Path


@dataclass(slots=True)
class ModelConfig:
    strategy_set: str = "default"
    horizon_days: int = 90
    objective: str = "expected_cash"
    features: List[str] = field(default_factory=list)
    label: str = "amount_collected"
    survival_label: Optional[str] = None


@dataclass(slots=True)
class ServingConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    reload: bool = False


@dataclass(slots=True)
class PredictorConfig:
    data_paths: DataPaths
    model: ModelConfig = field(default_factory=ModelConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    @staticmethod
    def from_env(base_dir: Path) -> "PredictorConfig":
        """Constructs a default config rooted at the supplied base directory."""

        base_dir = base_dir.expanduser().resolve()
        return PredictorConfig(
            data_paths=DataPaths(
                raw_data=base_dir / "data" / "raw",
                feature_store=base_dir / "data" / "features",
                model_artifacts=base_dir / "data" / "models",
                batch_output=base_dir / "data" / "scores",
            )
        )
