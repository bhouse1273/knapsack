from pathlib import Path

from predictor.config import PredictorConfig
from predictor.features.builder import build_features


def test_build_features(tmp_path: Path) -> None:
    config = PredictorConfig.from_env(tmp_path)
    frame = build_features(config)
    assert not frame.is_empty()
    assert (Path(config.data_paths.feature_store) / "features.parquet").exists()
