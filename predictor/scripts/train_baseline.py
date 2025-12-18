#!/usr/bin/env python3
"""CLI for training baseline predictor models."""

from __future__ import annotations

import argparse
from pathlib import Path

from predictor.config import PredictorConfig
from predictor.pipelines import train_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline predictor models")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Project root for data artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PredictorConfig.from_env(args.base_dir)
    artifact = train_baseline.run(config)
    print(f"Saved hurdle model to {artifact}")


if __name__ == "__main__":
    main()
