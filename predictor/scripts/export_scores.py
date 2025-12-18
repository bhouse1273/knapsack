#!/usr/bin/env python3
"""CLI for exporting predictor scores."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from predictor.config import PredictorConfig
from predictor.pipelines import export_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch export predictor scores")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Project root for data artifacts")
    parser.add_argument("--as-of", type=str, default=datetime.utcnow().date().isoformat(), help="As-of date")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PredictorConfig.from_env(args.base_dir)
    output = export_scores.run(config, as_of=datetime.fromisoformat(args.as_of))
    print(f"Wrote scored dataset to {output}")


if __name__ == "__main__":
    main()
