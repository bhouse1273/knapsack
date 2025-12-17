#!/usr/bin/env python3
"""Runs the FastAPI scoring service."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from predictor.config import PredictorConfig
from predictor.serve.app import load_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve predictor FastAPI app")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Project root for data artifacts")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PredictorConfig.from_env(args.base_dir)
    app = load_app(config)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
