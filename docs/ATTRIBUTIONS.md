# Third-Party Notices and Attribution

This document lists the upstream components bundled with or required by the knapsack project and describes their corresponding licenses. The core solver and bindings are distributed under the MIT License (see `LICENSE`).

## Vendored C++ Components

| Component | Version / Snapshot | Location | Purpose in repo | License |
| --- | --- | --- | --- | --- |
| Catch2 | v3.5.0 (amalgamated release generated 2023-12-11) | `third_party/catch2/` | Unit and benchmark test framework for the C++ solver and Metal backends | Boost Software License 1.0 (see header notice linking to https://www.boost.org/LICENSE_1_0.txt) |
| pybind11 | v3.0.2a0 (per `PYBIND11_VERSION_*` in `include/pybind11/detail/common.h`) | `third_party/pybind11/` | C++17 ↔ Python binding generator used by `setup.py` | BSD 3-Clause License (`third_party/pybind11/LICENSE`) |
| picojson | Header snapshot (copyright 2009–2014, single-file dist) | `third_party/picojson/` | Lightweight JSON parsing inside solver utilities | Simplified BSD License (2-Clause) embedded at the top of `third_party/picojson/picojson.h` |

> When updating a vendored dependency, copy in the upstream LICENSE (if not already present) and update this table with the new version/build date.

## Python Dependencies (`predictor/pyproject.toml`)

The predictor service relies on the following Python packages. License designations reflect upstream project statements as of April 2025; verify with `pip-licenses` before shipping.

| Package | Purpose | Declared license |
| --- | --- | --- |
| numpy | Numerical kernels backing feature engineering and batching | BSD 3-Clause |
| pandas | Tabular data transforms and dataset ingestion | BSD 3-Clause |
| polars | Columnar data pipelines for large benchmark conversions | MIT |
| pyarrow | Arrow/Parquet serialization for dataset interchange | Apache License 2.0 |
| pydantic | Data validation for FastAPI endpoints and configs | MIT |
| scikit-learn | Classical ML models for candidate scoring baselines | BSD 3-Clause |
| xgboost | Gradient boosted tree models used in ranking pipelines | Apache License 2.0 |
| lightgbm | GBM alternative for latency-sensitive runs | MIT |
| catboost | Gradient boosting for categorical features | Apache License 2.0 |
| lifelines | Survival analysis helpers for retention modeling | MIT |
| pycox | Deep survival modeling experiments | MIT |
| mlflow | Experiment tracking for model sweeps | Apache License 2.0 |
| fastapi | Serving layer for the predictor API | MIT |
| uvicorn | ASGI server for FastAPI | BSD 3-Clause |
| prefect | Workflow orchestration jobs | Apache License 2.0 |
| click | CLI ergonomics for local tools | BSD 3-Clause |

### Development extras

| Package | Role | Declared license |
| --- | --- | --- |
| pytest | Core test runner | MIT |
| pytest-cov | Coverage reporting | MIT |
| ruff | Linting | MIT |
| mypy | Type checking | MIT |

### Top-level Python extension build (`setup.py`)

The root `setup.py` exposes a `knapsack_py` extension built with CMake. Its `extras_require['dev']` block pulls in `pytest` and `numpy`, which are already documented above; no other runtime Python packages are bundled with the solver wheel.

## How to Keep This Page Current

1. **Vendored code**: whenever adding to `third_party/`, copy upstream notices verbatim, document the source commit/tag, and reference the correct license above.
2. **Python packages**: run `pip-licenses --format=markdown` inside `predictor` and update the tables when versions or licenses change.
3. **Other assets**: if you integrate different datasets, fonts, or binaries, add a new subsection covering their provenance and usage limits.

Maintaining this page ensures that MIT-licensed distributions of the knapsack solver remain compliant with all incorporated dependencies.
