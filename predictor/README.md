# Predictor module

`predictor/` hosts every predictive workflow that feeds the knapsack optimizers. It owns
feature definition, model training, batch/online scoring, and schema contracts that the
solver consumes.

## Directory layout

```
predictor/
  pyproject.toml        # Python environment definition (uv/poetry compatible)
  src/predictor/        # Library code
  scripts/              # Thin CLI wrappers around pipelines
  tests/                # pytest suites + fixtures
```

Important subpackages:

* `predictor.features`: deterministic feature builders backed by declarative configs.
* `predictor.models`: gradient-boosted trees, survival models, hurdle composition.
* `predictor.pipelines`: training + batch export DAGs and orchestration helpers.
* `predictor.serve`: FastAPI app that exposes online scoring.
* `predictor.io`: Arrow schema validation + Parquet writers.

## Quick start

```bash
cd predictor
uv sync  # or: PYTHON=python3.11 make predictor-setup
make -C .. predictor-setup  # optional convenience target defined at repo root
```

> **Python requirement**: CatBoost wheels are only published for CPython <= 3.12, so create your
> virtualenv with `python3.11` (or `3.12.x`). Running `make predictor-setup` automatically uses the
> interpreter pointed to by the `PYTHON` variable (defaults to `python3.11`).

Available CLI entrypoints (all under `predictor/scripts`):

* `train_baseline.py` – trains the baseline gradient-boosted + hurdle models.
* `export_scores.py` – materializes scored Arrow/Parquet datasets for the solver.
* `serve_api.py` – launches the FastAPI service for realtime inference.

Each CLI accepts `--config` pointing to a YAML/JSON config in `configs/` (to be added as
use cases sharpen). Default configs produce synthetic data so the solver can be wired up
before real datasets are available.

## Contracts

* Batch output schema lives at `shared/schemas/predictor/item_scores.schema.json`.
* Online scoring proto: `shared/schemas/predictor/score.proto`.
* Schema compatibility is enforced in `predictor/tests/test_schema_contract.py`.

## CI expectations

1. Lint + formatting: `ruff check predictor` and `ruff format --check predictor`.
2. Unit tests: `pytest predictor/tests`.
3. Contract smoke: run `python predictor/scripts/export_scores.py --as-of today` and
   ensure `knapsack` integration tests ingest the resulting Arrow file.

Keep dependencies Python-only; CUDA/Metal requirements remain inside the solver tree.
