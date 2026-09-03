# Repository Guidelines

## Project Structure & Module Organization

MitigV is a Python (>=3.10) library using a `src/` layout:

- `src/mitigv/core/` contains base mitigator classes, registries, and structural model/processor interfaces.
- `src/mitigv/algorithms/` contains decoding and steering implementations (VCD, ICD, PAI, M3ID, VISTA, AGLA, ONLY, OPERA, and probe steering).
- `src/mitigv/backends/` contains the framework-neutral `GenericMitigator` plus shared vision-language adapters and model-family adapters (`llava.py`, `qwen2_5_vl.py`). Select families through `backends.factory`.
- `src/mitigv/evaluation/` contains strict CHAIR, POPE/AMBER, DeepSeek + GroundingDINO, length-control analysis, CLI modules, and bundled prompts/data.
- `tests/` mirrors these modules with focused pytest coverage.

## Build, Test, and Development Commands

Create an editable development install with evaluation and test dependencies:

```bash
python -m pip install -e ".[test,eval]"
```

Run the full suite and lint checks:

```bash
python -m pytest -q
ruff check .
git diff --check
```

Use the installed evaluator entry points directly (for example, `mitigv-discriminative ...`). COCO/POPE/AMBER data are expected under `~/dataset`; do not download or commit datasets, model weights, caches, or generated `results/` files.

## Coding Style & Naming Conventions

Use four-space indentation, type annotations, and concise docstrings. Follow Ruff's configured defaults; keep imports sorted and lines readable. Use `snake_case` for functions and modules, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. New model integrations should implement the shared interfaces and expose a parameter-selectable factory path rather than duplicating decoding logic.

## Testing Guidelines

Tests use pytest and are named `tests/test_<module>.py`; test functions and methods start with `test_`. Add regression tests for every behavior change, using lightweight fakes when Torch/Transformers behavior can be isolated. Run the focused file first, then the full suite; keep tests deterministic with explicit seeds.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commit style, e.g. `feat(vista): ...`, `fix(opera,beam): ...`, or `test: ...`. Keep commits focused and explain compatibility or API changes in the body. Pull requests should describe the motivation, implementation, and validation commands, link the relevant issue when available, and include benchmark/evaluation details for metric or model changes. Do not include secrets; configure `DEEPSEEK_API_KEY` through the environment for judge runs.
