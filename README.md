# ecommerce-search-engine

Experimentation over an ecommerce search project using the WANDS dataset.

## Project Framing

This repository is framed with a **core retrieval-improvement track** (implement and demonstrate higher MAP@10), while also including substantial **engineering refactor** work:

- Object-oriented retrieval architecture (`SearchEngine` abstraction + interchangeable engines)
- Modular data loading, preprocessing, and evaluation components
- Logging/error-handling oriented structure for production-readiness

The notebook `notebooks/ecommerce_retrieval_project.ipynb` contains the full narrative, experiments, and reproducible results.

## Repository Layout

- `notebooks/ecommerce_retrieval_project.ipynb` — full project narrative and experiments
- `src/ecom_search/` — modular retrieval package (`data_loader`, `preprocessing`, `search_engine`, `evaluation`, `query_processor`)
- `scripts/` — lightweight entrypoints for running baseline/evaluation from the terminal

## Quick Start

```bash
uv sync
uv run python scripts/run_baseline.py
```
