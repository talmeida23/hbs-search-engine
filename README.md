# hbs-search-engine

Experimentation over the HBS search assignment using the WANDS dataset.

## Assignment Positioning

This repository is framed as **Option A primary** (implement and demonstrate retrieval improvements with higher MAP@10), while also including substantial **Option B** work:

- Object-oriented retrieval architecture (`SearchEngine` abstraction + interchangeable engines)
- Modular data loading, preprocessing, and evaluation components
- Logging/error-handling oriented structure for production-readiness

The notebook `notebooks/HBS_retrieval_assignment.ipynb` contains the full narrative, experiments, and reproducible results.

## Repository Layout

- `notebooks/HBS_retrieval_assignment.ipynb` — full assignment narrative and experiments
- `src/hbs_search/` — modular retrieval package (`data_loader`, `preprocessing`, `search_engine`, `evaluation`, `query_processor`)
- `scripts/` — lightweight entrypoints for running baseline/evaluation from the terminal

## Quick Start

```bash
uv sync
uv run python scripts/run_baseline.py
```
