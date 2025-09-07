
# Thesis Project — Repro

Minimal, code-only repo to reproduce analyses and figures.

## How to run
1. Create a Python env (e.g., conda/mamba) and install your own deps.
2. Run the entry scripts in `scripts/` (see filenames for chapter/task).
3. Outputs are written locally to `results/` and `figures/` (not tracked).

## Repo policy
- Tracked: only `scripts/**/*.py` and top-level `*.py` helpers.
- Ignored: data, logs, caches, CSV/XLSX, figures, LaTeX tables, environments.
