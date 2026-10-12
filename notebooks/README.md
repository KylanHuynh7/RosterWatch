# Notebooks

Exploration and visualization only. Reusable logic belongs in `src/`, and notebooks import it.

| Notebook | What it shows |
|---|---|
| `01_houston_case_study.ipynb` | Chubb vs. Marks, 2025: weekly usage, CUSUM tallies, and when each flag first fired |

Conventions:
- Prefix with a number (`02_...`) so the order is obvious.
- Run `python -m src.pipeline` first so data is cached in `data/`.
- Clear outputs before committing large plots (Kernel → Restart & Clear Output).
