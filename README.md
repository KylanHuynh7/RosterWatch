# RosterWatch

A UC San Diego DS3 project, built on the DS3 project template. RosterWatch flags NFL skill players whose **real usage** (snap share, share of team targets + carries) is shifting:

| Signal | Who | Rule (default) |
|---|---|---|
| **Waiver Riser** | < 50% rostered (ESPN) | shift score ≥ +2.0 SD and now playing ≥ 35% of snaps |
| **Fragile Starter** | ≥ 50% rostered | shift ≤ −2.0 SD, or a same-position teammate surging, or very volatile snaps *(experimental)* |
| **Injury Check** | ≥ 50% rostered | latest game's snap share < half his usual → probably got hurt mid-game, verify |
| **Slow Bleed** *(Phase 2)* | ≥ 50% rostered | CUSUM bleed tally ≤ −3 (usage keeps landing below his established level) |
| **Slow Climb** *(Phase 2)* | < 50% rostered | CUSUM climb tally ≥ +3 |

Every flag has a plain-English `reason`, plus `context` that says whether a same-position teammate lost snaps or missed the game (possible fill-in work).

## Quick start

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the example environment file (`.env` is never committed):

```bash
cp .env.example .env
```

4. Run the pipeline from the repo root. It writes the dashboard CSVs and `results/brief.html`:

```bash
python -m src.pipeline                   # default: shift threshold 2.0 SD, CUSUM +/-3
python -m src.pipeline --threshold 1.5   # more flags, lower hit rate
```

5. (Optional) Upload to BigQuery. Setup steps are in `infra/README.md`:

```bash
python -m src.warehouse --dry-run
```

6. Run the tests:

```bash
pytest
```

The first run downloads about 5 seasons of data (around a minute) and caches it in `data/`. Delete the files in `data/processed/` and `data/raw/` (not `data/raw/rostership/`) to pull fresh weekly data.

## Repository structure

```
.
├── README.md
├── .env.example            # settings + GCP placeholders; copy to .env
├── .gitignore
├── requirements.txt
├── notebooks/              # exploration only, e.g. 01_houston_case_study.ipynb
├── src/
│   ├── __init__.py
│   ├── example.py          # DS3 template example
│   ├── config.py           # paths + settings (reads .env)
│   ├── data.py             # nflverse, Sleeper, ESPN loaders
│   ├── signals.py          # Phase 1: shift score + flag rules
│   ├── trend.py            # Phase 2: CUSUM slow bleed / climb
│   ├── backtest.py         # threshold evaluation on 2022-2025
│   ├── warehouse.py        # uploads tables to BigQuery
│   ├── pipeline.py         # runs everything -> results/
│   └── brief.py            # fills frontend/brief_template.html
├── frontend/               # brief template + Looker Studio plan
├── infra/                  # BigQuery setup + starter SQL (infra/bigquery/)
├── docs/
│   ├── AGENTS.md           # instructions for AI coding tools working in this repo
│   ├── codeReviewer.md     # DS3 code-review checklist (run before opening a PR)
│   ├── DesignDoc.md        # design document
│   └── proposed_issues.md  # starting backlog for GitHub Issues
├── data/
│   ├── raw/                # API pulls (rostership/ snapshots are committed)
│   └── processed/          # weekly usage table
├── results/                # dashboard CSVs + brief.html (generated, not committed)
├── models/                 # fitted noise calibrations (JSON, generated, not committed)
└── tests/                  # pytest on synthetic players
```

## The method in plain terms

1. **Usage per game:** `snap_pct` (from PFR snap counts) and `opp_share = (targets + carries) / team (targets + carries)`. Games the player didn't play are *missing*, not zero.
2. **Shift:** the average of the last 2 games minus the average of his earlier games this season (at least 2).
3. **Shift score:** the shift divided by the *typical* size of a shift for that position, measured on 2022–2025. The two metrics' scores are averaged and rescaled so the result reads as "standard deviations of normal week-to-week noise."
4. **Threshold from a backtest:** the same calculation is run on every player-game in 2022–2025, and we check the next 3 games (`results/backtest_thresholds.csv`):

| Threshold | Flags/week (league-wide) | Stayed above old baseline | Kept ≥ half the gain |
|---|---|---|---|
| no shift (reference) | – | 47% | 47% |
| ≥ 1.0 SD | 36.6 | 81% | 53% |
| ≥ 1.5 SD | 18.8 | 86% | 57% |
| **≥ 2.0 SD** | **9.1** | **90%** | **60%** |
| ≥ 2.5 SD | 4.6 | 93% | 63% |

Persistence slope ≈ 0.59: on average, about 60% of a 2-game usage shift is still there over the next 3 games.

### Phase 2: slow bleed (CUSUM, `trend.py`)

The shift score misses slow declines. CUSUM handles them:
1. **Established level:** the average of the player's first 3 games (snap share, opportunity share, passing-down share).
2. **Each game after that:** how far it landed from the established level, in noise units, averaged over the 3 metrics and capped at ±3 SD so one injury game can't dominate.
3. **Bleed tally:** `min(0, tally + z + 0.5)`. Games within 0.5 SD of normal are ignored, and small misses accumulate. The climb tally mirrors it.

Backtest (`results/backtest_cusum.csv`): a first crossing of −3 is followed by snap share more than 5 points below the established level 81% of the time, versus 30% with no signal. A climb past +3 is followed by a rise 91% of the time, versus 35%. Case studies (`results/case_studies.csv`): in 2025, Woody Marks's climb flag fired in Week 7 and Nick Chubb's bleed flag in Week 13. Jordan Mason's bleed flag fired in Week 11.

**Passing-down share** = share of the team's targets + carries on 3rd/4th down or in the last 2 minutes of a half, computed from play-by-play. **Red-zone share** is computed the same way for plays inside the opponent's 20.

## Code

| File | What |
|---|---|
| `src/data.py` | Loads nflverse data (`nflreadpy`, including play-by-play situational shares), Sleeper (depth chart, injury status, trending adds/drops) and ESPN % rostered |
| `src/signals.py` | Windows, noise model, shift scores, flag rules |
| `src/backtest.py` | Threshold sweep + fill-in vs. earned comparison |
| `src/trend.py` | Phase 2: CUSUM slow bleed/climb, its backtest, case studies (+ an earlier 5-game-slope attempt kept for comparison) |
| `src/pipeline.py` | Runs everything, writes `results/*.csv`, `models/*.json`, `results/brief.html` |
| `src/brief.py` | Rebuilds `results/brief.html` from `results/` |
| `src/warehouse.py` | Uploads all tables to BigQuery (`--dry-run` to preview) |

### Outputs in `results/` (Looker Studio-ready)

- `signals_current.csv`: this week's flags with reasons → the main dashboard table
- `shift_scores_current.csv`: every player's scores → scatter (shift score vs. % rostered)
- `usage_trends_current.csv`: player-game usage → line charts per player
- `backtest_thresholds.csv`, `backtest_fill_in.csv`: methodology page
- `data/raw/rostership/rostership_snapshot_<date>.csv`: ESPN only exposes *today's* % rostered, so every pull is kept and committed

To get these into Looker Studio: upload the CSVs to Google Sheets (one tab each), then in Looker Studio choose Add data → Google Sheets. Later, a scheduled script can push to Sheets or BigQuery automatically.

## Known limitations (good discussion items)

- **Small samples early in the season:** at week 4 the baseline is just 2 games.
- **2-game vs. 3-game recent window:** 3 games is more reliable (slope 0.69 vs. 0.59) but flags a week later.
- **Noise is pooled per position:** a 90% snap player and a 30% snap player shouldn't have the same noise scale. A binomial or level-dependent model would fix this.
- **Injury handling is heuristic:** the "sudden drop" and "absent teammate" rules come from snap data only. Joining `nflreadpy.load_injuries()` would be more precise.
- **Volatility flag isn't backtested yet.**
- **Rostership:** Sleeper (our planned source) has no "% rostered", only top-100 trending adds/drops. ESPN's undocumented API fills the gap for now. Decide with the mentor.
- **CUSUM is quiet early in the season:** it needs 3 games to set the established level. Using last season as a prior would fix that.
- **Per-play on-field data** (`load_participation`) only covers 2016–2025, so live 2026 passing-down usage comes from targets and carries, not snaps.

## Repository guidelines

* Keep notebooks for exploration and visualization.
* Put reusable logic in `src` so notebooks stay readable.
* Do not commit secrets, API keys, or large data files.
* Use clear filenames and short, documented functions.
* Add tests for any core logic that will be reused.

## Git and GitHub workflow

1. **Issues:** one per task, bug, experiment or feature, with a short title, a clear description, acceptance criteria and labels (`bug`, `enhancement`, `documentation`, `blocked`). A starting backlog is in `docs/proposed_issues.md`.
2. **Branches:** one per person or feature, such as `feature/bigquery-load`.
3. **Pull requests:** open a PR from your branch into `main`. **Only project managers or mentors approve PRs.**

```bash
git checkout main
git pull origin main
git checkout -b feature/my-task
git add .
git commit -m "Describe your change"
git push origin feature/my-task
```

Then open a Pull Request on GitHub and wait for approval from the project manager or mentor.

Team habits: pull before starting work, keep commits small and descriptive, open PRs early for feedback, and resolve merge conflicts carefully and with the team.
