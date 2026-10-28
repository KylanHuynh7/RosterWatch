# RosterWatch

A UC San Diego DS3 project, built on the DS3 project template. RosterWatch flags NFL skill players (RB, WR, TE) whose **real usage** (snap share, share of team targets + carries) is shifting, and backs every flag with transparent, backtested statistics.

## What this repo includes

- The DS3 template's folder structure, Python environment file, `.gitignore`, and starter package with a test example
- A weekly data pipeline: nflverse (snaps, stats, play-by-play), Sleeper (depth chart, injuries, trending adds/drops) and ESPN (% rostered)
- Two detectors: a **shift score** for sudden usage changes (Phase 1) and a **CUSUM** tally for slow multi-week bleeds and climbs (Phase 2)
- Backtests on 2022–2025 that justify each flag threshold
- Dashboard-ready outputs, a BigQuery loader, starter SQL queries, and tests

## Repository structure

```text
.
├── README.md
├── .env.example            # settings + GCP placeholders; copy to .env
├── .gitignore
├── requirements.txt
├── notebooks/
│   ├── README.md
│   └── 01_houston_case_study.ipynb   # Chubb vs. Marks, 2025
├── src/
│   ├── __init__.py
│   ├── example.py          # DS3 template example
│   ├── config.py           # paths + settings (reads .env)
│   ├── data.py             # nflverse, Sleeper and ESPN loaders; clean tables; player ID mapping
│   ├── signals.py          # Phase 1: shift score + flag rules
│   ├── trend.py            # Phase 2: CUSUM slow bleed / climb
│   ├── backtest.py         # threshold evaluation on 2022-2025
│   ├── warehouse.py        # uploads tables to BigQuery
│   ├── quality.py          # player ID match checks
│   ├── pipeline.py         # runs everything -> results/
│   └── brief.py            # fills frontend/brief_template.html
├── frontend/
│   ├── README.md           # dashboard (Looker Studio / Data Studio) plan
│   └── brief_template.html # one-page summary template
├── infra/
│   ├── README.md           # BigQuery setup
│   └── bigquery/           # starter SQL queries
├── docs/
│   ├── AGENTS.md           # instructions for AI coding tools working in this repo
│   ├── codeReviewer.md     # DS3 code-review checklist (run before opening a PR)
│   ├── DesignDoc.md        # design document
│   └── proposed_issues.md  # starting backlog for GitHub Issues
├── data/
│   ├── raw/                # API pulls; rostership/ snapshots are committed
│   └── processed/          # weekly usage table (generated)
├── results/                # dashboard CSVs + brief.html (generated, not committed)
├── models/                 # fitted noise calibrations (generated, not committed)
└── tests/                  # pytest on synthetic players
```

- `frontend/` is for UI/UX code and web app assets.
- `infra/` is for deployment scripts, cloud config, and infrastructure as code for AWS/GCP.

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
3. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   **.env files are generally not shared on GitHub; they are included here only as an example template.**
4. Run the pipeline from the repo root. It writes the dashboard CSVs and `results/brief.html`:
   ```bash
   python -m src.pipeline                   # default: shift threshold 2.0 SD, CUSUM +/-3
   python -m src.pipeline --threshold 1.5   # more flags, lower hit rate
   python -m src.pipeline --bigquery        # also upload to BigQuery (setup: infra/README.md)
   ```
5. Run the tests:
   ```bash
   pytest
   ```

The first run downloads about 5 seasons of data (around a minute) and caches it in `data/`. To pull fresh weekly data, delete the files in `data/processed/` and rerun. Keep `data/raw/rostership/`: past snapshots can't be downloaded again.

## Signals

| Signal | Who | Rule (default) |
|---|---|---|
| **Waiver Riser** | < 50% rostered (ESPN) | shift score ≥ +2.0 SD and now playing ≥ 35% of snaps |
| **Fragile Starter** | ≥ 50% rostered | shift ≤ −2.0 SD, or a same-position teammate surging, or very volatile snaps *(experimental)* |
| **Injury Check** | ≥ 50% rostered | latest game's snap share < half his usual → probably got hurt mid-game, verify |
| **Slow Bleed** *(Phase 2)* | ≥ 50% rostered | CUSUM bleed tally ≤ −3 (usage keeps landing below his established level) |
| **Slow Climb** *(Phase 2)* | < 50% rostered | CUSUM climb tally ≥ +3 |

Every flag has a plain-English `reason`, plus `context` that says whether a same-position teammate lost snaps or missed the game (possible fill-in work).

## The method in plain terms

1. **Usage per game:** `snap_pct` (from PFR snap counts) and `opp_share = (targets + carries) / team (targets + carries)`. Games the player didn't play are *missing*, not zero.
2. **Shift:** the average of the last 2 games minus the average of his earlier games this season (at least 2).
3. **Shift score:** the shift divided by the *typical* size of a shift for that position, measured on 2022–2025. The two metrics' scores are averaged and rescaled so the result reads as "standard deviations of normal week-to-week noise."
4. **Threshold from a backtest:** the same calculation is run on every player-game in 2022–2025, and we check the next 3 games (`results/backtest_thresholds.csv`):

| Threshold | Flags/week (league-wide) | Stayed above old baseline | Kept ≥ half the gain |
|---|---|---|---|
| no shift (reference) | – | 46% | 46% |
| ≥ 1.0 SD | 35.9 | 81% | 53% |
| ≥ 1.5 SD | 18.6 | 86% | 57% |
| **≥ 2.0 SD** | **9.1** | **90.5%** | **60%** |
| ≥ 2.5 SD | 4.8 | 92% | 62% |

Persistence slope ≈ 0.61: on average, about 60% of a 2-game usage shift is still there over the next 3 games.

**Injury blips are left out of both averages.** A game where a regular's snap share fell below half his usual level, followed by a bounce-back game, is treated as an in-game injury, not a role change (for example, Ja'Marr Chase in 2026: 94%, 88%, 100%, **20%**, 89%). A drop with no bounce-back stays in, since that may be a real benching. Falling-usage flags at −2 SD now stay down 92.7% of the time (previously 89.0%).

### Phase 2: slow bleed (CUSUM, `trend.py`)

The shift score misses slow declines. CUSUM handles them:
1. **Established level:** the average of the player's first 3 games (snap share, opportunity share, passing-down share).
2. **Each game after that:** how far it landed from the established level, in noise units, averaged over the 3 metrics and capped at ±3 SD so one injury game can't dominate.
3. **Bleed tally:** `min(0, tally + z + 0.5)`. Games within 0.5 SD of normal are ignored, and small misses accumulate. The climb tally mirrors it.

Backtest (`results/backtest_cusum.csv`): a first crossing of −3 is followed by snap share more than 5 points below the established level 81% of the time, versus 30% with no signal. A climb past +3 is followed by a rise 91% of the time, versus 35%. Case studies (`results/case_studies.csv`): in 2025, Woody Marks's climb flag fired in Week 7 and Nick Chubb's bleed flag in Week 13. Jordan Mason's bleed flag fired in Week 11.

**Passing-down share** = share of the team's targets + carries on 3rd/4th down or in the last 2 minutes of a half, computed from play-by-play. **Red-zone share** is computed the same way for plays inside the opponent's 20.

## Outputs in `results/`

- `signals_current.csv`: this week's flags with reasons → the main dashboard table
- `id_mapping_report.csv`, `id_unmatched.csv`: how well player ids link across nflverse, ESPN and Sleeper, and which players still need fixing
- `shift_scores_current.csv`, `cusum_current.csv`: every player's scores
- `usage_trends_current.csv`: player-game usage → line charts per player
- `backtest_thresholds.csv`, `backtest_cusum.csv`, `backtest_fill_in.csv`, `case_studies.csv`: methodology evidence
- `data/raw/rostership/rostership_snapshot_<date>.csv`: ESPN only exposes *today's* % rostered, so every pull is kept and committed

`python -m src.warehouse` uploads all of these to BigQuery, where the dashboard reads them (see `infra/README.md` and `frontend/README.md`).

## Known limitations

- **Small samples early in the season:** at week 4 the baseline is just 2 games.
- **2-game vs. 3-game recent window:** 3 games was more reliable in testing but flags a week later.
- **Noise is pooled per position:** a 90% snap player and a 30% snap player shouldn't have the same noise scale. A binomial or level-dependent model would fix this.
- **Injury handling is heuristic:** the "sudden drop", "bounce-back" and "absent teammate" rules come from snap data. Sleeper injury status is attached to each flag for checking; joining `nflreadpy.load_injuries()` would be more precise.
- **Volatility flag isn't backtested yet.**
- **Rostership:** Sleeper (our planned source) has no "% rostered", only top-100 trending adds/drops. ESPN's undocumented API fills the gap for now. Decide with the mentor.
- **CUSUM is quiet early in the season:** it needs 3 games to set the established level. Using last season as a prior would fix that.
- **Per-play on-field data** (`load_participation`) only covers 2016–2025, so live 2026 passing-down usage comes from targets and carries, not snaps.

## Repository guidelines

- Keep notebooks for exploration and visualization.
- Put reusable logic in src so notebooks stay readable.
- Do not commit secrets, API keys, or large data files.
- Use clear filenames and short, documented functions.
- Add tests for any core logic that will be reused.

## Git and GitHub workflow for DS3 projects

Using Git and GitHub well is just as important as writing good code. A clean workflow makes collaboration easier and helps the project stay organized.

### 1. Use GitHub Issues for tracking work

Create an Issue for each task, bug, experiment, or feature request. Keep each issue focused and include:

- a short title
- a clear description of the work
- acceptance criteria if relevant
- labels such as bug, enhancement, documentation, or blocked

Example workflow:
- one issue for data cleaning
- one issue for model training
- one issue for documentation updates

### 2. Use branches to separate work

Each person or feature should usually work in its own branch. This prevents people from overwriting each other's changes.

Common branch commands:

```bash
git branch
git checkout -b feature/my-task
git checkout main
git branch -d feature/my-task
```

What they mean:
- `git branch` shows existing branches
- `git checkout -b feature/my-task` creates and switches to a new branch
- `git checkout main` switches back to the main branch
- `git branch -d feature/my-task` deletes a branch after it is merged

### 3. Use pull requests to merge code into main

When work is complete, open a Pull Request (PR) from your branch into `main`. A PR is the place to review changes, discuss issues, and confirm everything is ready.

Important rule:
- Only project managers or mentors should approve pull requests.
- Students should contribute code, but approvals should be handled by the designated maintainers.

### 4. Basic Git commands and what they mean

```bash
git clone <repo-url>
git status
git add <file>
git add .
git commit -m "Describe your change"
git push origin <branch-name>
git log
git merge <branch-name>
git rebase <branch-name>
```

What they mean:
- `git clone` copies a repository from GitHub to your local machine
- `git status` shows which files changed and what branch you are on
- `git add` stages files for commit
- `git commit` saves staged changes with a message
- `git push` uploads your local commits to GitHub
- `git log` shows recent commit history
- `git merge` combines another branch into your current branch
- `git rebase` reapplies your commits on top of another branch and can help keep history cleaner

### 5. Recommended team habits

- Pull the latest changes before starting work
- Keep commits small and descriptive
- Avoid committing large data files or secrets
- Open a PR early for feedback, not only at the end
- Resolve merge conflicts carefully and communicate with the team

### 6. Example workflow

```bash
git checkout main
git pull origin main
git checkout -b feature/data-cleaning
git status
git add .
git commit -m "Add initial data cleaning script"
git push origin feature/data-cleaning
```

Then open a Pull Request on GitHub and wait for approval from the appropriate project manager or mentor.

## Notes for DS3 projects

This template is intentionally simple so students can focus on the science instead of setup. As projects grow, teams can add:

- experiment tracking
- model checkpoints
- deployment scripts
- cloud infrastructure (AWS/GCP) in `infra/`
- frontend UI/UX code in `frontend/`
- CI checks
- docs pages
