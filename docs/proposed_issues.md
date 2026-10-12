# Proposed GitHub Issues

Starting backlog for the team. One issue each; adjust at the kickoff.

## Phase 1: foundation (guaranteed deliverable)

| Title | Description | Acceptance criteria | Labels |
|---|---|---|---|
| Review prototype pipeline | Read `src/` and run `python -m src.pipeline`; note anything unclear or wrong | Each team member has run it locally; questions filed as issues | documentation |
| Choose the "% rostered" source | Sleeper has no % rostered. Decide between ESPN's number, Sleeper trending adds, or both | Decision recorded in README; `data.py` updated | enhancement, blocked (mentor) |
| Defend the shift-score threshold | Review `backtest_thresholds.csv`; compare 2- vs. 3-game recent windows | One-paragraph written justification of the chosen setting | enhancement |
| Level-dependent noise model | Noise is pooled per position; a 90%-snap player and a 30%-snap player shouldn't share one scale | Backtest hit rate compared before and after | enhancement |
| Backtest the volatility flag | Currently experimental and untested | Either validated with a hit rate or removed | enhancement |
| Set up GCP project + BigQuery | Create the project, give teammates access, run `python -m src.warehouse` (loader already written; see `infra/README.md`) | 12 tables visible in BigQuery; every teammate can query them; no keys in the repo | infra |
| Weekly GitHub Action | Scheduled run of tests + pipeline + BigQuery load | Runs green on a schedule | infra |
| Looker Studio: Phase 1 views | Waiver opportunity, role instability and player trend views | Public link works and every flag shows its trend data | frontend |

## Phase 2: stretch

| Title | Description | Acceptance criteria | Labels |
|---|---|---|---|
| CUSUM from Week 1 | Use last season's role as the prior for the established level | Flags available before Week 6; backtest compared | enhancement |
| Earned vs. inherited roles | Join injury data and label each rise as earned or fill-in | Label on every riser; backtest of what happens when the starter returns | enhancement |
| Does passing-down share lead? | Test whether passing-down share drops before total snaps, across all backfields | Lead time in weeks with a confidence interval | enhancement |
| Volume vs. one big play | Share of a player's points from his single biggest play | Shown next to each flag; long-TD-only cases filtered | enhancement |
| Market lag study | Weekly Sleeper add/drop snapshots vs. our flag dates | Median weeks of lead time | enhancement |
| Grounded Gemini explanations | Rephrase computed `reason` text with Vertex AI | Every number in the output matches the input | enhancement, frontend |
