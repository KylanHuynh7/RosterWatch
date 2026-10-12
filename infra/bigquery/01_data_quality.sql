-- Data-quality checks on weekly_usage. Every count should be 0 (except total_rows).
-- Change `rosterwatch` if your BQ_DATASET is named differently.
SELECT
  COUNT(*)                                                         AS total_rows,
  SUM(CASE WHEN snap_pct NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END)    AS bad_snap_pct,
  SUM(CASE WHEN opp_share NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END)   AS bad_opp_share,
  SUM(CASE WHEN gsis_id IS NULL THEN 1 ELSE 0 END)                 AS missing_gsis_id,
  SUM(CASE WHEN position NOT IN ('RB', 'WR', 'TE') THEN 1 ELSE 0 END) AS wrong_position
FROM rosterwatch.weekly_usage;
