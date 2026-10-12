-- How much does snap share normally move from one game to the next, by position?
-- This is the "noise" the shift score divides by. LAG() looks one game back.
WITH changes AS (
  SELECT
    position,
    snap_pct - LAG(snap_pct) OVER (PARTITION BY season, pfr_player_id ORDER BY week) AS change
  FROM rosterwatch.weekly_usage
  WHERE season < 2026
)
SELECT
  position,
  COUNT(change)                 AS game_pairs,
  ROUND(AVG(change), 4)         AS avg_change,
  ROUND(STDDEV(change), 4)      AS sd_change,
  ROUND(AVG(ABS(change)), 4)    AS avg_abs_change
FROM changes
WHERE change IS NOT NULL
GROUP BY position
ORDER BY position;
