-- Rolling 3-game snap share and volatility for one player.
-- ROWS BETWEEN 2 PRECEDING AND CURRENT ROW = "this game and the 2 before it".
-- PARTITION BY keeps each player-season separate, so windows never mix players.
SELECT
  season,
  week,
  player,
  team,
  ROUND(snap_pct, 2) AS snap_pct,
  ROUND(AVG(snap_pct) OVER w, 3)    AS rolling_avg_3,
  ROUND(STDDEV(snap_pct) OVER w, 3) AS rolling_sd_3,
  COUNT(*) OVER w                   AS games_in_window
FROM rosterwatch.weekly_usage
WHERE player = 'Nick Chubb' AND season = 2025
WINDOW w AS (PARTITION BY season, pfr_player_id ORDER BY week ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
ORDER BY week;
