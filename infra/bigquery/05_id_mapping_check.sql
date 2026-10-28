-- Player ID mapping check: how each player was matched, and which ids he has.
-- pfr_id = matched on his Pro Football Reference id (reliable)
-- name_match = matched on unique name + position (check by hand)
-- unmatched = no ids: no stats, % rostered or injury status for him
SELECT
  id_source,
  COUNT(*)                                                    AS players,
  SUM(games)                                                  AS player_games,
  SUM(CASE WHEN gsis_id IS NOT NULL THEN 1 ELSE 0 END)        AS with_gsis_id,
  SUM(CASE WHEN espn_id IS NOT NULL THEN 1 ELSE 0 END)        AS with_espn_id,
  SUM(CASE WHEN sleeper_id IS NOT NULL THEN 1 ELSE 0 END)     AS with_sleeper_id
FROM rosterwatch.player_ids
GROUP BY id_source
ORDER BY players DESC;
