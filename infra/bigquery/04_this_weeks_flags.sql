-- This week's flags, waiver risers first.
SELECT signal, player, position, team, pct_rostered, ROUND(shift_score, 1) AS shift_score,
       injury_status, reason
FROM rosterwatch.signals_current
ORDER BY CASE signal WHEN 'Waiver Riser' THEN 1 WHEN 'Slow Climb' THEN 2 ELSE 3 END,
         shift_score DESC;
