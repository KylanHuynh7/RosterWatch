"""Build results/brief.html (the one-page summary) from the CSVs in results/.

Runs automatically at the end of src.pipeline; or on its own: python -m src.brief
"""
import json

import pandas as pd

from .config import FRONTEND, RESULTS as OUT, ROSTERSHIP_SNAPSHOTS


def _flag_rows():
    flags = pd.read_csv(OUT / "signals_current.csv")
    usage = pd.read_csv(OUT / "usage_trends_current.csv")
    rows = []
    for _, r in flags.iterrows():
        g = usage[(usage.player == r.player) & (usage.team == r.team)].sort_values("week")
        rec = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
        rec = {k: (int(v) if hasattr(v, "item") and float(v).is_integer() else v) for k, v in rec.items()}
        rows.append({**rec, "weeks": g.week.tolist(), "snap": g.snap_pct.round(3).tolist(),
                     "opp": g.opp_share.round(3).tolist()})
    return rows


def build():
    snapshot = sorted(ROSTERSHIP_SNAPSHOTS.glob("rostership_snapshot_*.csv"))[-1].stem.split("_")[-1]
    data = {"flags": _flag_rows(),
            "backtest": pd.read_csv(OUT / "backtest_thresholds.csv").to_dict("records"),
            "fill_in": pd.read_csv(OUT / "backtest_fill_in.csv").to_dict("records"),
            "cusum_bt": pd.read_csv(OUT / "backtest_cusum.csv").to_dict("records"),
            "cases": pd.read_csv(OUT / "case_studies.csv")[["team", "player", "week", "snap_pct"]].to_dict("records"),
            "case_flags": pd.read_csv(OUT / "case_study_flags.csv").astype(object)
                            .where(lambda d: d.notna(), None).to_dict("records"),
            "n_players": int(len(pd.read_csv(OUT / "shift_scores_current.csv")))}
    html = (FRONTEND / "brief_template.html").read_text()
    html = html.replace("__DATA__", json.dumps(data)).replace("__DATE__", snapshot)
    out = OUT / "brief.html"
    out.write_text(html)
    return out


if __name__ == "__main__":
    print(f"wrote {build()}")
