"""BigQuery loader tests with a fake client (no Google account needed). Run: pytest"""
import pandas as pd

from src import warehouse


class FakeJob:
    def result(self):
        return self


class FakeClient:
    project = "test-project"

    def __init__(self):
        self.datasets, self.loads = [], []

    def create_dataset(self, ds, exists_ok=False):
        self.datasets.append(ds.dataset_id)

    def load_table_from_dataframe(self, df, table_id, job_config=None):
        self.loads.append((table_id, len(df), job_config.write_disposition))
        return FakeJob()


def test_upload_replaces_every_table_in_the_dataset():
    client = FakeClient()
    tables = {"weekly_usage": pd.DataFrame({"a": [1, 2]}), "signals_current": pd.DataFrame({"b": [3]})}
    warehouse.upload(tables, client, dataset="rw_test")
    assert client.datasets == ["rw_test"]
    assert client.loads == [("test-project.rw_test.weekly_usage", 2, "WRITE_TRUNCATE"),
                            ("test-project.rw_test.signals_current", 1, "WRITE_TRUNCATE")]


def test_collect_tables_rebuilds_rostership_history_from_all_snapshots(tmp_path, monkeypatch):
    for name in ("PROCESSED", "RESULTS", "RAW"):
        monkeypatch.setattr(warehouse.config, name, tmp_path / name.lower())
        (tmp_path / name.lower()).mkdir()
    snaps = tmp_path / "raw" / "rostership"
    snaps.mkdir()
    monkeypatch.setattr(warehouse.config, "ROSTERSHIP_SNAPSHOTS", snaps)
    for day in ("2026-10-07", "2026-10-14"):
        pd.DataFrame({"espn_id": [1.0], "pct_rostered": [50.0], "snapshot_date": [day]}) \
            .to_csv(snaps / f"rostership_snapshot_{day}.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "results" / "signals_current.csv", index=False)

    tables = warehouse.collect_tables()
    assert set(tables) == {"signals_current", "rostership_snapshots"}
    assert len(tables["rostership_snapshots"]) == 2   # both weeks kept, nothing lost
