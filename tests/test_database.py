"""Tests del wrapper de SQLite — operaciones básicas y configuración WAL."""
import os
import sqlite3
import tempfile

import pandas as pd
import pytest

from database import TradingDatabase


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = TradingDatabase(db_path=path, auto_cleanup_snapshots=False)
    yield db
    db.close()
    # Limpieza de WAL/SHM si quedaron
    for ext in ("", "-wal", "-shm"):
        try:
            os.remove(path + ext)
        except OSError:
            pass


def test_creates_tables(tmp_db):
    conn = tmp_db.connect()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    expected = {"precios", "señales", "operaciones", "config",
                "journal_trades", "signal_snapshots"}
    assert expected.issubset(tables)


def test_wal_enabled(tmp_db):
    conn = tmp_db.connect()
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_synchronous_normal(tmp_db):
    conn = tmp_db.connect()
    sync = conn.execute("PRAGMA synchronous").fetchone()[0]
    # NORMAL == 1
    assert sync == 1


def test_signal_snapshot_roundtrip(tmp_db):
    df = pd.DataFrame([
        {"symbol": "AAPL", "confidence": "ALTA", "confidence_score": 4.0,
         "vol_signal": "COMPRAR", "rs_signal": "LIDER", "risk_signal": "BAJO RIESGO",
         "total_score": 75.0},
        {"symbol": "MSFT", "confidence": "MEDIA", "confidence_score": 2.0,
         "vol_signal": "VENDER", "rs_signal": "MERCADO", "risk_signal": "RIESGO MEDIO",
         "total_score": 55.0},
    ])
    tmp_db.save_signal_snapshot(df, run_timestamp="2026-05-06T10:00:00")
    tmp_db.save_signal_snapshot(df, run_timestamp="2026-05-06T11:00:00")

    last = tmp_db.get_last_snapshot()
    assert last is not None
    assert set(last["symbol"]) == {"AAPL", "MSFT"}


def test_cleanup_keeps_latest(tmp_db):
    df = pd.DataFrame([{"symbol": "AAPL", "confidence": "ALTA",
                        "confidence_score": 1.0, "vol_signal": "COMPRAR",
                        "rs_signal": "LIDER", "risk_signal": "BAJO RIESGO",
                        "total_score": 70.0}])
    for i in range(5):
        tmp_db.save_signal_snapshot(df, run_timestamp=f"2026-05-0{i+1}T10:00:00")

    tmp_db.cleanup_old_snapshots(keep=2)

    conn = tmp_db.connect()
    runs = conn.execute(
        "SELECT DISTINCT run_timestamp FROM signal_snapshots ORDER BY run_timestamp DESC"
    ).fetchall()
    assert len(runs) == 2
    # Los dos más recientes deben ser 2026-05-05 y 2026-05-04
    kept = {r[0] for r in runs}
    assert "2026-05-05T10:00:00" in kept
    assert "2026-05-04T10:00:00" in kept


def test_journal_open_and_get_open(tmp_db):
    tmp_db.journal_open(
        symbol="AAPL", entry_date="2026-05-01", entry_price=180.0,
        quantity=10.0, amount_eur=1800.0, stop_loss=170.0, notes="test"
    )
    open_df = tmp_db.journal_get_open()
    assert len(open_df) == 1
    assert open_df.iloc[0]["symbol"] == "AAPL"
    assert open_df.iloc[0]["status"] == "OPEN"
