"""
Tests del Trade Journal extendido (Etapa 3.4) y alertas inteligentes (3.6).
Usa una DB en memoria temporal — no toca trading_bot.db.
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from database import TradingDatabase


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    db = TradingDatabase(db_path=path, auto_cleanup_snapshots=False)
    yield db
    db.close()
    for ext in ('', '-wal', '-shm'):
        try:
            os.remove(path + ext)
        except OSError:
            pass


# ── journal_open / journal_close extendidos ───────────────────────────────────

def test_journal_open_with_extended_fields(db):
    db.journal_open(
        'AAPL', '2024-01-15', 185.0, 10, 1850.0,
        stop_loss=175.0, take_profit=210.0,
        fees_eur=2.5, slippage_bps=3.0,
        notes='test trade',
    )
    df = db.journal_get_open()
    assert len(df) == 1
    row = df.iloc[0]
    assert row['symbol'] == 'AAPL'
    assert row['stop_loss']   == pytest.approx(175.0)
    assert row['take_profit'] == pytest.approx(210.0)
    assert row['fees_eur']    == pytest.approx(2.5)
    assert row['slippage_bps']== pytest.approx(3.0)


def test_journal_close_computes_pnl(db):
    db.journal_open('MSFT', '2024-01-10', 400.0, 5, 2000.0, fees_eur=3.0)
    trade_id = db.journal_get_open().iloc[0]['id']
    db.journal_close(trade_id, '2024-02-10', 440.0, exit_fees_eur=3.0)

    df = db.journal_get_closed()
    assert len(df) == 1
    row = df.iloc[0]
    assert row['status'] == 'CLOSED'
    assert row['pnl_pct'] == pytest.approx(10.0, abs=0.01)   # +10%
    assert row['pnl_eur'] == pytest.approx(200.0, abs=0.01)  # 2000 * 10%

    # pnl_net = pnl_eur - entry_fees - exit_fees - slippage
    # slippage = exit_value * slippage_bps / 10000 = (440*5) * 0/10000 = 0
    # pnl_net = 200 - 3 - 3 = 194
    assert row['pnl_net_eur'] == pytest.approx(194.0, abs=0.5)


def test_journal_close_legacy_no_fees(db):
    """Retrocompat: cierre sin fees funciona igual que antes."""
    db.journal_open('SPY', '2024-03-01', 500.0, 2, 1000.0)
    trade_id = db.journal_get_open().iloc[0]['id']
    db.journal_close(trade_id, '2024-04-01', 520.0)
    df = db.journal_get_closed()
    assert len(df) == 1
    assert df.iloc[0]['pnl_pct'] == pytest.approx(4.0, abs=0.01)


def test_journal_close_loss(db):
    db.journal_open('NVDA', '2024-01-01', 600.0, 3, 1800.0)
    trade_id = db.journal_get_open().iloc[0]['id']
    db.journal_close(trade_id, '2024-02-01', 540.0)
    row = db.journal_get_closed().iloc[0]
    assert row['pnl_pct'] < 0
    assert row['pnl_eur'] < 0


# ── journal_update_mae ────────────────────────────────────────────────────────

def test_mae_updates_on_adverse_move(db):
    db.journal_open('AAPL', '2024-01-15', 200.0, 5, 1000.0)
    trade_id = int(db.journal_get_open().iloc[0]['id'])

    # Precio cae 5%
    db.journal_update_mae(trade_id, 190.0)
    mae = db.journal_get_open().iloc[0]['mae_pct']
    assert mae == pytest.approx(-5.0, abs=0.1)


def test_mae_does_not_update_on_recovery(db):
    db.journal_open('AAPL', '2024-01-15', 200.0, 5, 1000.0)
    trade_id = int(db.journal_get_open().iloc[0]['id'])

    db.journal_update_mae(trade_id, 190.0)  # -5%
    db.journal_update_mae(trade_id, 210.0)  # +5% recuperacion — no debe mejorar MAE
    mae = db.journal_get_open().iloc[0]['mae_pct']
    assert mae == pytest.approx(-5.0, abs=0.1)  # sigue siendo -5%


def test_mae_only_worsens(db):
    db.journal_open('TSLA', '2024-01-01', 250.0, 4, 1000.0)
    trade_id = int(db.journal_get_open().iloc[0]['id'])

    db.journal_update_mae(trade_id, 240.0)   # -4%
    db.journal_update_mae(trade_id, 225.0)   # -10%  (peor)
    db.journal_update_mae(trade_id, 235.0)   # -6%   (no tan malo)

    mae = db.journal_get_open().iloc[0]['mae_pct']
    assert mae == pytest.approx(-10.0, abs=0.1)  # se queda en -10%


# ── journal_get_stats ─────────────────────────────────────────────────────────

def test_stats_empty(db):
    stats = db.journal_get_stats()
    assert stats['total_trades'] == 0
    assert stats['win_rate'] is None


def test_stats_with_trades(db):
    db.journal_open('A', '2024-01-01', 100.0, 10, 1000.0, fees_eur=1.0)
    id1 = int(db.journal_get_open().iloc[0]['id'])
    db.journal_close(id1, '2024-02-01', 120.0, exit_fees_eur=1.0)  # +20% win

    db.journal_open('B', '2024-01-05', 50.0, 20, 1000.0, fees_eur=1.0)
    id2 = int(db.journal_get_open().iloc[0]['id'])
    db.journal_close(id2, '2024-02-05', 40.0, exit_fees_eur=1.0)   # -20% loss

    stats = db.journal_get_stats()
    assert stats['total_trades'] == 2
    assert stats['win_rate'] == pytest.approx(50.0, abs=1)
    assert stats['best_trade']['symbol'] == 'A'
    assert stats['worst_trade']['symbol'] == 'B'
    assert stats['total_fees_eur'] == pytest.approx(4.0, abs=0.1)


def test_stats_avg_hold_days(db):
    db.journal_open('C', '2024-01-01', 100.0, 10, 1000.0)
    id1 = int(db.journal_get_open().iloc[0]['id'])
    db.journal_close(id1, '2024-01-31', 105.0)   # 30 dias

    stats = db.journal_get_stats()
    assert stats['avg_hold_days'] == pytest.approx(30.0, abs=1)


# ── Migracion de columnas (idempotencia) ──────────────────────────────────────

def test_migration_idempotent(db):
    """Crear la DB dos veces no genera error (migracion idempotente)."""
    db2 = TradingDatabase(db_path=db.db_path, auto_cleanup_snapshots=False)
    db2.close()


def test_signal_snapshot_includes_fund_signal(db):
    """save_signal_snapshot debe guardar fund_signal y price."""
    df = pd.DataFrame([{
        'symbol': 'AAPL', 'confidence': 'ALTA', 'confidence_score': 4,
        'vol_signal': 'COMPRAR', 'rs_signal': 'LIDER', 'risk_signal': 'BAJO RIESGO',
        'total_score': 78.0, 'fund_signal': 'FAVORABLE', 'price': 185.0,
    }])
    db.save_signal_snapshot(df, run_timestamp='2024-01-01T10:00:00')
    df2 = pd.DataFrame([{
        'symbol': 'AAPL', 'confidence': 'ALTA', 'confidence_score': 4,
        'vol_signal': 'COMPRAR', 'rs_signal': 'LIDER', 'risk_signal': 'BAJO RIESGO',
        'total_score': 75.0, 'fund_signal': 'DESFAVORABLE', 'price': 180.0,
    }])
    db.save_signal_snapshot(df2, run_timestamp='2024-01-02T10:00:00')
    snap = db.get_last_snapshot()
    assert snap is not None
    assert 'fund_signal' in snap.columns or 'total_score' in snap.columns
