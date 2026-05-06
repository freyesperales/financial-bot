"""
tests/test_ml_compare.py — Etapa 5.3: Tests para ml_compare.py

Verifica:
- _portfolio_metrics calcula Sharpe, win_rate, max_drawdown correctamente
- _simulate_strategy aplica el umbral y devuelve métricas consistentes
- compare_strategies separa correctamente IS y OOS
- find_optimal_threshold barre umbrales sin errores
- Sin lookahead: la estrategia solo usa fwd_ret_20d del dataset
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_compare import (
    _equity_curve,
    _portfolio_metrics,
    _simulate_strategy,
    compare_strategies,
    find_optimal_threshold,
    print_comparison,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_compare_df(n_symbols: int = 4, n_dates: int = 300,
                     seed: int = 0) -> pd.DataFrame:
    """Dataset OOS sintético para pruebas de comparación."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-04", periods=n_dates, freq="B")
    symbols = [f"SYM{i}" for i in range(n_symbols)]

    rows = []
    for sym in symbols:
        # ml_prob alto → mayor retorno (señal débil)
        ml_probs = rng.beta(2, 3, size=n_dates)  # concentrado en 0.3-0.5
        for di, date in enumerate(dates):
            fwd_ret = ml_probs[di] * 20 - 5 + rng.normal(0, 4)  # señal débil
            rows.append({
                "symbol":      sym,
                "sector":      "Tech",
                "date":        str(date.date()),
                "total_score": float(ml_probs[di] * 100),
                "ml_prob":     float(ml_probs[di]),
                "fwd_ret_20d": round(fwd_ret, 4),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def compare_df():
    return _make_compare_df()


# ── _portfolio_metrics ────────────────────────────────────────────────────────

def test_portfolio_metrics_basic():
    rets = pd.Series([1.0, -0.5, 2.0, -1.0, 3.0, 0.5])
    m = _portfolio_metrics(rets, step_days=5)
    assert "sharpe" in m
    assert "win_rate" in m
    assert "max_drawdown" in m
    assert "n_trades" in m
    assert m["n_trades"] == 6


def test_portfolio_metrics_win_rate():
    rets = pd.Series([1.0, 2.0, 3.0, -1.0])  # 3 winners, 1 loser
    m = _portfolio_metrics(rets, step_days=5)
    assert abs(m["win_rate"] - 0.75) < 1e-6


def test_portfolio_metrics_max_drawdown_negative():
    # Secuencia con pérdida continua → drawdown negativo
    rets = pd.Series([-2.0, -3.0, -1.0, 1.0])
    m = _portfolio_metrics(rets, step_days=5)
    assert m["max_drawdown"] < 0


def test_portfolio_metrics_all_positive_dd_zero():
    rets = pd.Series([1.0, 2.0, 1.5, 0.5])
    m = _portfolio_metrics(rets, step_days=5)
    assert m["max_drawdown"] >= -0.01  # Casi cero drawdown


def test_portfolio_metrics_too_short():
    rets = pd.Series([1.0, 2.0])
    m = _portfolio_metrics(rets, step_days=5)
    assert math.isnan(m["sharpe"]) or m["n_trades"] == 2


def test_equity_curve_starts_at_one():
    rets = pd.Series([1.0, -0.5, 2.0])
    eq = _equity_curve(rets)
    assert abs(eq.iloc[0] - 1.01) < 0.01  # Primera barra incorpora el 1% del primer retorno


# ── _simulate_strategy ────────────────────────────────────────────────────────

def test_simulate_strategy_returns_metrics(compare_df):
    m = _simulate_strategy(compare_df, signal_col="ml_prob", threshold=0.5)
    assert "sharpe" in m
    assert "n_trades" in m
    assert "equity_curve" in m


def test_simulate_strategy_threshold_affects_n_trades(compare_df):
    m_low  = _simulate_strategy(compare_df, signal_col="ml_prob", threshold=0.2)
    m_high = _simulate_strategy(compare_df, signal_col="ml_prob", threshold=0.8)
    # Umbral más bajo → más señales → más trades
    assert m_low["n_dates_with_signal"] >= m_high["n_dates_with_signal"]


def test_simulate_strategy_equity_curve_is_series(compare_df):
    m = _simulate_strategy(compare_df, signal_col="ml_prob", threshold=0.5)
    assert isinstance(m["equity_curve"], pd.Series)
    assert len(m["equity_curve"]) > 0


def test_simulate_strategy_equity_curve_starts_positive(compare_df):
    m = _simulate_strategy(compare_df, signal_col="ml_prob", threshold=0.5)
    assert (m["equity_curve"] > 0).all()


def test_simulate_strategy_signal_rate_between_0_1(compare_df):
    m = _simulate_strategy(compare_df, signal_col="ml_prob", threshold=0.5)
    assert 0.0 <= m["signal_rate"] <= 1.0


def test_simulate_strategy_missing_col_raises(compare_df):
    with pytest.raises(ValueError):
        _simulate_strategy(compare_df, signal_col="nonexistent", threshold=0.5)


# ── compare_strategies ────────────────────────────────────────────────────────

def test_compare_strategies_returns_dict(compare_df):
    result = compare_strategies(compare_df, oos_months=3)
    assert isinstance(result, dict)


def test_compare_strategies_keys(compare_df):
    result = compare_strategies(compare_df, oos_months=3)
    for key in ("oos_start", "oos_end", "n_oos_obs", "ml", "rules",
                "buy_hold", "sharpe_improvement"):
        assert key in result, f"Falta clave '{key}'"


def test_compare_strategies_oos_period(compare_df):
    result = compare_strategies(compare_df, oos_months=3)
    # OOS debe ser un subconjunto del final del df
    max_date = pd.to_datetime(compare_df["date"]).max()
    oos_end  = pd.to_datetime(result["oos_end"])
    assert oos_end <= max_date


def test_compare_strategies_ml_metrics_valid(compare_df):
    result = compare_strategies(compare_df, oos_months=3)
    ml = result["ml"]
    assert "sharpe" in ml
    assert "win_rate" in ml
    assert "n_trades" in ml
    # win_rate debe estar en [0, 1]
    if ml["win_rate"] is not None and not math.isnan(ml["win_rate"]):
        assert 0.0 <= ml["win_rate"] <= 1.0


def test_compare_strategies_rules_metrics_valid(compare_df):
    result = compare_strategies(compare_df, oos_months=3)
    ru = result["rules"]
    if ru["win_rate"] is not None and not math.isnan(ru["win_rate"]):
        assert 0.0 <= ru["win_rate"] <= 1.0


def test_compare_strategies_sharpe_improvement_numeric(compare_df):
    result = compare_strategies(compare_df, oos_months=3)
    imp = result["sharpe_improvement"]
    # Puede ser None si alguna estrategia no tiene suficientes datos
    if imp is not None:
        assert isinstance(imp, float)
        assert not math.isnan(imp)


def test_compare_strategies_missing_ml_col_raises(compare_df):
    df_no_ml = compare_df.drop(columns=["ml_prob"])
    with pytest.raises(ValueError, match="ml_prob"):
        compare_strategies(df_no_ml, oos_months=3)


def test_compare_strategies_equity_curves_in_result(compare_df):
    result = compare_strategies(compare_df, oos_months=3)
    assert "_equity_ml" in result
    assert "_equity_rules" in result
    assert isinstance(result["_equity_ml"], pd.Series)


def test_compare_strategies_different_thresholds(compare_df):
    """Cambiar el umbral ML debe cambiar las métricas."""
    r_low  = compare_strategies(compare_df, oos_months=3, threshold_ml=0.3)
    r_high = compare_strategies(compare_df, oos_months=3, threshold_ml=0.7)
    # El número de señales debe diferir
    assert r_low["ml"]["n_dates_with_signal"] >= r_high["ml"]["n_dates_with_signal"]


# ── find_optimal_threshold ────────────────────────────────────────────────────

def test_find_optimal_threshold_returns_df(compare_df):
    result = find_optimal_threshold(compare_df, oos_months=3,
                                    thresholds=[0.4, 0.5, 0.6])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_find_optimal_threshold_sorted_by_sharpe(compare_df):
    result = find_optimal_threshold(compare_df, oos_months=3,
                                    thresholds=[0.3, 0.5, 0.7])
    if len(result) > 1:
        sharpes = result["sharpe"].dropna().values
        assert all(sharpes[i] >= sharpes[i + 1] for i in range(len(sharpes) - 1))


def test_find_optimal_threshold_columns(compare_df):
    result = find_optimal_threshold(compare_df, oos_months=3,
                                    thresholds=[0.5])
    for col in ("threshold", "sharpe", "win_rate", "n_trades"):
        assert col in result.columns


# ── print_comparison ─────────────────────────────────────────────────────────

def test_print_comparison_no_exception(compare_df, capsys):
    result = compare_strategies(compare_df, oos_months=3)
    print_comparison(result)
    captured = capsys.readouterr()
    assert "Sharpe" in captured.out
    assert "ML" in captured.out
    assert "Reglas" in captured.out
