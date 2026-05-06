"""
Tests del scoring de StockAnalyzer.

Filosofía:
- Aseveramos rangos y propiedades estables (no valores numéricos exactos)
  para evitar fragilidad ante cambios menores de yfinance / pandas_ta.
- Comprobamos que CAMBIAR un peso del scoring afecta al total — eso garantiza
  que el sistema de scoring está realmente conectado.
- Tests deterministas: solo dependen de los CSV de fixture, no de internet.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from database import TradingDatabase
from stock_analyzer import StockAnalyzer


@pytest.fixture(scope="module")
def analyzer_with_tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = TradingDatabase(db_path=path, auto_cleanup_snapshots=False)
    yield StockAnalyzer(db)
    db.close()
    for ext in ("", "-wal", "-shm"):
        try:
            os.remove(path + ext)
        except OSError:
            pass


# ── calculate_technical_score ───────────────────────────────────────────────


def test_technical_score_in_range(analyzer_with_tmp_db, aapl_df):
    res = analyzer_with_tmp_db.calculate_technical_score(aapl_df, "AAPL")
    assert res is not None
    assert 0 <= res["total_score"] <= 100
    for sub in ("trend_strength", "momentum", "volatility",
                "volume", "price_action", "support_resistance"):
        assert 0 <= res["scores"][sub] <= 100


def test_technical_score_returns_none_with_few_rows(analyzer_with_tmp_db, aapl_df):
    short = aapl_df.head(50)
    res = analyzer_with_tmp_db.calculate_technical_score(short, "AAPL")
    assert res is None


def test_weight_change_affects_total_score(analyzer_with_tmp_db, aapl_df):
    """Cambiar el régimen (que controla los pesos) debe alterar el total score.

    Los pesos ya no se mutan directamente: _get_weights() usa el régimen activo
    via scoring_service.get_weights(). Verificamos que el mecanismo esté conectado
    al cálculo comparando risk_on (trend heavy) vs panic (sr heavy).
    """
    a = analyzer_with_tmp_db
    original_regime = a._current_regime

    a._current_regime = 'risk_on'   # trend=0.30, momentum=0.25, sr=0.05
    res_risk_on = a.calculate_technical_score(aapl_df, "AAPL")

    a._current_regime = 'panic'     # trend=0.15, momentum=0.10, sr=0.20
    res_panic = a.calculate_technical_score(aapl_df, "AAPL")

    a._current_regime = original_regime

    assert res_risk_on is not None and res_panic is not None
    # Si trend_score y sr_score difieren, los totales deben diferir
    trend_s = res_risk_on["scores"]["trend_strength"]
    sr_s    = res_risk_on["scores"]["support_resistance"]
    if abs(trend_s - sr_s) > 1:
        assert res_risk_on["total_score"] != res_panic["total_score"]


def test_etf_uses_etf_weights(analyzer_with_tmp_db, spy_df):
    a = analyzer_with_tmp_db
    res = a.calculate_technical_score(spy_df, "SPY")
    assert res is not None
    # SPY es ETF de equity → trend_strength tiene peso 0.30
    weights = a._get_weights("SPY")
    assert weights["trend_strength"] == 0.30
    assert weights["support_resistance"] == 0.05


def test_bond_etf_uses_bond_weights(analyzer_with_tmp_db, tlt_df):
    a = analyzer_with_tmp_db
    res = a.calculate_technical_score(tlt_df, "TLT")
    assert res is not None
    weights = a._get_weights("TLT")
    assert weights["volatility"] == 0.20
    assert weights["volume"] == 0.10


# ── calculate_volume_ma_signal ──────────────────────────────────────────────


def test_volume_ma_signal_shape(analyzer_with_tmp_db, aapl_df):
    res = analyzer_with_tmp_db.calculate_volume_ma_signal(aapl_df)
    assert res is not None
    assert res["signal"] in ("COMPRAR", "VENDER")
    assert 0 <= res["signal_strength"] <= 100
    assert res["ratio"] > 0
    assert res["ma7"] >= 0
    assert res["ma60"] >= 0


def test_volume_ma_signal_short_returns_none(analyzer_with_tmp_db, aapl_df):
    res = analyzer_with_tmp_db.calculate_volume_ma_signal(aapl_df.head(30))
    assert res is None


# ── calculate_adx_signal ────────────────────────────────────────────────────


def test_adx_signal_shape(analyzer_with_tmp_db, aapl_df):
    res = analyzer_with_tmp_db.calculate_adx_signal(aapl_df)
    assert res is not None
    assert 0 <= res["adx"] <= 100
    assert res["strength"] in ("débil", "moderada", "fuerte", "muy fuerte")
    assert res["direction"] in ("alcista", "bajista")
    assert 0 < res["conf_mult"] <= 2


# ── calculate_risk_metrics ──────────────────────────────────────────────────


def test_risk_metrics_shape(analyzer_with_tmp_db, aapl_df):
    res = analyzer_with_tmp_db.calculate_risk_metrics(aapl_df)
    assert res is not None
    # Sharpe puede ser positivo o negativo, pero finito
    assert isinstance(res["sharpe"], float)
    assert res["sharpe"] == res["sharpe"]  # NaN check
    # Max drawdown siempre <= 0
    assert res["max_dd"] <= 0
    # Kelly bootstrap clamped a [0, 25] (Half-Kelly cap 25%)
    assert 0 <= res["kelly_pct"] <= 25
    assert 0 <= res["kelly_pct_p5"]  <= 25
    assert 0 <= res["kelly_pct_p95"] <= 25
    assert res["kelly_pct_p5"] <= res["kelly_pct_p95"]
    assert res["risk_signal"] in ("BAJO RIESGO", "RIESGO MEDIO", "ALTO RIESGO")


def test_risk_metrics_short_returns_none(analyzer_with_tmp_db, aapl_df):
    res = analyzer_with_tmp_db.calculate_risk_metrics(aapl_df.head(30))
    assert res is None


# ── calculate_timeframe_alignment ───────────────────────────────────────────


def test_timeframe_alignment_shape(analyzer_with_tmp_db, aapl_df):
    res = analyzer_with_tmp_db.calculate_timeframe_alignment(aapl_df)
    assert res is not None
    assert res["signal"] in ("ALCISTA", "BAJISTA", "NEUTRO")
    assert -2 <= res["conf_pts"] <= 2
    assert res["weekly_trend"] in ("ALCISTA", "BAJISTA", "LATERAL")
    assert res["daily_trend"]  in ("ALCISTA", "BAJISTA", "LATERAL")


# ── calculate_backtest ──────────────────────────────────────────────────────


def test_backtest_shape(analyzer_with_tmp_db, aapl_df):
    res = analyzer_with_tmp_db.calculate_backtest(aapl_df, forward_days=20)
    assert res is not None
    if res["bt_n_tech"] > 0:
        assert 0 <= res["bt_win_rate_tech"] <= 100
        assert res["bt_exposure_pct_tech"] is not None
        assert res["bt_avg_hold_days_tech"] is not None
    if res["bt_n_vol"] > 0:
        assert 0 <= res["bt_win_rate_vol"] <= 100
        assert res["bt_exposure_pct_vol"] is not None
    assert isinstance(res["bt_buy_hold"], float)


def test_walk_forward_backtest(analyzer_with_tmp_db, aapl_df):
    res = analyzer_with_tmp_db.walk_forward_backtest(aapl_df, is_years=2, oos_months=6)
    if res is None:
        return  # datos insuficientes — test de smoke pasa igualmente
    assert res['n_folds'] >= 1
    assert res['is_win_rate'] is None or 0 <= res['is_win_rate'] <= 100
    assert res['oos_win_rate'] is None or 0 <= res['oos_win_rate'] <= 100
    if res['degradation'] is not None:
        assert res['degradation'] >= 0


def test_backtest_no_overlapping_trades(analyzer_with_tmp_db, aapl_df):
    """State machine must prevent duplicate entries during trending periods."""
    res = analyzer_with_tmp_db.calculate_backtest(aapl_df, forward_days=20)
    assert res is not None
    # With non-overlapping trades, n_tech must be < len(df)/forward_days
    # (old buggy version produced n ≈ days-in-uptrend, easily 100+ for AAPL)
    n = res["bt_n_tech"]
    max_possible = len(aapl_df) // 20
    assert n <= max_possible, f"n={n} suggests overlapping trades (max_possible={max_possible})"


# ── calculate_signal_confidence ─────────────────────────────────────────────


def test_signal_confidence_strong_alignment_yields_high_confidence(analyzer_with_tmp_db):
    a = analyzer_with_tmp_db
    row = {
        "total_score": 80,
        "adx_value": 35,
        "vol_signal": "COMPRAR",
        "candle_signal": "COMPRA",
        "candle_strength": "muy fuerte",
        "candle_pattern": "Bullish Engulfing",
        "fund_signal": "FAVORABLE",
        "rs_signal": "LIDER", "rs_60d": 12.0,
        "tf_conf_pts": 2, "tf_alignment": "ALINEADO ALCISTA",
        "bt_win_rate_vol": 70, "bt_expectancy_vol": 2.0,
        "sharpe": 1.6, "max_dd": -10,
    }
    vix = {"vix": 30.0, "level": "MIEDO", "signal": "COMPRAR", "color": "buy"}
    res = a.calculate_signal_confidence(row, vix)
    assert res["confidence"] in ("ALTA", "MUY ALTA")
    assert res["score"] >= 3


def test_signal_confidence_strong_misalignment_yields_low_confidence(analyzer_with_tmp_db):
    a = analyzer_with_tmp_db
    row = {
        "total_score": 30,
        "adx_value": 12,
        "vol_signal": "VENDER",
        "candle_signal": "VENTA",
        "candle_strength": "fuerte",
        "candle_pattern": "Bearish Engulfing",
        "fund_signal": "DESFAVORABLE",
        "rs_signal": "REZAGADO", "rs_60d": -12.0,
        "tf_conf_pts": -2, "tf_alignment": "ALINEADO BAJISTA",
        "bt_win_rate_vol": 30, "bt_expectancy_vol": -1.0,
        "sharpe": -0.5, "max_dd": -45,
    }
    res = a.calculate_signal_confidence(row, None)
    assert res["confidence"] in ("BAJA", "MUY BAJA")
    assert res["score"] <= -1


# ── _target_time_range ──────────────────────────────────────────────────────


def test_target_time_range_monotone_in_adx(analyzer_with_tmp_db):
    """A mayor ADX, menos días para alcanzar el mismo objetivo (en el mínimo)."""
    a = analyzer_with_tmp_db
    weak = a._target_time_range(gain_pct=10, atr_pct=2.0, adx_value=15)
    strong = a._target_time_range(gain_pct=10, atr_pct=2.0, adx_value=40)
    assert strong[0] <= weak[0]
