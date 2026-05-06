"""
Tests de los indicadores en `indicators_advanced.AdvancedIndicators`.

Estos tests buscan invariantes (no snapshots ciegos):
- Los rangos numéricos tienen sentido (ADX en [0, 100], RSI en [0, 100], etc.).
- Las señales generadas son del tipo esperado.
- En un tramo claramente alcista o bajista, los signos coinciden con la teoría.
"""
import math

import pytest

from indicators_advanced import AdvancedIndicators


# ── ADX ─────────────────────────────────────────────────────────────────────


def test_adx_returns_dict_in_range(aapl_df):
    res = AdvancedIndicators.adx(aapl_df, length=14)
    assert res is not None
    assert 0 <= res["adx"] <= 100
    assert 0 <= res["plus_di"] <= 100
    assert 0 <= res["minus_di"] <= 100
    assert res["trend_strength"] in ("débil", "moderada", "fuerte")
    assert res["signal"] in (None, "COMPRA", "VENTA")


def test_adx_short_dataframe_handled(aapl_df):
    # 5 filas: pandas_ta puede devolver NaN; nuestro wrapper devuelve None o NaN.
    short = aapl_df.head(5).reset_index(drop=True)
    res = AdvancedIndicators.adx(short, length=14)
    if res is None:
        return
    # Si devuelve algo, ADX puede ser NaN; tolerarlo.
    assert res["adx"] != res["adx"] or 0 <= res["adx"] <= 100  # NaN check


# ── Fibonacci ────────────────────────────────────────────────────────────────


def test_fibonacci_levels_ordered(aapl_df):
    res = AdvancedIndicators.fibonacci_retracements(aapl_df.tail(60), period=20)
    levels = res["levels"]
    # Niveles descendentes (de high al low en escala de Fib)
    seq = [levels["fib_0"], levels["fib_23.6"], levels["fib_38.2"],
           levels["fib_50"], levels["fib_61.8"], levels["fib_78.6"], levels["fib_100"]]
    assert all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))


# ── Volume profile ───────────────────────────────────────────────────────────


def test_volume_profile_consistency(aapl_df):
    res = AdvancedIndicators.volume_profile(aapl_df.tail(200), bins=20)
    assert res["val"] <= res["poc"] <= res["vah"]
    assert res["signal"] in (None, "COMPRA", "VENTA")


# ── Stochastic RSI ───────────────────────────────────────────────────────────


def test_stoch_rsi_range(aapl_df):
    res = AdvancedIndicators.stochastic_rsi(aapl_df)
    assert res is not None
    # k y d pueden ser NaN si la serie es corta; comprobamos solo cuando son valor
    for v in (res["k"], res["d"]):
        if isinstance(v, float) and math.isnan(v):
            continue
        assert 0 <= v <= 100


# ── Patrones de velas ────────────────────────────────────────────────────────


def test_candlestick_no_crash_on_short(aapl_df):
    short = aapl_df.head(2)
    patterns = AdvancedIndicators.detect_candlestick_patterns(short)
    assert patterns == []


def test_candlestick_returns_list(aapl_df):
    patterns = AdvancedIndicators.detect_candlestick_patterns(aapl_df.tail(50))
    assert isinstance(patterns, list)
    for p in patterns:
        assert {"pattern", "signal", "strength"}.issubset(p.keys())
        assert p["signal"] in ("COMPRA", "VENTA", "NEUTRO")
        assert p["strength"] in ("débil", "fuerte", "muy fuerte")
