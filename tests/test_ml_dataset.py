"""
tests/test_ml_dataset.py — Etapa 5.1: Tests para ml_dataset.py

Usa datos sintéticos (sin necesidad de DB real) para verificar:
- Construcción correcta de features
- Ausencia de lookahead bias (target usa solo precio futuro)
- Sector features cross-seccionales
- Carga de fundamentales desde caché en disco
- dataset_summary y feature_columns
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import indicators_core as ic
from ml_dataset import (
    _add_sector_features,
    _features_at_row,
    _load_fund_features,
    _prepare_price_df,
    dataset_summary,
    feature_columns,
)


# ── Fixture: OHLCV sintético (250+ filas para subscores) ─────────────────────

def _make_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Serie OHLCV diaria sintética con trend suave."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, size=n))
    high  = close * (1 + rng.uniform(0.001, 0.02, size=n))
    low   = close * (1 - rng.uniform(0.001, 0.02, size=n))
    open_ = close * (1 + rng.normal(0, 0.005, size=n))
    vol   = rng.integers(500_000, 5_000_000, size=n).astype(float)
    df = pd.DataFrame({
        "timestamp": dates,
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": vol,
    })
    return df


@pytest.fixture(scope="module")
def ohlcv_df():
    return _make_ohlcv(300)


@pytest.fixture(scope="module")
def ohlcv_with_indicators(ohlcv_df):
    return ic.compute_all(ohlcv_df.copy())


# ── _prepare_price_df ─────────────────────────────────────────────────────────

def test_prepare_price_df_computes_indicators(ohlcv_df):
    result = _prepare_price_df(ohlcv_df)
    assert result is not None
    # Debe tener columnas de indicadores
    for col in ("ema_9", "ema_21", "ema_50", "rsi", "macd", "bb_upper", "bb_lower", "atr"):
        assert col in result.columns, f"Falta columna {col}"


def test_prepare_price_df_none_on_short_data():
    short_df = _make_ohlcv(50)
    result = _prepare_price_df(short_df)
    assert result is None


def test_prepare_price_df_none_on_empty():
    result = _prepare_price_df(pd.DataFrame())
    assert result is None


def test_prepare_price_df_sorted_ascending(ohlcv_df):
    shuffled = ohlcv_df.sample(frac=1, random_state=42)
    result = _prepare_price_df(shuffled)
    assert result is not None
    ts = pd.to_datetime(result["timestamp"])
    assert ts.is_monotonic_increasing


# ── _features_at_row ─────────────────────────────────────────────────────────

def test_features_at_row_returns_dict(ohlcv_with_indicators):
    feats = _features_at_row(ohlcv_with_indicators, idx=150)
    assert feats is not None
    assert isinstance(feats, dict)


def test_features_at_row_contains_all_subscores(ohlcv_with_indicators):
    feats = _features_at_row(ohlcv_with_indicators, idx=150)
    assert feats is not None
    for key in ("trend_strength", "momentum", "volatility", "volume_score",
                "price_action", "support_resistance", "total_score"):
        assert key in feats, f"Falta feature {key}"


def test_features_at_row_scores_in_range(ohlcv_with_indicators):
    feats = _features_at_row(ohlcv_with_indicators, idx=150)
    assert feats is not None
    for key in ("trend_strength", "momentum", "volatility", "volume_score",
                "price_action", "support_resistance", "total_score"):
        val = feats[key]
        assert 0 <= val <= 100, f"{key}={val} fuera de [0, 100]"


def test_features_at_row_no_lookahead(ohlcv_with_indicators):
    """Verificar que las features en idx=150 son idénticas si hay más datos después."""
    df = ohlcv_with_indicators
    feats_150 = _features_at_row(df, idx=150)
    # Con más datos hacia adelante, los indicadores en fila 150 deben ser iguales
    feats_150_extended = _features_at_row(df.iloc[:200].copy(), idx=149)
    # Tendencia y momentum deben ser iguales (misma historia hasta ese punto)
    # Nota: idx=150 en df de 300 vs idx=149 en df[:200] — distinto idx, misma fila 150
    # Usamos la misma sub-serie para verificar reproducibilidad
    feats_a = _features_at_row(df.iloc[:175].copy(), idx=150)
    feats_b = _features_at_row(df.iloc[:200].copy(), idx=150)
    assert feats_a is not None and feats_b is not None
    # Los scores deben ser idénticos (misma historia hasta fila 150)
    assert abs(feats_a["trend_strength"] - feats_b["trend_strength"]) < 1e-6
    assert abs(feats_a["rsi"] - feats_b["rsi"]) < 1e-6


def test_features_at_row_none_if_too_short(ohlcv_with_indicators):
    # Con solo 50 filas y idx=49, compute_subscores fallará (< 100 filas)
    short_df = ohlcv_with_indicators.iloc[:50].copy()
    feats = _features_at_row(short_df, idx=49)
    assert feats is None


# ── _add_sector_features ─────────────────────────────────────────────────────

@pytest.fixture
def multi_sector_df():
    """DataFrame con múltiples símbolos de 2 sectores, una sola fecha."""
    rows = []
    for i, sym in enumerate(["AAPL", "MSFT", "NVDA"]):
        rows.append({"symbol": sym, "sector": "Technology", "date": "2023-06-01",
                     "vol_ratio": 1.0 + 0.5 * i, "change_20d": 5.0 + i})
    for i, sym in enumerate(["JPM", "BAC"]):
        rows.append({"symbol": sym, "sector": "Financials", "date": "2023-06-01",
                     "vol_ratio": 0.8 + 0.4 * i, "change_20d": 2.0 + i})
    return pd.DataFrame(rows)


def test_add_sector_features_adds_columns(multi_sector_df):
    result = _add_sector_features(multi_sector_df)
    assert "vol_zscore_sector" in result.columns
    assert "rs_20d_vs_sector" in result.columns


def test_vol_zscore_sector_mean_zero(multi_sector_df):
    result = _add_sector_features(multi_sector_df)
    # Para grupos con >= 3 elementos el z-score debe sumar ~0
    tech_z = result[result["sector"] == "Technology"]["vol_zscore_sector"]
    assert abs(tech_z.mean()) < 1e-10


def test_rs_20d_vs_sector_median_zero(multi_sector_df):
    result = _add_sector_features(multi_sector_df)
    # rs_20d_vs_sector = change_20d - mediana sectorial → valores centrados en mediana
    tech_rs = result[result["sector"] == "Technology"]["rs_20d_vs_sector"]
    assert abs(tech_rs.median()) < 1e-10


def test_add_sector_features_empty_returns_empty():
    result = _add_sector_features(pd.DataFrame())
    assert result.empty


# ── _load_fund_features ───────────────────────────────────────────────────────

def test_load_fund_features_missing_file():
    feats = _load_fund_features("XYZNOTEXIST")
    assert isinstance(feats, dict)
    # Debe devolver NaN para todas las features
    for v in feats.values():
        assert math.isnan(v)


def test_load_fund_features_parses_cache(tmp_path, monkeypatch):
    """Verifica que el parser lee correctamente el JSON de caché."""
    import ml_dataset as md
    # Apuntar FUND_CACHE a directorio temporal
    fund_dir = tmp_path / "fundamentals"
    fund_dir.mkdir()
    monkeypatch.setattr(md, "FUND_CACHE", fund_dir)

    data = {
        "forwardPE": 25.5,
        "trailingPE": 28.0,
        "priceToBook": 3.2,
        "profitMargins": 0.22,
        "debtToEquity": 45.0,
        "_cached_at": "2024-01-01T00:00:00",
    }
    (fund_dir / "AAPL.json").write_text(json.dumps(data), encoding="utf-8")

    feats = _load_fund_features("AAPL")
    assert abs(feats["pe_forward"] - 25.5) < 1e-6
    assert abs(feats["profit_margin"] - 0.22) < 1e-6
    assert abs(feats["price_to_book"] - 3.2) < 1e-6


def test_load_fund_features_handles_corrupt_file(tmp_path, monkeypatch):
    import ml_dataset as md
    fund_dir = tmp_path / "fundamentals"
    fund_dir.mkdir()
    monkeypatch.setattr(md, "FUND_CACHE", fund_dir)
    (fund_dir / "BAD.json").write_text("{{not valid json", encoding="utf-8")
    feats = _load_fund_features("BAD")
    # No debe lanzar excepción; devuelve NaN
    for v in feats.values():
        assert math.isnan(v)


# ── dataset_summary ───────────────────────────────────────────────────────────

def test_dataset_summary_empty():
    result = dataset_summary(pd.DataFrame())
    assert result == {}


def test_dataset_summary_shape(ohlcv_with_indicators):
    # Construir un mini-dataset sintético para probar summary
    rows = []
    df_ind = ohlcv_with_indicators
    for idx in range(110, 250, 10):
        feats = _features_at_row(df_ind, idx)
        if feats is None:
            continue
        rows.append({
            "symbol": "AAPL", "sector": "Technology",
            "date": str(pd.to_datetime(df_ind["timestamp"].iloc[idx]).date()),
            **{k: v for k, v in feats.items() if not k.startswith("_")},
            "target_20d": 1, "fwd_ret_20d": 5.0,
        })
    df = pd.DataFrame(rows)
    summary = dataset_summary(df)
    assert summary["n_rows"] == len(df)
    assert summary["n_symbols"] == 1
    assert "feature_cols" in summary
    assert len(summary["feature_cols"]) > 5


# ── feature_columns ──────────────────────────────────────────────────────────

def test_feature_columns_no_metadata():
    cols = feature_columns()
    for forbidden in ("symbol", "sector", "date", "target_20d", "fwd_ret_20d"):
        assert forbidden not in cols


def test_feature_columns_from_df():
    df = pd.DataFrame(columns=[
        "symbol", "sector", "date", "trend_strength", "momentum",
        "target_20d", "fwd_ret_20d", "rsi",
    ])
    cols = feature_columns(df)
    assert "trend_strength" in cols
    assert "rsi" in cols
    assert "symbol" not in cols
    assert "target_20d" not in cols
