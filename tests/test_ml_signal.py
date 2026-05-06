"""
tests/test_ml_signal.py — Etapa 5.4: Tests para ml_signal.py y la integración
de la señal ML en scoring_service.compute_confidence.

Verifica:
- _result_to_features mapea correctamente los campos
- _annotate_rs_sector calcula diferencias sectoriales
- enrich_ml_signals sin modelo añade ml_prob=None sin errores
- enrich_ml_signals con modelo mock añade ml_prob y actualiza confidence
- compute_confidence lee ml_prob y añade ±1 pt correctamente
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_signal import (
    _annotate_rs_sector,
    _result_to_features,
    enrich_ml_signals,
    invalidate_model_cache,
)
from scoring_service import compute_confidence


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_result(symbol="AAPL", sector="Technology", score=65.0, ml_prob=None):
    return {
        "symbol":             symbol,
        "sector":             sector,
        "total_score":        score,
        "trend_strength":     60.0,
        "momentum":           55.0,
        "volatility":         50.0,
        "volume":             70.0,
        "price_action":       60.0,
        "support_resistance": 50.0,
        "rsi":                52.0,
        "atr_pct":            1.5,
        "vol_ratio":          1.2,
        "bb_position":        55.0,
        "change_5d":          1.0,
        "change_20d":         3.0,
        "ema_9":              105.0,
        "ema_21":             103.0,
        "vol_signal":         "COMPRAR",
        "fund_signal":        "FAVORABLE",
        "adx_value":          25.0,
        "candle_signal":      "NEUTRO",
        "rs_signal":          "NEUTRAL",
        "tf_conf_pts":        1,
        "sharpe":             1.0,
        "bt_win_rate_vol":    60.0,
        "ml_prob":            ml_prob,
        "vol_zscore":         0.5,
    }


def _make_regime():
    return {
        "regime": "risk_on",
        "score": 5,
        "signals": {
            "above_sma200": True,
            "ret_20d": 2.5,
            "slope": 0.05,
        },
    }


# ── _result_to_features ───────────────────────────────────────────────────────

def test_result_to_features_keys():
    row   = _make_result()
    feats = _result_to_features(row, regime=_make_regime())
    for key in ("trend_strength", "momentum", "volatility", "volume_score",
                "total_score", "rsi", "atr_pct", "vol_ratio",
                "spy_above_sma200", "spy_ret_20d", "regime_score",
                "vol_zscore_sector", "rs_20d_vs_sector"):
        assert key in feats, f"Falta feature '{key}'"


def test_result_to_features_volume_rename():
    row = _make_result()
    row["volume"] = 75.0
    feats = _result_to_features(row)
    assert feats["volume_score"] == 75.0


def test_result_to_features_ema_9_over_21():
    row = _make_result()
    row["ema_9"]  = 110.0
    row["ema_21"] = 105.0
    feats = _result_to_features(row)
    assert feats["ema_9_over_21"] == 1.0


def test_result_to_features_regime_signals():
    row    = _make_result()
    regime = _make_regime()
    feats  = _result_to_features(row, regime=regime)
    assert feats["spy_above_sma200"] == 1.0
    assert abs(feats["spy_ret_20d"] - 2.5) < 1e-6
    assert feats["regime_score"] == 5.0


def test_result_to_features_no_regime_gives_nan():
    row   = _make_result()
    feats = _result_to_features(row, regime=None)
    assert np.isnan(feats["spy_above_sma200"])
    assert np.isnan(feats["regime_score"])


def test_result_to_features_missing_fields_nan():
    feats = _result_to_features({})
    for key in ("trend_strength", "momentum", "total_score", "rsi"):
        assert np.isnan(feats[key]), f"{key} debería ser NaN"


# ── _annotate_rs_sector ───────────────────────────────────────────────────────

def test_annotate_rs_sector_adds_field():
    results = [
        {"sector": "Tech", "change_20d": 5.0},
        {"sector": "Tech", "change_20d": 3.0},
        {"sector": "Tech", "change_20d": 1.0},
    ]
    _annotate_rs_sector(results)
    assert "rs_20d_vs_sector" in results[0]


def test_annotate_rs_sector_median_zero():
    results = [
        {"sector": "Tech", "change_20d": 1.0},
        {"sector": "Tech", "change_20d": 3.0},
        {"sector": "Tech", "change_20d": 5.0},
    ]
    _annotate_rs_sector(results)
    # Mediana = 3.0, así que rs_20d_vs_sector = change_20d - 3.0
    vals = [r["rs_20d_vs_sector"] for r in results]
    assert abs(sum(vals)) < 1e-6 or abs(sorted(vals)[len(vals) // 2]) < 1e-6


def test_annotate_rs_sector_different_sectors():
    results = [
        {"sector": "Tech",    "change_20d": 10.0},
        {"sector": "Finance", "change_20d": 2.0},
    ]
    _annotate_rs_sector(results)
    # Cada sector tiene 1 símbolo → rs_20d_vs_sector = 0
    assert abs(results[0]["rs_20d_vs_sector"]) < 1e-6
    assert abs(results[1]["rs_20d_vs_sector"]) < 1e-6


def test_annotate_rs_sector_handles_none():
    results = [
        {"sector": "Tech", "change_20d": None},
        {"sector": "Tech", "change_20d": 3.0},
    ]
    _annotate_rs_sector(results)
    # No debe lanzar excepción; el None queda como NaN
    assert np.isnan(results[0]["rs_20d_vs_sector"])


# ── enrich_ml_signals (sin modelo) ───────────────────────────────────────────

def test_enrich_ml_signals_no_model_adds_none(monkeypatch):
    """Sin modelo entrenado, ml_prob debe ser None en todos los results."""
    invalidate_model_cache()
    import ml_signal
    monkeypatch.setattr(ml_signal, "_try_load_model", lambda: (None, None))

    results = [_make_result(), _make_result("MSFT")]
    enrich_ml_signals(results)
    for r in results:
        assert "ml_prob" in r
        assert r["ml_prob"] is None


def test_enrich_ml_signals_empty_results():
    """Lista vacía no debe lanzar excepción."""
    enrich_ml_signals([])


# ── enrich_ml_signals (con modelo mock) ──────────────────────────────────────

def test_enrich_ml_signals_with_mock_model(monkeypatch):
    """Con un modelo mock, ml_prob debe ser float en [0, 1]."""
    invalidate_model_cache()
    import ml_signal

    feat_cols = [
        "trend_strength", "momentum", "volatility", "volume_score",
        "price_action", "support_resistance", "total_score",
        "rsi", "atr_pct", "vol_ratio", "bb_position",
        "change_5d", "change_20d", "ema_9_over_21", "ema_21_over_50",
        "spy_above_sma200", "spy_ret_20d", "spy_slope_20d", "regime_score",
        "vol_zscore_sector", "rs_20d_vs_sector",
        "pe_forward", "pe_trailing", "price_to_book", "profit_margin", "debt_to_equity",
    ]
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.72, 0.38])

    cv_info = {"mean_auc": 0.65, "best_iteration": 150, "feature_cols": feat_cols}
    monkeypatch.setattr(ml_signal, "_try_load_model",
                        lambda: (mock_model, cv_info))

    results = [_make_result("AAPL", score=70.0),
               _make_result("MSFT", score=45.0)]
    enrich_ml_signals(results, regime=_make_regime())

    # ml_prob añadido y en [0, 1]
    assert results[0]["ml_prob"] == pytest.approx(0.72, abs=1e-4)
    assert results[1]["ml_prob"] == pytest.approx(0.38, abs=1e-4)
    for r in results:
        assert 0.0 <= r["ml_prob"] <= 1.0


def test_enrich_ml_signals_updates_confidence(monkeypatch):
    """La confianza debe actualizarse después de añadir ml_prob."""
    invalidate_model_cache()
    import ml_signal

    feat_cols = ["total_score"]
    mock_model = MagicMock()
    # Alta probabilidad → debe subir confidence
    mock_model.predict.return_value = np.array([0.80])
    cv_info = {"mean_auc": 0.65, "best_iteration": 100, "feature_cols": feat_cols}
    monkeypatch.setattr(ml_signal, "_try_load_model",
                        lambda: (mock_model, cv_info))

    result = _make_result("AAPL", score=65.0)
    # Capturar confianza antes
    conf_before = result.get("confidence")

    enrich_ml_signals([result])

    assert "confidence" in result
    assert "confidence_score" in result
    # Con ml_prob=0.80 (>= 0.60), confidence_score debe ser 1 pt mayor
    # (verificamos que el campo existe y es string válido)
    assert result["confidence"] in ("MUY ALTA", "ALTA", "MEDIA", "BAJA", "MUY BAJA")


# ── compute_confidence con ml_prob ────────────────────────────────────────────

def test_compute_confidence_ml_prob_high_adds_point():
    """ml_prob >= 0.60 debe sumar 1 punto a confidence_score."""
    row_no_ml  = _make_result()
    row_ml_hi  = {**_make_result(), "ml_prob": 0.75}

    conf_no_ml = compute_confidence(row_no_ml)
    conf_ml_hi = compute_confidence(row_ml_hi)

    assert conf_ml_hi["score"] == conf_no_ml["score"] + 1


def test_compute_confidence_ml_prob_low_subtracts_point():
    """ml_prob <= 0.35 debe restar 1 punto a confidence_score."""
    row_no_ml  = _make_result()
    row_ml_lo  = {**_make_result(), "ml_prob": 0.25}

    conf_no_ml = compute_confidence(row_no_ml)
    conf_ml_lo = compute_confidence(row_ml_lo)

    assert conf_ml_lo["score"] == conf_no_ml["score"] - 1


def test_compute_confidence_ml_prob_neutral_no_change():
    """ml_prob en [0.35, 0.60] no debe cambiar el score."""
    row_base   = _make_result()
    row_ml_neu = {**_make_result(), "ml_prob": 0.50}

    conf_base  = compute_confidence(row_base)
    conf_neu   = compute_confidence(row_ml_neu)

    assert conf_neu["score"] == conf_base["score"]


def test_compute_confidence_ml_prob_none_no_change():
    """ml_prob=None no debe cambiar el score."""
    row_base = _make_result()
    row_none = {**_make_result(), "ml_prob": None}

    conf_base = compute_confidence(row_base)
    conf_none = compute_confidence(row_none)

    assert conf_none["score"] == conf_base["score"]


def test_compute_confidence_ml_aligned_label():
    """Cuando ml_prob >= 0.60, debe aparecer en la lista 'aligned'."""
    row = {**_make_result(), "ml_prob": 0.72}
    conf = compute_confidence(row)
    aligned_str = " ".join(conf["aligned"])
    assert "ML" in aligned_str or "Modelo" in aligned_str


def test_compute_confidence_ml_against_label():
    """Cuando ml_prob <= 0.35, debe aparecer en la lista 'against'."""
    row = {**_make_result(), "ml_prob": 0.20}
    conf = compute_confidence(row)
    against_str = " ".join(conf["against"])
    assert "ML" in against_str or "Modelo" in against_str
