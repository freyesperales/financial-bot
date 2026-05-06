"""
tests/test_ml_model.py — Etapa 5.2: Tests para ml_model.py

Usa datasets sintéticos para verificar:
- PurgedWalkForwardCV genera folds sin solapamiento de etiquetas
- El purge elimina correctamente las observaciones problemáticas
- train_model entrena y devuelve AUC razonable
- predict_proba devuelve probabilidad en [0, 1]
- feature_importance tiene las columnas correctas
- save/load model funciona correctamente
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

from ml_model import (
    PurgedWalkForwardCV,
    evaluate_oos,
    feature_importance,
    load_model,
    predict_proba,
    save_model,
    train_model,
)


# ── Fixtures: dataset sintético ───────────────────────────────────────────────

def _make_synthetic_dataset(n_symbols: int = 5, n_dates: int = 400,
                             seed: int = 0) -> pd.DataFrame:
    """
    Dataset sintético con panel de n_symbols × n_dates observaciones.
    Features aleatorias + target con señal débil (para tests, no para predicción real).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-04", periods=n_dates, freq="B")
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    sectors = ["Tech", "Finance", "Healthcare", "Energy", "Consumer"]

    records = []
    for sym_i, sym in enumerate(symbols):
        sector = sectors[sym_i % len(sectors)]
        # Señal débil: target correlaciona levemente con trend_strength
        trend_vals = rng.uniform(20, 80, size=n_dates)
        for di, date in enumerate(dates):
            feat = {
                "symbol":             sym,
                "sector":             sector,
                "date":               str(date.date()),
                "trend_strength":     trend_vals[di],
                "momentum":           rng.uniform(20, 80),
                "volatility":         rng.uniform(20, 80),
                "volume_score":       rng.uniform(20, 80),
                "price_action":       rng.uniform(20, 80),
                "support_resistance": rng.uniform(20, 80),
                "total_score":        trend_vals[di] * 0.3 + rng.uniform(10, 40),
                "rsi":                rng.uniform(30, 70),
                "atr_pct":            rng.uniform(0.5, 3.0),
                "vol_ratio":          rng.uniform(0.5, 2.0),
                "bb_position":        rng.uniform(20, 80),
                "change_5d":          rng.normal(0.5, 3.0),
                "change_20d":         rng.normal(1.0, 5.0),
                "ema_9_over_21":      int(rng.random() > 0.4),
                "ema_21_over_50":     int(rng.random() > 0.4),
                "spy_above_sma200":   int(rng.random() > 0.3),
                "spy_ret_20d":        rng.normal(1.0, 3.0),
                "spy_slope_20d":      rng.normal(0.01, 0.1),
                "regime_score":       int(rng.integers(1, 6)),
                "vol_zscore_sector":  rng.normal(0, 1),
                "rs_20d_vs_sector":   rng.normal(0, 2),
                "pe_forward":         rng.uniform(10, 40) if rng.random() > 0.3 else np.nan,
                "profit_margin":      rng.uniform(0.05, 0.35) if rng.random() > 0.3 else np.nan,
                "price_to_book":      rng.uniform(1, 8) if rng.random() > 0.3 else np.nan,
                "debt_to_equity":     rng.uniform(20, 150) if rng.random() > 0.4 else np.nan,
                "pe_trailing":        rng.uniform(10, 50) if rng.random() > 0.3 else np.nan,
                # target con señal débil
                "target_20d":         int((trend_vals[di] + rng.normal(0, 30)) > 55),
                "fwd_ret_20d":        rng.normal(2.0, 8.0),
            }
            records.append(feat)

    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def synthetic_df():
    return _make_synthetic_dataset(n_symbols=5, n_dates=400)


FEATURE_COLS = [
    "trend_strength", "momentum", "volatility", "volume_score",
    "price_action", "support_resistance", "total_score",
    "rsi", "atr_pct", "vol_ratio", "bb_position",
    "change_5d", "change_20d", "ema_9_over_21", "ema_21_over_50",
    "spy_above_sma200", "spy_ret_20d", "spy_slope_20d", "regime_score",
    "vol_zscore_sector", "rs_20d_vs_sector",
    "pe_forward", "profit_margin", "price_to_book", "debt_to_equity", "pe_trailing",
]


# ── PurgedWalkForwardCV ───────────────────────────────────────────────────────

def test_cv_generates_folds(synthetic_df):
    cv    = PurgedWalkForwardCV(n_splits=3, test_months=3, purge_days=28)
    folds = list(cv.split(pd.Series(synthetic_df["date"])))
    assert len(folds) >= 1, "Debe generarse al menos un fold"


def test_cv_train_test_no_overlap(synthetic_df):
    """Train y test indices no deben solaparse."""
    cv    = PurgedWalkForwardCV(n_splits=3, test_months=3, purge_days=28)
    dates = pd.Series(synthetic_df["date"])
    for train_idx, test_idx in cv.split(dates):
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"Solapamiento de {len(overlap)} índices"


def test_cv_purge_respected(synthetic_df):
    """
    Ningún índice de train debe tener fecha en [test_start - purge, test_start].
    Verifica que el gap de purge existe entre train y test.
    """
    purge_days = 28
    cv    = PurgedWalkForwardCV(n_splits=3, test_months=3, purge_days=purge_days)
    dates = pd.to_datetime(pd.Series(synthetic_df["date"]))
    for train_idx, test_idx in cv.split(dates):
        test_start  = dates.iloc[test_idx].min()
        train_dates = dates.iloc[train_idx]
        # Ninguna fecha de train debe estar en la zona de purge
        too_close = train_dates[train_dates >= test_start - pd.Timedelta(days=purge_days)]
        assert len(too_close) == 0, (
            f"Purge violado: {len(too_close)} obs de train dentro del gap de purge"
        )


def test_cv_test_window_is_contiguous(synthetic_df):
    """La ventana de test debe ser un rango continuo de fechas."""
    cv    = PurgedWalkForwardCV(n_splits=3, test_months=3, purge_days=28)
    dates = pd.to_datetime(pd.Series(synthetic_df["date"]))
    for _, test_idx in cv.split(dates):
        test_dates = dates.iloc[test_idx].sort_values()
        span_days  = (test_dates.max() - test_dates.min()).days
        # El span no debe ser mayor que ~3 meses + margen
        assert span_days <= 100, f"Test span demasiado largo: {span_days} días"


def test_cv_walk_forward_order(synthetic_df):
    """Los folds deben estar ordenados cronológicamente."""
    cv    = PurgedWalkForwardCV(n_splits=3, test_months=3, purge_days=28)
    dates = pd.to_datetime(pd.Series(synthetic_df["date"]))
    test_starts = []
    for _, test_idx in cv.split(dates):
        test_starts.append(dates.iloc[test_idx].min())
    for i in range(1, len(test_starts)):
        assert test_starts[i] >= test_starts[i - 1], "Folds no ordenados cronológicamente"


def test_cv_empty_dataset_no_folds():
    cv    = PurgedWalkForwardCV(n_splits=5, test_months=6)
    dates = pd.Series(pd.date_range("2024-01-01", periods=10, freq="B"))
    folds = list(cv.split(dates))
    assert len(folds) == 0


# ── train_model ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_model(synthetic_df):
    model, cv_info = train_model(
        synthetic_df,
        feature_cols=FEATURE_COLS,
        n_splits=3,
        test_months=3,
        purge_days=28,
        embargo_days=5,
        num_boost_round=100,
        early_stopping_rounds=20,
    )
    return model, cv_info


def test_train_model_returns_model_and_info(trained_model):
    model, cv_info = trained_model
    assert model is not None
    assert isinstance(cv_info, dict)


def test_train_model_cv_info_keys(trained_model):
    _, cv_info = trained_model
    for key in ("fold_results", "mean_auc", "std_auc", "best_iteration",
                "feature_cols", "positive_rate", "n_samples"):
        assert key in cv_info, f"cv_info missing '{key}'"


def test_train_model_auc_in_range(trained_model):
    _, cv_info = trained_model
    assert 0.4 <= cv_info["mean_auc"] <= 1.0, f"AUC fuera de rango: {cv_info['mean_auc']}"


def test_train_model_fold_results_structure(trained_model):
    _, cv_info = trained_model
    for fold in cv_info["fold_results"]:
        assert "fold" in fold
        assert "auc" in fold
        assert "n_train" in fold
        assert "n_test" in fold
        assert 0 < fold["auc"] <= 1.0


def test_train_model_positive_rate_between_0_1(trained_model):
    _, cv_info = trained_model
    assert 0 < cv_info["positive_rate"] < 1


def test_train_model_feature_cols_match(trained_model):
    _, cv_info = trained_model
    for col in FEATURE_COLS:
        assert col in cv_info["feature_cols"]


# ── predict_proba ─────────────────────────────────────────────────────────────

def test_predict_proba_in_range(trained_model):
    model, cv_info = trained_model
    features = {col: 50.0 for col in cv_info["feature_cols"]}
    prob = predict_proba(model, features, cv_info["feature_cols"])
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_predict_proba_with_nan_features(trained_model):
    """LightGBM maneja NaN nativamente — no debe lanzar excepción."""
    model, cv_info = trained_model
    features = {col: np.nan for col in cv_info["feature_cols"]}
    prob = predict_proba(model, features, cv_info["feature_cols"])
    assert 0.0 <= prob <= 1.0


def test_predict_proba_missing_feature_defaults_nan(trained_model):
    """Features ausentes del dict deben usar NaN por defecto."""
    model, cv_info = trained_model
    # Solo proporcionamos la mitad de las features
    half_features = {col: 50.0 for col in cv_info["feature_cols"][:5]}
    prob = predict_proba(model, half_features, cv_info["feature_cols"])
    assert 0.0 <= prob <= 1.0


# ── feature_importance ────────────────────────────────────────────────────────

def test_feature_importance_columns(trained_model):
    model, cv_info = trained_model
    fi = feature_importance(model, cv_info["feature_cols"])
    assert "feature" in fi.columns
    assert "importance_gain" in fi.columns
    assert "importance_split" in fi.columns
    assert "importance_gain_norm" in fi.columns


def test_feature_importance_sorted_descending(trained_model):
    model, cv_info = trained_model
    fi = feature_importance(model, cv_info["feature_cols"])
    gains = fi["importance_gain"].values
    assert all(gains[i] >= gains[i + 1] for i in range(len(gains) - 1))


def test_feature_importance_normalized_sums_1(trained_model):
    model, cv_info = trained_model
    fi = feature_importance(model, cv_info["feature_cols"])
    total = fi["importance_gain_norm"].sum()
    assert abs(total - 1.0) < 1e-5


def test_feature_importance_all_features_present(trained_model):
    model, cv_info = trained_model
    fi = feature_importance(model, cv_info["feature_cols"])
    assert set(fi["feature"]) == set(cv_info["feature_cols"])


# ── evaluate_oos ──────────────────────────────────────────────────────────────

def test_evaluate_oos_returns_dict(trained_model, synthetic_df):
    model, cv_info = trained_model
    result = evaluate_oos(model, synthetic_df, cv_info["feature_cols"],
                          test_months=3)
    assert isinstance(result, dict)
    assert "auc" in result or "error" in result


def test_evaluate_oos_auc_in_range(trained_model, synthetic_df):
    model, cv_info = trained_model
    result = evaluate_oos(model, synthetic_df, cv_info["feature_cols"],
                          test_months=3)
    if "error" not in result:
        assert 0.0 <= result["auc"] <= 1.0


# ── save / load model ─────────────────────────────────────────────────────────

def test_save_and_load_model(tmp_path, trained_model):
    model, cv_info = trained_model
    save_model(model, cv_info, model_dir=tmp_path)

    assert (tmp_path / "lgbm_model.txt").exists()
    assert (tmp_path / "lgbm_info.json").exists()

    loaded_model, loaded_info = load_model(model_dir=tmp_path)
    assert loaded_model is not None
    assert loaded_info["mean_auc"] == cv_info["mean_auc"]


def test_save_load_predictions_identical(tmp_path, trained_model):
    """El modelo cargado desde disco debe dar las mismas predicciones."""
    model, cv_info = trained_model
    save_model(model, cv_info, model_dir=tmp_path)
    loaded_model, loaded_info = load_model(model_dir=tmp_path)

    features = {col: 50.0 for col in cv_info["feature_cols"]}
    prob_orig   = predict_proba(model,        features, cv_info["feature_cols"])
    prob_loaded = predict_proba(loaded_model, features, loaded_info["feature_cols"])

    assert abs(prob_orig - prob_loaded) < 1e-6


def test_load_model_raises_if_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(model_dir=tmp_path)
