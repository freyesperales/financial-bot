"""
ml_signal.py — Etapa 5.4: Integración de la señal ML en el análisis en vivo.

Enriquece la lista de result dicts producida por StockAnalyzer.analyze_all_stocks()
con la probabilidad ML (ml_prob) y actualiza la confianza de cada símbolo.

Flujo:
  1. Al final de analyze_all_stocks, después de annotate_sector_vol_zscores,
     se llama enrich_ml_signals(results, vix_data, regime).
  2. enrich_ml_signals carga el modelo (lazy, caché en proceso), construye la
     matrix de features para todos los símbolos y llama model.predict() una vez.
  3. Añade ml_prob a cada result dict.
  4. Re-ejecuta compute_confidence para que el nuevo punto ML quede reflejado
     en los campos confidence y confidence_score.

Mapping result_row → feature_names del modelo:
  result_row["volume"]        → "volume_score"  (renombre)
  result_row["vol_zscore"]    → "vol_zscore_sector"
  computed across results     → "rs_20d_vs_sector"
  regime dict                 → "spy_above_sma200", "spy_ret_20d", etc.
  fundamentals cache          → "pe_forward", "profit_margin", etc.
  derived                     → "ema_9_over_21", "ema_21_over_50"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from logging_setup import get_logger

log = get_logger(__name__)

# ── Caché del modelo en proceso (evita I/O repetido) ─────────────────────────
_MODEL_CACHE: dict = {"model": None, "cv_info": None, "loaded": False}


def _try_load_model():
    """Carga el modelo LightGBM si existe, con caché en proceso."""
    if _MODEL_CACHE["loaded"]:
        return _MODEL_CACHE["model"], _MODEL_CACHE["cv_info"]

    _MODEL_CACHE["loaded"] = True  # marcar aunque falle (no reintentar en cada llamada)

    model_path = Path(__file__).resolve().parent / "cache" / "ml" / "lgbm_model.txt"
    if not model_path.exists():
        log.debug("ml_signal: modelo no encontrado en %s — señal ML omitida", model_path)
        return None, None

    try:
        from ml_model import load_model
        model, cv_info = load_model()
        _MODEL_CACHE["model"]   = model
        _MODEL_CACHE["cv_info"] = cv_info
        log.info("ml_signal: modelo cargado (AUC=%.3f, iter=%d)",
                 cv_info.get("mean_auc", 0), cv_info.get("best_iteration", 0))
        return model, cv_info
    except Exception as e:
        log.warning("ml_signal: no se pudo cargar el modelo: %s", e)
        return None, None


def invalidate_model_cache() -> None:
    """Fuerza recarga del modelo en la próxima llamada (útil tras re-entrenamiento)."""
    _MODEL_CACHE["loaded"] = False
    _MODEL_CACHE["model"]  = None
    _MODEL_CACHE["cv_info"] = None


# ── Mapping de features ───────────────────────────────────────────────────────

def _result_to_features(
    row: dict,
    regime: Optional[dict] = None,
    vol_zscore_sector: float = np.nan,
    rs_20d_vs_sector: float = np.nan,
) -> dict:
    """
    Convierte un result dict de StockAnalyzer a un feature dict para el modelo ML.

    Los campos ausentes quedan como NaN — LightGBM los maneja nativamente.
    """
    def _f(key, default=np.nan):
        v = row.get(key)
        if v is None:
            return float(default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    ema_9  = _f("ema_9")
    ema_21 = _f("ema_21")

    # Intenta leer ema_50 del row si el downloader lo almacenó
    ema_50 = _f("ema_50")

    # Regime signals (del dict devuelto por market_regime.get_regime o compute_regime)
    reg_signals = {
        "spy_above_sma200": np.nan,
        "spy_ret_20d":      np.nan,
        "spy_slope_20d":    np.nan,
        "regime_score":     np.nan,
    }
    if regime:
        sig = regime.get("signals", {})
        reg_signals = {
            "spy_above_sma200": float(int(bool(sig.get("above_sma200", False)))),
            "spy_ret_20d":      float(sig.get("ret_20d", np.nan) or np.nan),
            "spy_slope_20d":    float(sig.get("slope", np.nan) or np.nan),
            "regime_score":     float(regime.get("score", np.nan) or np.nan),
        }

    return {
        # 6 sub-scores
        "trend_strength":     _f("trend_strength"),
        "momentum":           _f("momentum"),
        "volatility":         _f("volatility"),
        "volume_score":       _f("volume"),       # renombre
        "price_action":       _f("price_action"),
        "support_resistance": _f("support_resistance"),
        "total_score":        _f("total_score"),
        # indicadores raw
        "rsi":                _f("rsi"),
        "atr_pct":            _f("atr_pct"),
        "vol_ratio":          _f("vol_ratio"),
        "bb_position":        _f("bb_position"),
        "change_5d":          _f("change_5d"),
        "change_20d":         _f("change_20d"),
        "ema_9_over_21":      float(int(ema_9 > ema_21)) if not (np.isnan(ema_9) or np.isnan(ema_21)) else np.nan,
        "ema_21_over_50":     float(int(ema_21 > ema_50)) if not (np.isnan(ema_21) or np.isnan(ema_50)) else np.nan,
        # macro
        **reg_signals,
        # sector relativo
        "vol_zscore_sector":  float(vol_zscore_sector),
        "rs_20d_vs_sector":   float(rs_20d_vs_sector),
        # fundamentales (de fund_data en el result row)
        "pe_forward":         _f("fund_pe"),
        "pe_trailing":        np.nan,
        "price_to_book":      np.nan,
        "profit_margin":      _f("fund_margins"),
        "debt_to_equity":     _f("fund_debt"),
    }


def _annotate_rs_sector(results: list[dict]) -> None:
    """
    Añade 'rs_20d_vs_sector' a cada result: diferencia del change_20d respecto
    a la mediana sectorial. Modifica results in-place.
    """
    from collections import defaultdict
    sector_vals: dict[str, list] = defaultdict(list)
    for r in results:
        s = r.get("sector") or "Unknown"
        v = r.get("change_20d")
        if v is not None:
            try:
                sector_vals[s].append(float(v))
            except (TypeError, ValueError):
                pass

    sector_med: dict[str, float] = {}
    for s, vals in sector_vals.items():
        sector_med[s] = float(np.median(vals)) if vals else 0.0

    for r in results:
        s   = r.get("sector") or "Unknown"
        c20 = r.get("change_20d")
        if c20 is not None:
            try:
                r["rs_20d_vs_sector"] = round(float(c20) - sector_med.get(s, 0.0), 3)
            except (TypeError, ValueError):
                r["rs_20d_vs_sector"] = np.nan
        else:
            r["rs_20d_vs_sector"] = np.nan


# ── Función principal de enriquecimiento ─────────────────────────────────────

def enrich_ml_signals(
    results: list[dict],
    vix_data: Optional[dict] = None,
    regime: Optional[dict] = None,
) -> None:
    """
    Añade 'ml_prob' a cada result dict y actualiza 'confidence' / 'confidence_score'.

    Debe llamarse DESPUÉS de annotate_sector_vol_zscores (que añade 'vol_zscore')
    y ANTES de convertir results a DataFrame.

    Si el modelo no existe, añade ml_prob=None a cada row sin cambiar la confianza.

    Parámetros
    ----------
    results  : lista de result dicts producida por analyze_all_stocks
    vix_data : dict con {'vix': float, 'signal': str} (para recalcular confianza)
    regime   : dict devuelto por market_regime.get_regime() (para features macro)
    """
    if not results:
        return

    model, cv_info = _try_load_model()

    if model is None:
        # Sin modelo: añadir ml_prob=None para que el GUI lo muestre como N/A
        for r in results:
            r["ml_prob"] = None
        return

    feature_cols: list[str] = cv_info.get("feature_cols", [])
    if not feature_cols:
        log.warning("ml_signal: cv_info sin feature_cols — señal ML omitida")
        for r in results:
            r["ml_prob"] = None
        return

    # ── Calcular features sector-relativas para todos los results ─────────────
    _annotate_rs_sector(results)

    # ── Construir matrix de features X ───────────────────────────────────────
    rows_feat = []
    for r in results:
        feat = _result_to_features(
            r,
            regime=regime,
            vol_zscore_sector=r.get("vol_zscore", np.nan),
            rs_20d_vs_sector=r.get("rs_20d_vs_sector", np.nan),
        )
        rows_feat.append([feat.get(c, np.nan) for c in feature_cols])

    X = np.array(rows_feat, dtype=np.float32)

    # ── Inferencia batch ──────────────────────────────────────────────────────
    try:
        probs = model.predict(X)
    except Exception as e:
        log.error("ml_signal: error en model.predict: %s", e)
        for r in results:
            r["ml_prob"] = None
        return

    # ── Añadir ml_prob y re-calcular confianza ────────────────────────────────
    from scoring_service import compute_confidence

    for r, prob in zip(results, probs):
        r["ml_prob"] = round(float(prob), 4)

        # Re-calcular confianza con la señal ML incluida
        conf = compute_confidence(r, vix_data)
        r["confidence"]       = conf["confidence"]
        r["confidence_score"] = conf["score"]
