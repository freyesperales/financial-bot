"""
ml_dataset.py — Etapa 5.1: Construcción del dataset para el modelo ML predictivo.

Genera un DataFrame con features técnicas + macro + sector relativo + fundamentales
y el target binario: retorno forward 20 días > +3 %.

Uso rápido:
    from ml_dataset import load_or_build_dataset
    df = load_or_build_dataset()         # carga caché si existe
    df = load_or_build_dataset(force=True)  # fuerza reconstrucción

Uso avanzado:
    from ml_dataset import build_dataset
    df = build_dataset(db_path='trading_bot.db', lookback_years=3)

Grupos de features:
  técnicas (por símbolo/fecha):
    trend_strength, momentum, volatility, volume_score, price_action,
    support_resistance, total_score, rsi, atr_pct, vol_ratio, bb_position,
    change_5d, change_20d, ema_9_over_21
  macro (SPY en esa fecha, sin lookahead):
    spy_above_sma200, spy_ret_20d, spy_slope_20d, regime_score
  sector relativo (cross-sectional por fecha):
    vol_zscore_sector, rs_20d_vs_sector
  fundamentales (último caché disponible, lagged):
    pe_forward, profit_margin, price_to_book, debt_to_equity
  target:
    target_20d  — 1 si retorno forward 20 días > TARGET_PCT %, else 0
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from database import TradingDatabase
from stock_universe import STOCK_UNIVERSE, get_sector, is_etf
import indicators_core as ic
from scoring_service import compute_subscores, compute_total_score, get_weights

log = logging.getLogger(__name__)

# ── Parámetros por defecto ─────────────────────────────────────────────────────
TARGET_DAYS   = 20       # días hábiles hacia adelante para el target
TARGET_PCT    = 3.0      # retorno mínimo para clase positiva (%)
STEP_DAYS     = 5        # intervalo de muestreo (cada 5 filas ≈ semana)
MIN_HISTORY   = 110      # filas mínimas para compute_subscores (≥ 100)
LOOKBACK_ROWS = 756      # ~3 años de trading days

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "ml_dataset.parquet"
FUND_CACHE = Path(__file__).resolve().parent / "cache" / "fundamentals"

# Features fundamentales a extraer (clave yfinance → nombre interno)
_FUND_MAP = {
    "forwardPE":      "pe_forward",
    "trailingPE":     "pe_trailing",
    "priceToBook":    "price_to_book",
    "profitMargins":  "profit_margin",
    "debtToEquity":   "debt_to_equity",
}


# ── Utilidades ────────────────────────────────────────────────────────────────

def _all_stock_symbols() -> list[str]:
    """Símbolos de renta variable (excluye ETFs)."""
    syms: list[str] = []
    for sector_syms in STOCK_UNIVERSE.values():
        syms.extend(sector_syms)
    return syms


def _load_fund_features(symbol: str) -> dict:
    """Carga features fundamentales desde caché en disco (sin red)."""
    path = FUND_CACHE / f"{symbol.upper()}.json"
    result = {v: np.nan for v in _FUND_MAP.values()}
    if not path.exists():
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for yf_key, feat_name in _FUND_MAP.items():
            val = data.get(yf_key)
            if val is not None:
                try:
                    result[feat_name] = float(val)
                except (TypeError, ValueError):
                    pass
    except Exception:
        pass
    return result


def _prepare_price_df(raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Convierte DataFrame crudo de get_precios a OHLCV + indicadores completos."""
    if raw_df is None or raw_df.empty:
        return None
    df = raw_df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            return None
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["close"]  = pd.to_numeric(df["close"],  errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if len(df) < MIN_HISTORY + TARGET_DAYS:
        return None
    # Recomputa indicadores desde cero (garantiza ema_50, adx, vol_ma_*)
    df = ic.compute_all(df)
    return df


def _features_at_row(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """Extrae features para la observación en row `idx` (sin lookahead)."""
    sub = df.iloc[: idx + 1]
    subscores = compute_subscores(sub, symbol="")
    if subscores is None:
        return None

    weights = get_weights("transition", etf=False)
    total   = compute_total_score(subscores, weights)

    row = sub.iloc[-1]
    ema_9  = subscores.get("ema_9")  or 0.0
    ema_21 = subscores.get("ema_21") or 0.0

    def _ema50() -> Optional[float]:
        v = row.get("ema_50")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)

    ema_50 = _ema50()

    return {
        # 6 subscores
        "trend_strength":     subscores["trend_strength"],
        "momentum":           subscores["momentum"],
        "volatility":         subscores["volatility"],
        "volume_score":       subscores["volume"],
        "price_action":       subscores["price_action"],
        "support_resistance": subscores["support_resistance"],
        "total_score":        total,
        # indicadores raw
        "rsi":                subscores.get("rsi") or 50.0,
        "atr_pct":            subscores.get("atr_pct") or 2.0,
        "vol_ratio":          subscores.get("vol_ratio") or 1.0,
        "bb_position":        subscores.get("bb_position") or 50.0,
        "change_5d":          subscores.get("change_5d") or 0.0,
        "change_20d":         subscores.get("change_20d") or 0.0,
        "ema_9_over_21":      int(ema_9 > ema_21) if ema_21 else 0,
        "ema_21_over_50":     int(ema_21 > ema_50) if ema_50 else 0,
        # metadatos internos (prefijo _ para limpiar después)
        "_price":             subscores.get("price") or float(row.get("close", 0)),
        "_ret_20d_raw":       subscores.get("change_20d") or 0.0,
    }


# ── Régimen macro histórico ───────────────────────────────────────────────────

def _build_spy_regime_map(db: TradingDatabase) -> dict[str, dict]:
    """
    Construye un mapa {date_str → dict de señales macro} usando SPY de la DB.

    Cada entrada del mapa contiene:
        spy_above_sma200  — 0/1
        spy_ret_20d       — retorno porcentual 20d
        spy_slope_20d     — diferencia (%) de SMA200 vs 20d antes
        regime_score      — 0-6 aprox. (señales sumadas)
    """
    raw = db.get_precios("SPY")
    if raw is None or raw.empty:
        return {}

    df = raw.sort_values("timestamp").reset_index(drop=True)
    df = ic.compute_emas(df, spans=(9, 21, 50))
    closes = df["close"].astype(float).values
    dates  = pd.to_datetime(df["timestamp"]).dt.date
    n      = len(closes)

    regime_map: dict[str, dict] = {}
    for i in range(200, n):
        sma200      = float(np.mean(closes[i - 199: i + 1]))
        above_sma   = int(closes[i] > sma200)
        ret_20d     = (closes[i] / closes[i - 20] - 1) * 100 if i >= 20 else 0.0

        slope = 0.0
        if i >= 220:
            sma200_prev = float(np.mean(closes[i - 219: i - 19]))
            slope = (sma200 / sma200_prev - 1) * 100 if sma200_prev else 0.0

        # Score simplificado (sin VIX disponible en DB)
        score = above_sma
        if ret_20d > 0:
            score += 1
        if ret_20d > 2:
            score += 1
        if slope > 0:
            score += 1

        regime_map[str(dates[i])] = {
            "spy_above_sma200": above_sma,
            "spy_ret_20d":      round(ret_20d, 3),
            "spy_slope_20d":    round(slope, 4),
            "regime_score":     score,
        }

    return regime_map


# ── Construcción del dataset ──────────────────────────────────────────────────

def build_dataset(
    db_path: str = "trading_bot.db",
    symbols: Optional[list[str]] = None,
    lookback_years: float = 3.0,
    step_days: int = STEP_DAYS,
    target_days: int = TARGET_DAYS,
    target_pct: float = TARGET_PCT,
) -> pd.DataFrame:
    """
    Construye el dataset ML desde la base de datos local.

    Parámetros
    ----------
    db_path       : ruta a trading_bot.db
    symbols       : lista de símbolos (None = todas las acciones)
    lookback_years: años de historia hacia atrás desde hoy
    step_days     : frecuencia de muestreo (filas entre observaciones)
    target_days   : horizonte en días hábiles para el target
    target_pct    : umbral de retorno (%) para clase positiva

    Retorna
    -------
    pd.DataFrame con una fila por (symbol, date) y columnas de features + target_20d.
    Las filas sin target válido (última ventana) son excluidas.
    """
    db = TradingDatabase(db_path, auto_cleanup_snapshots=False)

    if symbols is None:
        symbols = _all_stock_symbols()

    log.info("build_dataset: %d símbolos, step=%d, target=%dd>%.1f%%",
             len(symbols), step_days, target_days, target_pct)

    # ── 1. Régimen macro (SPY) ────────────────────────────────────────────────
    spy_map = _build_spy_regime_map(db)
    if not spy_map:
        log.warning("SPY no encontrado en DB — features macro serán NaN")

    # ── 2. Fundamentales (estáticos por símbolo) ─────────────────────────────
    fund_map: dict[str, dict] = {}
    for sym in symbols:
        fund_map[sym] = _load_fund_features(sym)

    # ── 3. Features por símbolo y fecha ──────────────────────────────────────
    lookback_rows = int(lookback_years * 252)
    records: list[dict] = []

    for sym in symbols:
        raw = db.get_precios(sym)
        df  = _prepare_price_df(raw)
        if df is None:
            log.debug("Sin datos suficientes para %s", sym)
            continue

        # Recortar a lookback_rows + margen
        max_rows = lookback_rows + target_days + MIN_HISTORY + 10
        if len(df) > max_rows:
            df = df.iloc[-max_rows:].reset_index(drop=True)

        sector = get_sector(sym) or "Unknown"
        closes = df["close"].astype(float).values
        dates  = pd.to_datetime(df["timestamp"]).dt.date
        n      = len(df)

        # Punto de inicio: necesitamos MIN_HISTORY filas hacia atrás
        start_idx = MIN_HISTORY - 1
        # Punto de fin: necesitamos target_days filas hacia adelante
        end_idx = n - target_days - 1

        for idx in range(start_idx, end_idx + 1, step_days):
            feats = _features_at_row(df, idx)
            if feats is None:
                continue

            # Target: retorno forward
            fwd_close = closes[idx + target_days]
            cur_close = closes[idx]
            if cur_close <= 0 or fwd_close <= 0:
                continue
            fwd_ret = (fwd_close / cur_close - 1) * 100.0
            target  = int(fwd_ret > target_pct)

            date_str = str(dates[idx])

            # Régimen macro
            macro = spy_map.get(date_str, {
                "spy_above_sma200": np.nan,
                "spy_ret_20d":      np.nan,
                "spy_slope_20d":    np.nan,
                "regime_score":     np.nan,
            })

            rec = {
                "symbol":  sym,
                "sector":  sector,
                "date":    date_str,
                # técnicas
                **{k: v for k, v in feats.items() if not k.startswith("_")},
                # macro
                **macro,
                # fundamentales
                **fund_map[sym],
                # target
                "target_20d":  target,
                "fwd_ret_20d": round(fwd_ret, 4),
            }
            records.append(rec)

    if not records:
        log.warning("build_dataset: 0 observaciones generadas")
        return pd.DataFrame()

    df_out = pd.DataFrame(records)

    # ── 4. Features sector-relativas (cross-sectional por fecha) ─────────────
    df_out = _add_sector_features(df_out)

    df_out = df_out.sort_values(["symbol", "date"]).reset_index(drop=True)

    log.info("build_dataset: %d observaciones (%d símbolos, %d fechas únicas)",
             len(df_out),
             df_out["symbol"].nunique(),
             df_out["date"].nunique())
    return df_out


def _add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade vol_zscore_sector y rs_20d_vs_sector calculados cross-seccionally
    por (sector, date) para evitar lookahead.
    """
    if df.empty:
        return df

    out = df.copy()

    # vol_zscore dentro de sector y fecha
    def _vol_zscore(g: pd.DataFrame) -> pd.Series:
        mu  = g["vol_ratio"].mean()
        std = g["vol_ratio"].std()
        if std and std > 0:
            return (g["vol_ratio"] - mu) / std
        return pd.Series(0.0, index=g.index)

    out["vol_zscore_sector"] = (
        out.groupby(["sector", "date"], group_keys=False)["vol_ratio"]
           .transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    )

    # RS 20d vs sector: change_20d - mediana sectorial
    out["rs_20d_vs_sector"] = (
        out.groupby(["sector", "date"], group_keys=False)["change_20d"]
           .transform(lambda x: x - x.median())
    )

    return out


# ── Caché en parquet ──────────────────────────────────────────────────────────

def load_or_build_dataset(
    db_path: str = "trading_bot.db",
    force: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Carga el dataset desde caché parquet si existe, o lo construye desde cero.

    Parámetros
    ----------
    db_path : ruta a trading_bot.db
    force   : si True, ignora la caché y reconstruye
    **kwargs: se pasan a build_dataset()
    """
    if not force and CACHE_PATH.exists():
        try:
            df = pd.read_parquet(CACHE_PATH)
            log.info("Dataset cargado desde caché: %s (%d filas)", CACHE_PATH, len(df))
            return df
        except Exception as e:
            log.warning("Error leyendo caché, reconstruyendo: %s", e)

    df = build_dataset(db_path=db_path, **kwargs)

    if not df.empty:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_PATH, index=False)
        log.info("Dataset guardado en %s", CACHE_PATH)

    return df


# ── Utilidades de inspección ──────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> dict:
    """Devuelve métricas de resumen del dataset."""
    if df.empty:
        return {}
    pos_rate = df["target_20d"].mean() if "target_20d" in df.columns else float("nan")
    return {
        "n_rows":       len(df),
        "n_symbols":    df["symbol"].nunique(),
        "n_dates":      df["date"].nunique(),
        "date_range":   f"{df['date'].min()} → {df['date'].max()}",
        "positive_rate": round(pos_rate, 3),
        "sectors":      df["sector"].nunique() if "sector" in df.columns else 0,
        "feature_cols": [c for c in df.columns
                         if c not in ("symbol", "sector", "date",
                                      "target_20d", "fwd_ret_20d")],
    }


def feature_columns(df: Optional[pd.DataFrame] = None) -> list[str]:
    """Lista de columnas de features (excluye metadatos y target)."""
    exclude = {"symbol", "sector", "date", "target_20d", "fwd_ret_20d"}
    if df is not None:
        return [c for c in df.columns if c not in exclude]
    # Columnas esperadas cuando df no está disponible
    return [
        "trend_strength", "momentum", "volatility", "volume_score",
        "price_action", "support_resistance", "total_score",
        "rsi", "atr_pct", "vol_ratio", "bb_position",
        "change_5d", "change_20d", "ema_9_over_21", "ema_21_over_50",
        "spy_above_sma200", "spy_ret_20d", "spy_slope_20d", "regime_score",
        "vol_zscore_sector", "rs_20d_vs_sector",
        "pe_forward", "pe_trailing", "price_to_book", "profit_margin", "debt_to_equity",
    ]
