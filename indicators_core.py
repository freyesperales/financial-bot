"""
indicators_core.py — Funciones puras de indicadores técnicos.

Fuente única de verdad para todos los módulos del proyecto.
Cada función recibe un DataFrame OHLCV y devuelve un DataFrame
con las columnas extra añadidas. No hay estado, no hay side-effects.

Convención de columnas de entrada:
    timestamp, open, high, low, close, volume  (minúsculas)

Convención de columnas de salida (prefijos):
    ema_{span}           → EMA de n períodos
    rsi                  → RSI(14)
    macd, macd_signal, macd_hist
    bb_upper, bb_middle, bb_lower
    atr                  → ATR(14)
    adx, adx_plus_di, adx_minus_di
    vol_ma_{span}        → MA del monto transado (close × volume)
    vol_ratio            → vol_ma_7 / vol_ma_60
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta


# ── EMAs ─────────────────────────────────────────────────────────────────────

def compute_emas(df: pd.DataFrame, spans: tuple[int, ...] = (9, 21, 50)) -> pd.DataFrame:
    """Añade columnas ema_{span} al DataFrame.

    Usa ewm con adjust=False (equivalente a EMA clásica en plataformas de trading).
    """
    out = df.copy()
    for span in spans:
        out[f"ema_{span}"] = out["close"].ewm(span=span, adjust=False).mean()
    return out


# ── RSI ──────────────────────────────────────────────────────────────────────

def compute_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Añade columna `rsi`."""
    out = df.copy()
    out["rsi"] = ta.rsi(out["close"], length=length)
    return out


# ── MACD ─────────────────────────────────────────────────────────────────────

def compute_macd(df: pd.DataFrame,
                 fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Añade columnas `macd`, `macd_signal`, `macd_hist`."""
    out = df.copy()
    result = ta.macd(out["close"], fast=fast, slow=slow, signal=signal)
    if result is not None:
        out["macd"]        = result[f"MACD_{fast}_{slow}_{signal}"]
        out["macd_signal"] = result[f"MACDs_{fast}_{slow}_{signal}"]
        out["macd_hist"]   = result[f"MACDh_{fast}_{slow}_{signal}"]
    else:
        out["macd"] = out["macd_signal"] = out["macd_hist"] = float("nan")
    return out


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def compute_bbands(df: pd.DataFrame,
                   length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Añade columnas `bb_upper`, `bb_middle`, `bb_lower`.

    Se calcula manualmente (rolling mean ± k*std) para evitar discrepancias
    entre versiones de LAPACK en distintos entornos.
    """
    out = df.copy()
    mid = out["close"].rolling(window=length).mean()
    dev = out["close"].rolling(window=length).std()
    out["bb_middle"] = mid
    out["bb_upper"]  = mid + std * dev
    out["bb_lower"]  = mid - std * dev
    return out


# ── ATR ───────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Añade columna `atr`."""
    out = df.copy()
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=length)
    return out


# ── ADX ───────────────────────────────────────────────────────────────────────

def compute_adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Añade columnas `adx`, `adx_plus_di`, `adx_minus_di`."""
    out = df.copy()
    result = ta.adx(out["high"], out["low"], out["close"], length=length)
    if result is not None and not result.empty:
        out["adx"]          = result[f"ADX_{length}"]
        out["adx_plus_di"]  = result[f"DMP_{length}"]
        out["adx_minus_di"] = result[f"DMN_{length}"]
    else:
        out["adx"] = out["adx_plus_di"] = out["adx_minus_di"] = float("nan")
    return out


# ── Volume MA (monto transado) ────────────────────────────────────────────────

def compute_volume_ma(df: pd.DataFrame,
                      spans: tuple[int, ...] = (7, 60)) -> pd.DataFrame:
    """Añade columnas `vol_ma_{span}` y `vol_ratio` (spans[0] / spans[1]).

    Monto transado = close × volume (proxy del flujo de dinero diario).
    """
    out = df.copy()
    out["_monto"] = out["close"] * out["volume"]
    for span in spans:
        out[f"vol_ma_{span}"] = out["_monto"].rolling(window=span).mean()
    out.drop(columns=["_monto"], inplace=True)

    if len(spans) >= 2:
        s0, s1 = spans[0], spans[1]
        denom = out[f"vol_ma_{s1}"]
        out["vol_ratio"] = out[f"vol_ma_{s0}"].div(denom.replace(0, float("nan")))
    return out


# ── All-in-one ────────────────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todos los indicadores estándar en un solo paso.

    Orden: EMA → RSI → MACD → BB → ATR → ADX → Volume MA.
    El DataFrame de salida tiene las columnas del input más todas las de indicadores.
    """
    out = df.copy()
    out = compute_emas(out, spans=(9, 21, 50))
    out = compute_rsi(out)
    out = compute_macd(out)
    out = compute_bbands(out)
    out = compute_atr(out)
    out = compute_adx(out)
    out = compute_volume_ma(out, spans=(7, 60))
    return out
