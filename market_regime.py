"""
market_regime.py — Detector de régimen de mercado.

Régimen    | Score | Descripción
-----------|-------|---------------------------------------------
risk_on    |  5-6  | Tendencia alcista sana, volatilidad baja
transition |  3-4  | Consolidación o giro de tendencia
risk_off   |  0-2  | Tendencia bajista o volatilidad elevada
panic      |  —    | Override: VIX > 35 o SPY 5d < -5%

Señales ponderadas (máx. 6 puntos):
  +1  SPY cotiza sobre SMA200
  +1  Slope SMA200 positivo (comparando ventana de 200d hace 20 días vs ahora)
  +1  SPY retorno 20d > 0%
  +1  SPY retorno 20d > +2%  (bonus de convicción)
  +1  ADX > 20 con DI+ > DI-  (tendencia alcista confirmada)
  +1  VIX ≤ 20  (+2 si VIX ≤ 15, pero score total cap a 6)

Panic override: VIX > 35 o caída SPY 5d < -5%.

Caché en memoria TTL 1h para no picar la red en cada análisis.

Uso:
    from market_regime import get_regime, compute_regime
    r = get_regime()        # con caché + red
    r = compute_regime(spy_df, vix=18.5)  # sin red — para tests
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

import indicators_core as ic
from logging_setup import get_logger

log = get_logger(__name__)

# ── Caché en memoria ──────────────────────────────────────────────────────────
_cache: dict = {'result': None, 'ts': None}
_CACHE_TTL_MINUTES = 60


def _cache_fresh() -> bool:
    if _cache['ts'] is None or _cache['result'] is None:
        return False
    return (datetime.now() - _cache['ts']) < timedelta(minutes=_CACHE_TTL_MINUTES)


# ── Descarga de datos ─────────────────────────────────────────────────────────

def _fetch_spy() -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.download('SPY', period='2y', interval='1d', progress=False,
                         auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df = df.sort_index().reset_index()
        # yfinance devuelve columna 'Date' o 'Datetime' según la versión
        for ts_col in ('date', 'datetime', 'index'):
            if ts_col in df.columns:
                df = df.rename(columns={ts_col: 'timestamp'})
                break
        return df
    except Exception as e:
        log.warning("No se pudo descargar SPY para market_regime: %s", e)
        return None


def _fetch_vix() -> Optional[float]:
    try:
        import yfinance as yf
        df = yf.download('^VIX', period='5d', interval='1d', progress=False,
                         auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return float(df['close'].dropna().iloc[-1])
    except Exception as e:
        log.warning("No se pudo descargar VIX: %s", e)
        return None


# ── Lógica de régimen (sin I/O — testable) ────────────────────────────────────

def compute_regime(spy_df: pd.DataFrame, vix: Optional[float] = None) -> dict:
    """
    Calcula el régimen de mercado a partir de un DataFrame SPY y el VIX actual.

    spy_df debe tener columna 'close' (y opcionalmente 'high', 'low').
    No hace llamadas de red — idóneo para tests con datos sintéticos.

    Devuelve:
        regime       — 'risk_on' | 'transition' | 'risk_off' | 'panic'
        score        — int 0-6
        signals      — dict con señales individuales y sus valores
        computed_at  — ISO timestamp
    """
    df = spy_df.copy()

    # Normalizar nombres de columnas
    if 'Close' in df.columns:
        df = df.rename(columns={
            'Close': 'close', 'High': 'high', 'Low': 'low',
            'Open': 'open', 'Volume': 'volume',
        })
    if 'close' not in df.columns:
        raise ValueError("spy_df debe tener columna 'close'")

    close = df['close'].astype(float).values
    n     = len(close)
    score = 0
    sig: dict = {}

    # ── 1. Precio sobre SMA200 ────────────────────────────────────────────────
    if n >= 200:
        sma200 = float(np.mean(close[-200:]))
        above  = bool(close[-1] > sma200)
        sig['sma200']        = round(sma200, 2)
        sig['above_sma200']  = above
        if above:
            score += 1
    else:
        sig['above_sma200'] = None

    # ── 2. Slope SMA200 (20d atrás) ──────────────────────────────────────────
    if n >= 220:
        sma200_now  = float(np.mean(close[-200:]))
        sma200_prev = float(np.mean(close[-220:-20]))
        slope_pct   = (sma200_now - sma200_prev) / sma200_prev * 100
        sig['sma200_slope_pct']      = round(slope_pct, 3)
        sig['sma200_slope_positive'] = bool(slope_pct > 0)
        if slope_pct > 0:
            score += 1
    else:
        sig['sma200_slope_positive'] = None

    # ── 3-4. Retorno 20d ─────────────────────────────────────────────────────
    if n >= 21:
        ret_20d = (close[-1] / close[-21] - 1) * 100
        sig['ret_20d'] = round(ret_20d, 2)
        if ret_20d > 0:
            score += 1
        if ret_20d > 2:
            score += 1  # bonus de convicción
    else:
        sig['ret_20d'] = None

    # ── 5. ADX con DI+ > DI- ─────────────────────────────────────────────────
    if n >= 60:
        sub = df.tail(60).reset_index(drop=True)
        # compute_adx necesita columnas high/low; si no existen, genera aproximación
        if 'high' not in sub.columns:
            sub['high'] = sub['close']
            sub['low']  = sub['close']
        try:
            sub_adx = ic.compute_adx(sub)
            last    = sub_adx.iloc[-1]
            adx_v   = None if pd.isna(last.get('adx', float('nan')))      else float(last['adx'])
            pdi     = None if pd.isna(last.get('adx_plus_di', float('nan'))) else float(last['adx_plus_di'])
            mdi     = None if pd.isna(last.get('adx_minus_di', float('nan'))) else float(last['adx_minus_di'])
            if adx_v is not None and pdi is not None and mdi is not None:
                sig['adx']         = round(adx_v, 1)
                sig['adx_plus_di'] = round(pdi, 1)
                sig['adx_minus_di']= round(mdi, 1)
                sig['adx_bullish'] = bool(adx_v > 20 and pdi > mdi)
                if sig['adx_bullish']:
                    score += 1
        except Exception:
            pass

    # ── 6. VIX ───────────────────────────────────────────────────────────────
    vix_level = None
    if vix is not None:
        sig['vix'] = round(float(vix), 1)
        if vix <= 15:
            score += 2
            vix_level = 'bajo'
        elif vix <= 20:
            score += 1
            vix_level = 'bajo'
        elif vix <= 30:
            vix_level = 'medio'
        elif vix <= 35:
            vix_level = 'alto'
        else:
            vix_level = 'extremo'
        sig['vix_level'] = vix_level

    score = min(score, 6)

    # ── Retorno 5d (para panic override) ─────────────────────────────────────
    if n >= 6:
        ret_5d = (close[-1] / close[-6] - 1) * 100
        sig['ret_5d'] = round(ret_5d, 2)
    else:
        ret_5d = None

    # ── Clasificar régimen ───────────────────────────────────────────────────
    panic = (
        (vix is not None and vix > 35) or
        (ret_5d is not None and ret_5d < -5)
    )

    if panic:
        regime = 'panic'
    elif score >= 5:
        regime = 'risk_on'
    elif score >= 3:
        regime = 'transition'
    else:
        regime = 'risk_off'

    return {
        'regime':      regime,
        'score':       score,
        'signals':     sig,
        'computed_at': datetime.now().isoformat(),
    }


# ── API pública con caché ─────────────────────────────────────────────────────

def get_regime(force_refresh: bool = False) -> dict:
    """
    Devuelve el régimen de mercado actual (caché 1h).

    Si la red no está disponible y no hay caché válida, devuelve un régimen
    neutro ('transition', score=3) marcado con 'fallback': True para que el
    consumidor pueda decidir si actuar sobre él.
    """
    if not force_refresh and _cache_fresh():
        log.debug("market_regime cache HIT")
        return _cache['result']  # type: ignore[return-value]

    log.debug("market_regime cache MISS — descargando SPY + VIX")
    spy_df = _fetch_spy()
    vix    = _fetch_vix()

    if spy_df is None:
        log.warning("SPY no disponible — régimen por defecto: transition")
        return {
            'regime': 'transition',
            'score':   3,
            'signals': {},
            'computed_at': datetime.now().isoformat(),
            'fallback': True,
        }

    result = compute_regime(spy_df, vix)
    _cache['result'] = result
    _cache['ts']     = datetime.now()
    log.info("Régimen de mercado: %s (score=%d)", result['regime'], result['score'])
    return result
