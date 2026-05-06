"""
scoring_service.py — Lógica de scoring separada de la presentación.

Contiene tres familias de funciones:

1. SUBSCORES TÉCNICOS (ya venían en StockAnalyzer.calculate_technical_score)
   compute_subscores(df, symbol, weights) → dict de sub-scores 0-100
   compute_total_score(subscores, weights) → float 0-100

2. BUY / DIP / BIGDIP / BOND SCORES (venían en InvestmentReportGUI)
   compute_buy_score(row) → float 0-100
   compute_bond_score(row) → float 0-100
   compute_dip_score(row) → float 0-100
   compute_bigdip_score(row) → float 0-100

3. CONFIANZA COMBINADA (venía en StockAnalyzer.calculate_signal_confidence)
   compute_confidence(result_row, vix_data) → dict

SUAVIZADO DE CLIFFS
Todos los sub-scores que antes usaban bandas duras (if 40<=rsi<=60: 100, elif ...)
ahora usan `_lerp` (interpolación lineal por tramos) para que no haya saltos bruscos
en el borde de un umbral.
"""
from __future__ import annotations

import math
from typing import Optional


# ── Utilidades ───────────────────────────────────────────────────────────────

def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Interpolación lineal de [x0, x1] a [y0, y1]. Fuera de rango: clamped."""
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


def _piecewise(x: float, knots: list[tuple[float, float]]) -> float:
    """Tramos lineales definidos por puntos (x_i, y_i) ordenados por x.

    Para x < knots[0][0]  → devuelve knots[0][1]
    Para x > knots[-1][0] → devuelve knots[-1][1]
    En medio → interpola linealmente entre el tramo correspondiente.
    """
    if x <= knots[0][0]:
        return knots[0][1]
    if x >= knots[-1][0]:
        return knots[-1][1]
    for i in range(len(knots) - 1):
        x0, y0 = knots[i]
        x1, y1 = knots[i + 1]
        if x0 <= x <= x1:
            return _lerp(x, x0, x1, y0, y1)
    return knots[-1][1]


# ── Sub-scores técnicos ───────────────────────────────────────────────────────

def _score_trend(ema_9: float, ema_21: float, ema_50: float) -> float:
    """Tendencia via EMAs.  Antes: if/elif con valores discretos 0/25/50/75/100."""
    if ema_9 > ema_21 > ema_50:    return 100.0
    if ema_9 > ema_21:             return 75.0
    if ema_21 > ema_50:            return 50.0
    if ema_9 < ema_21 < ema_50:   return 0.0
    if ema_9 < ema_21:             return 25.0
    return 50.0


def _score_rsi(rsi: float) -> float:
    """RSI → score 0-100 con curva continua.

    Antes había 5 bandas duras. Ahora se interpola suavemente:
      RSI  0  → 45   (oversold extremo — puede ser trampa bajista)
      RSI 30  → 60
      RSI 40  → 100  (inicio zona óptima de entrada)
      RSI 60  → 100  (fin zona óptima)
      RSI 70  → 65
      RSI 85  → 20
      RSI 100 → 20
    """
    return _piecewise(rsi, [
        (0,   45),
        (30,  60),
        (40, 100),
        (60, 100),
        (70,  65),
        (85,  20),
        (100, 20),
    ])


def _score_macd(macd_diff: float, macd_prev_diff: float) -> float:
    """MACD diferencial → score 0-100. Se mantiene discreto (4 estados claros)."""
    if macd_diff > 0 and macd_diff > macd_prev_diff:   return 100.0
    if macd_diff > 0:                                   return 75.0
    if macd_diff < 0 and macd_diff > macd_prev_diff:   return 50.0
    return 25.0


def _score_bb(bb_position: float) -> float:
    """Posición en Bollinger Bands (0-100, donde 50 = middle).

    Antes: bandas duras a 20/40/60/80.
    Ahora: curva en forma de campana con máximo en 50 (precio en medio = ideal).
    """
    return _piecewise(bb_position, [
        (0,   50),
        (20,  75),
        (40, 100),
        (60, 100),
        (80,  75),
        (100, 50),
    ])


def _score_atr(atr_pct: float) -> float:
    """ATR como % del precio → score 0-100 (menor volatilidad = mejor para inversión).

    Antes: bandas duras <2/4/6 → 100/75/50/25.
    Ahora: curva continua descendente.
    """
    return _piecewise(atr_pct, [
        (0,  100),
        (2,  100),
        (4,   75),
        (6,   45),
        (10,  10),
        (20,   0),
    ])


def _score_volume(vol_ratio: float) -> float:
    """Ratio volumen actual vs media 20d → score 0-100.

    Antes: 4 bandas duras 0.8/1.2/1.5 → 25/50/75/100.
    Ahora: interpolación lineal continua.
    """
    return _piecewise(vol_ratio, [
        (0.0,  10),
        (0.8,  25),
        (1.0,  50),
        (1.2,  75),
        (1.5, 100),
        (3.0, 100),
    ])


def _score_price_action(change_5d: float, change_20d: float) -> float:
    """Momentum de precio reciente → score 0-100 promediando 5d y 20d."""
    s5 = _piecewise(change_5d, [
        (-15,   0),
        (-5,   20),
        (0,    40),
        (2,    60),
        (5,    85),
        (10,  100),
        (20,  100),
    ])
    s20 = _piecewise(change_20d, [
        (-20,   0),
        (-5,   20),
        (0,    45),
        (5,    70),
        (10,   90),
        (20,  100),
        (40,  100),
    ])
    return (s5 + s20) / 2.0


def _score_support_resistance(distance_to_high: float) -> float:
    """Distancia % al máximo 52 semanas → score 0-100 (lejos = más espacio).

    Antes: 4 bandas duras >5/10/20 → 25/50/75/100.
    Ahora: interpolación continua.
    """
    return _piecewise(distance_to_high, [
        (0,    0),
        (5,   25),
        (10,  55),
        (20,  85),
        (35, 100),
        (100,100),
    ])


def compute_subscores(
    df,
    symbol: str,
) -> Optional[dict]:
    """Calcula los seis sub-scores técnicos a partir del DataFrame con indicadores.

    Requiere que el DataFrame ya tenga las columnas de indicators_core.compute_all().
    Devuelve None si los datos son insuficientes.
    """
    import pandas as pd

    if len(df) < 100:
        return None

    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)

    current = df.iloc[-1]
    prev    = df.iloc[-2]

    def _safe(col):
        v = current.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)

    ema_9  = _safe("ema_9")
    ema_21 = _safe("ema_21")
    ema_50 = _safe("ema_50")
    rsi    = _safe("rsi")
    macd   = _safe("macd")
    msig   = _safe("macd_signal")
    bb_up  = _safe("bb_upper")
    bb_lo  = _safe("bb_lower")
    price  = _safe("close")
    atr    = _safe("atr")

    if any(v is None for v in (ema_9, ema_21, ema_50, rsi, price)):
        return None

    # Trend
    trend_score = _score_trend(ema_9, ema_21, ema_50)

    # Momentum (RSI + MACD)
    rsi_score = _score_rsi(rsi)
    if macd is not None and msig is not None:
        prev_macd = float(prev.get("macd") or 0)
        prev_msig = float(prev.get("macd_signal") or 0)
        macd_diff      = macd - msig
        macd_prev_diff = prev_macd - prev_msig
        macd_score = _score_macd(macd_diff, macd_prev_diff)
    else:
        macd_score = 50.0
    momentum_score = (rsi_score + macd_score) / 2.0

    # Volatility (BB + ATR)
    if bb_up is not None and bb_lo is not None and (bb_up - bb_lo) > 0:
        bb_pos = (price - bb_lo) / (bb_up - bb_lo) * 100.0
        bb_score = _score_bb(bb_pos)
    else:
        bb_pos = 50.0
        bb_score = 50.0
    atr_pct = (atr / price * 100.0) if atr is not None and price > 0 else 2.0
    atr_score = _score_atr(atr_pct)
    volatility_score = bb_score * 0.6 + atr_score * 0.4

    # Volume
    vol_20 = df["volume"].rolling(20).mean().iloc[-1]
    vol_current = float(current.get("volume") or 0)
    vol_ratio = vol_current / vol_20 if vol_20 and vol_20 > 0 else 1.0
    volume_score = _score_volume(vol_ratio)

    # Price action
    try:
        change_5d  = (price / float(df.iloc[-6]["close"]) - 1) * 100
    except (IndexError, ZeroDivisionError):
        change_5d = 0.0
    try:
        change_20d = (price / float(df.iloc[-21]["close"]) - 1) * 100
    except (IndexError, ZeroDivisionError):
        change_20d = 0.0
    price_action_score = _score_price_action(change_5d, change_20d)

    # Support / Resistance (52w high)
    high_52w = df["high"].rolling(252).max().iloc[-1]
    if high_52w and high_52w > 0:
        distance_to_high = (high_52w - price) / price * 100.0
    else:
        distance_to_high = 20.0
    sr_score = _score_support_resistance(distance_to_high)

    return {
        "trend_strength":    trend_score,
        "momentum":          momentum_score,
        "volatility":        volatility_score,
        "volume":            volume_score,
        "price_action":      price_action_score,
        "support_resistance": sr_score,
        # extras para el resultado final
        "price":             price,
        "rsi":               rsi,
        "atr_pct":           atr_pct,
        "vol_ratio":         vol_ratio,
        "bb_position":       bb_pos,
        "bb_lower":          bb_lo,
        "bb_upper":          bb_up,
        "ema_9":             ema_9,
        "ema_21":            ema_21,
        "ema_50":            ema_50,
        "change_5d":         change_5d,
        "change_20d":        change_20d,
        "support_52w":       float(df["low"].rolling(252).min().iloc[-1]),
        "resistance_52w":    float(high_52w) if high_52w else price,
    }


def compute_total_score(subscores: dict, weights: dict) -> float:
    """Pondera los 6 sub-scores según weights. Devuelve float 0-100."""
    keys = ("trend_strength", "momentum", "volatility",
            "volume", "price_action", "support_resistance")
    return sum(subscores.get(k, 0) * weights.get(k, 0) for k in keys)


# ── Pesos adaptativos por régimen de mercado ──────────────────────────────────

# En risk_on los factores de tendencia/momentum tienen más poder predictivo.
# En risk_off y panic, soporte/resistencia y volatilidad se vuelven clave.
# ETF equity: menor peso a S/R, mayor peso a tendencia (índices son más limpios).
# ETF fixed-income: la lógica se maneja aparte en compute_bond_score.

_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    'risk_on': {
        'trend_strength':    0.30,
        'momentum':          0.25,
        'volatility':        0.10,
        'volume':            0.15,
        'price_action':      0.15,
        'support_resistance':0.05,
    },
    'transition': {          # = defaults actuales
        'trend_strength':    0.25,
        'momentum':          0.20,
        'volatility':        0.15,
        'volume':            0.15,
        'price_action':      0.15,
        'support_resistance':0.10,
    },
    'risk_off': {
        'trend_strength':    0.20,
        'momentum':          0.15,
        'volatility':        0.20,
        'volume':            0.15,
        'price_action':      0.10,
        'support_resistance':0.20,
    },
    'panic': {
        'trend_strength':    0.15,
        'momentum':          0.10,
        'volatility':        0.25,
        'volume':            0.15,
        'price_action':      0.15,
        'support_resistance':0.20,
    },
}

_REGIME_WEIGHTS_ETF: dict[str, dict[str, float]] = {
    'risk_on': {
        'trend_strength':    0.35,
        'momentum':          0.30,
        'volatility':        0.10,
        'volume':            0.15,
        'price_action':      0.07,
        'support_resistance':0.03,
    },
    'transition': {
        'trend_strength':    0.30,
        'momentum':          0.25,
        'volatility':        0.15,
        'volume':            0.15,
        'price_action':      0.10,
        'support_resistance':0.05,
    },
    'risk_off': {
        'trend_strength':    0.25,
        'momentum':          0.15,
        'volatility':        0.20,
        'volume':            0.15,
        'price_action':      0.10,
        'support_resistance':0.15,
    },
    'panic': {
        'trend_strength':    0.20,
        'momentum':          0.10,
        'volatility':        0.25,
        'volume':            0.20,
        'price_action':      0.10,
        'support_resistance':0.15,
    },
}


def get_weights(regime: str = 'transition', etf: bool = False) -> dict:
    """Devuelve el dict de pesos adecuado para el régimen y tipo de activo.

    Parámetros:
        regime — 'risk_on' | 'transition' | 'risk_off' | 'panic'
        etf    — True para ETFs de renta variable (usa tabla _REGIME_WEIGHTS_ETF)
    """
    table = _REGIME_WEIGHTS_ETF if etf else _REGIME_WEIGHTS
    return table.get(regime, table['transition'])


# ── Buy / Dip / BigDip / Bond scores (trasladados desde el GUI) ──────────────

_BOND_ETFS = frozenset({"TLT", "AGG", "HYG", "TIP"})


def compute_buy_score(row: dict) -> float:
    """Score 0-100 orientado a mediano plazo (objetivo >15%/año).

    Trasladado de InvestmentReportGUI._compute_buy_score.
    """
    symbol = row.get("symbol", "")
    if symbol in _BOND_ETFS:
        return compute_bond_score(row)

    score = 50.0

    # 1) Confianza técnica (±10 pts)
    conf = row.get("confidence_score", 5) or 5
    score += (conf - 5) * 2

    # 2) Vol MA — flujo de dinero (±14 pts)
    vol   = row.get("vol_signal", "")
    ratio = row.get("vol_ratio") or 1.0
    if vol == "COMPRAR":
        score += 10 + min((ratio - 1) * 15, 4)
    elif vol == "VENDER":
        score -= 14

    # 3) Multi-TF alignment (±8 pts)
    tf_pts = row.get("tf_conf_pts", 0) or 0
    score += tf_pts * 4

    # 4) RS 252d vs SPY (±16 pts) — suavizado continuo
    rs_252 = row.get("rs_252d")
    if rs_252 is not None:
        score += _piecewise(rs_252, [
            (-30, -16), (-15, -10), (-5, -4),
            (0,    4),  (5,   10),  (15,  16), (30, 16),
        ])

    # 5) Riesgo / Sharpe (±8 pts)
    risk   = row.get("risk_signal", "")
    sharpe = row.get("sharpe")
    if "BAJO" in risk:   score += 3
    elif "ALTO" in risk: score -= 3
    if sharpe is not None:
        score += _piecewise(sharpe, [
            (-2, -5), (0, -5), (0.5, 2), (1.5, 5), (3, 5),
        ])

    # 6) Fundamentales (±18 pts)
    fund_score    = row.get("fund_score", 50) or 50
    sector        = row.get("sector", "")
    fund_pe       = row.get("fund_pe")
    _commodity    = any(s in sector for s in ("Energy", "Materials"))
    _pe_negative  = fund_pe is None or (isinstance(fund_pe, (int, float)) and fund_pe <= 0)
    if _commodity and _pe_negative and fund_score < 50:
        fund_score = 50
    score += (fund_score - 50) / 50 * 18

    # 7) Retorno anualizado histórico vs objetivo 15%/año (±12 pts) — suavizado
    ann_ret = row.get("ann_return")
    if ann_ret is not None:
        score += _piecewise(ann_ret, [
            (-30, -10), (0, -10), (5, -4), (10, 0),
            (15, 4), (20, 8), (30, 12), (60, 12),
        ])

    # 8) Predicción (±8 pts)
    pred = row.get("pred_signal", "NEUTRO")
    _pred_pts = {"SUBE FUERTE": 8, "SUBE": 4, "NEUTRO": 0, "BAJA": -4, "BAJA FUERTE": -8}
    score += _pred_pts.get(pred, 0)

    # 9) Backtesting win rate Téc+Vol (±6 pts)
    wr = row.get("bt_win_rate_vol")
    if wr is not None:
        score += _piecewise(wr, [
            (0, -6), (40, -6), (50, -2), (55, 2), (65, 6), (100, 6),
        ])

    return round(min(max(score, 0.0), 100.0), 1)


def compute_bond_score(row: dict) -> float:
    """Score específico para bond ETFs. Trasladado de InvestmentReportGUI._compute_bond_score."""
    score = 50.0

    tf_pts = row.get("tf_conf_pts", 0) or 0
    score += tf_pts * 5

    vol   = row.get("vol_signal", "")
    ratio = row.get("vol_ratio") or 1.0
    if vol == "COMPRAR":
        score += 8 + min((ratio - 1) * 10, 4)
    elif vol == "VENDER":
        score -= 10

    sharpe = row.get("sharpe")
    if sharpe is not None:
        score += _piecewise(sharpe, [(-2, -8), (0, -8), (0.3, 5), (1.0, 10), (3, 10)])

    conf = row.get("confidence_score", 5) or 5
    score += (conf - 5) * 2

    ann_ret = row.get("ann_return")
    if ann_ret is not None:
        score += _piecewise(ann_ret, [
            (-10, -8), (0, -8), (4, 5), (8, 10), (20, 10),
        ])

    fund_score = row.get("fund_score", 50) or 50
    score += (fund_score - 50) / 50 * 12

    return round(min(max(score, 0.0), 100.0), 1)


def compute_dip_score(row: dict) -> float:
    """Score 0-100 para 'comprar en caída'. Trasladado de InvestmentReportGUI._compute_dip_score."""
    base  = compute_buy_score(row)
    score = base * 0.40

    # Tendencia de fondo 20d
    c20 = row.get("change_20d") or 0
    score += _piecewise(c20, [(-20, -28), (-12, -28), (-4, -14), (0, 8), (5, 15), (12, 22), (40, 22)])

    # Caída reciente 5d
    c5 = row.get("change_5d") or 0
    score += _piecewise(c5, [(-20, 22), (-8, 22), (-5, 17), (-3, 12), (-1, 6), (3, -18), (20, -18)])

    # Distancia desde máximo 15 días
    dip = row.get("dip_from_high_pct") or 0
    score += _piecewise(dip, [(0, -6), (1.5, -6), (4, 16), (12, 16), (25, 10), (40, -8)])

    # RSI zona de entrada
    rsi = row.get("rsi")
    if rsi is not None:
        score += _piecewise(rsi, [(0, 3), (30, 6), (40, 14), (52, 14), (60, 7), (68, -14), (100, -14)])

    # Vol MA
    vol = row.get("vol_signal", "")
    if vol == "COMPRAR":   score += 9
    elif vol == "VENDER":  score -= 9

    # Caída inusual vs ATR
    dip_atr = row.get("dip_atr_multiple")
    if dip_atr is not None:
        score += _piecewise(dip_atr, [(0, 0), (1.5, 4), (2.0, 8), (3.0, 12), (10, 12)])

    return round(min(max(score, 0.0), 100.0), 1)


def compute_bigdip_score(row: dict) -> float:
    """Score 0-100 para 'big dip / turnaround'. Trasladado de InvestmentReportGUI._compute_bigdip_score."""
    price   = row.get("price") or 0
    res_52w = row.get("resistance_52w") or 0
    if price > 0 and res_52w > price:
        pct_from_high = (res_52w - price) / res_52w * 100
    else:
        pct_from_high = 0

    if pct_from_high < 15:
        return 0.0

    score = _piecewise(pct_from_high, [(15, 8), (22, 12), (35, 20), (50, 25), (80, 25)])

    # Señales de giro
    c5  = row.get("change_5d")  or 0
    c20 = row.get("change_20d") or 0
    d3  = row.get("dip_3d")     or 0

    if c20 >= 0:
        return 0.0
    if c20 < -40:
        score -= 10

    score += _piecewise(c5, [(-10, -20), (-5, -20), (0, 10), (2, 18), (5, 25), (20, 25)])

    if d3 > 1.5:  score += 10
    elif d3 > 0:  score += 5

    # RSI
    rsi = row.get("rsi")
    if rsi is not None:
        score += _piecewise(rsi, [(0, -5), (25, -5), (33, 12), (50, 20), (58, 8), (65, -10), (100, -10)])

    # Vol MA
    vol     = row.get("vol_signal", "")
    vol_rat = row.get("vol_ratio") or 1.0
    if vol == "COMPRAR":       score += 18
    elif vol_rat > 0.90:       score += 6
    elif vol == "VENDER":      score -= 10

    # Predicción
    pred = row.get("pred_signal", "")
    _pred_pts = {"SUBE FUERTE": 15, "SUBE": 10, "BAJA": -12, "BAJA FUERTE": -18}
    score += _pred_pts.get(pred, 0)

    # Calidad fundamental
    fund = row.get("fund_signal", "")
    if fund == "DESFAVORABLE": score -= 18
    elif fund == "FAVORABLE":  score += 8

    return round(min(max(score, 0.0), 100.0), 1)


# ── Confianza combinada ──────────────────────────────────────────────────────

def compute_confidence(result_row: dict, vix_data: Optional[dict] = None) -> dict:
    """Combina todas las señales en un nivel de confianza.

    Trasladado de StockAnalyzer.calculate_signal_confidence.
    Sin cambios de lógica (se puede optimizar en Etapa 2).
    """
    pts     = 0
    aligned = []
    against = []

    # Score técnico
    tech = result_row.get("total_score", 0) or 0
    if tech >= 70:       pts += 2; aligned.append(f"Score técnico {tech:.0f}/100")
    elif tech >= 55:     pts += 1
    elif tech < 40:      pts -= 1

    # ADX
    adx = result_row.get("adx_value")
    if adx:
        if adx > 30:     pts += 1; aligned.append(f"Tendencia confirmada (ADX {adx:.0f})")
        elif adx < 18:   pts -= 1; against.append(f"Sin tendencia real (ADX {adx:.0f})")

    # Vol MA
    vs = result_row.get("vol_signal")
    if vs == "COMPRAR":  pts += 1; aligned.append("Dinero entrando (Vol MA)")
    elif vs == "VENDER": pts -= 1; against.append("Dinero saliendo (Vol MA)")

    # Patrón vela
    cs = result_row.get("candle_signal")
    ck = result_row.get("candle_strength", "")
    cp = result_row.get("candle_pattern", "")
    if cs == "COMPRA":
        if ck == "muy fuerte": pts += 2; aligned.append(f"Vela {cp} (muy fuerte)")
        else:                  pts += 1; aligned.append(f"Vela {cp}")
    elif cs == "VENTA":
        pts -= 1; against.append(f"Vela bajista: {cp}")

    # Fundamentales
    fs = result_row.get("fund_signal")
    if fs == "FAVORABLE":     pts += 1; aligned.append("Fundamentales sólidos")
    elif fs == "DESFAVORABLE": pts -= 1; against.append("Fundamentales débiles")

    # VIX
    if vix_data:
        vix_sig = vix_data.get("signal", "")
        if vix_sig in ("COMPRAR AGRESIVO", "COMPRAR"):
            pts += 1; aligned.append(f"Miedo en mercado — oportunidad (VIX {vix_data['vix']:.0f})")
        elif vix_sig == "REDUCIR EXPOSICIÓN":
            pts -= 1; against.append(f"Mercado complaciente (VIX {vix_data['vix']:.0f})")

    # Backtest
    bt_wr = result_row.get("bt_win_rate_vol")
    if bt_wr is not None:
        if bt_wr >= 65:  pts += 1; aligned.append(f"Señal históricamente fiable ({bt_wr:.0f}% win rate)")
        elif bt_wr < 40: pts -= 1; against.append(f"Señal poco fiable ({bt_wr:.0f}% win rate)")

    # Sharpe + DD
    sharpe = result_row.get("sharpe")
    max_dd = result_row.get("max_dd")
    if sharpe is not None:
        if sharpe >= 1.5:  pts += 1; aligned.append(f"Sharpe excelente ({sharpe:.2f})")
        elif sharpe < 0:   pts -= 1; against.append(f"Sharpe negativo ({sharpe:.2f})")
    if max_dd is not None and max_dd < -35:
        pts -= 1; against.append(f"Drawdown severo ({max_dd:.1f}%)")

    # Fuerza Relativa + Multi-TF — cap combinado ±2 para evitar doble-conteo.
    # Ambas señales capturan la misma información (momentum relativo y tendencia);
    # sumarlas sin límite inflaría el score artificialmente cuando confluyen.
    rs_sig = result_row.get("rs_signal")
    rs_60d = result_row.get("rs_60d")
    rs_pts = 0
    if rs_sig == "LIDER":
        rs_pts = 1; aligned.append(f"Supera al S&P500 (RS60d {rs_60d:+.1f}%)" if rs_60d else "Líder vs S&P500")
    elif rs_sig == "REZAGADO":
        rs_pts = -1; against.append(f"Rezagado vs S&P500 (RS60d {rs_60d:+.1f}%)" if rs_60d else "Rezagado vs S&P500")

    tf_pts_raw = result_row.get("tf_conf_pts")
    tf_align   = result_row.get("tf_alignment", "")
    tf_contrib = 0
    if tf_pts_raw is not None:
        if tf_pts_raw >= 2:    tf_contrib = 2; aligned.append("Multi-TF alineado alcista")
        elif tf_pts_raw == 1:  tf_contrib = 1; aligned.append("Tendencia semanal alcista")
        elif tf_pts_raw == -1: tf_contrib = -1; against.append(f"Tendencia semanal bajista ({tf_align})")
        elif tf_pts_raw <= -2: tf_contrib = -2; against.append("Multi-TF alineado bajista")

    # Cap combinado: máximo ±2 para la señal RS + TF
    combined_rs_tf = max(-2, min(2, rs_pts + tf_contrib))
    pts += combined_rs_tf

    # Señal ML (máx ±1)
    ml_prob = result_row.get("ml_prob")
    if ml_prob is not None:
        try:
            p = float(ml_prob)
            if p >= 0.60:
                pts += 1; aligned.append(f"Modelo ML: prob. alta ({p:.0%})")
            elif p <= 0.35:
                pts -= 1; against.append(f"Modelo ML: prob. baja ({p:.0%})")
        except (TypeError, ValueError):
            pass

    if pts >= 5:      confidence = "MUY ALTA"
    elif pts >= 3:    confidence = "ALTA"
    elif pts >= 1:    confidence = "MEDIA"
    elif pts >= -1:   confidence = "BAJA"
    else:             confidence = "MUY BAJA"

    return {"confidence": confidence, "score": pts, "aligned": aligned, "against": against}
