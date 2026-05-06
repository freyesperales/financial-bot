"""
portfolio_optimizer.py — Optimizacion de portafolio.

Funciones publicas:
  optimize_mv(returns_df, ...)      — Mean-Variance con Ledoit-Wolf
  optimize_erc(returns_df, ...)     — Equal Risk Contribution
  simulate_rebalance(price_df, ...) — Equity curve con costes de transaccion
  portfolio_var(weights, ret_df)    — VaR-95 parametrico e historico
  stress_test(weights, price_df)    — Simulacion en periodos de estres

Constraints dict aceptado por optimize_mv y optimize_erc:
  max_weight        (float, default 0.20)  — peso maximo por activo
  max_sector_weight (float, default 0.35)  — peso maximo por sector
  max_beta          (float, optional)      — beta maximo del portafolio vs SPY
  min_weight        (float, default 0.0)   — peso minimo por activo incluido
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from stock_universe import get_sector
from logging_setup import get_logger

log = get_logger(__name__)

# ── Periodos de estres predefinidos ───────────────────────────────────────────
STRESS_PERIODS: dict[str, tuple[str, str]] = {
    'COVID_crash':  ('2020-02-19', '2020-03-23'),
    'Bear_2022':    ('2022-01-03', '2022-10-12'),
}

_RF_DAILY = 0.05 / 252   # 5 % anual → diario


# ── Utilidades internas ───────────────────────────────────────────────────────

def _ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    """Covarianza shrinkage Ledoit-Wolf. Fallback a covarianza muestral."""
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf(assume_centered=False)
        lw.fit(returns)
        return lw.covariance_
    except Exception as e:
        log.warning("LedoitWolf no disponible (%s) — usando cov muestral", e)
        return np.cov(returns.T)


def _sector_groups(symbols: list[str]) -> dict[str, list[int]]:
    """Devuelve {sector: [indices]} para construir restricciones sectoriales."""
    groups: dict[str, list[int]] = {}
    for i, sym in enumerate(symbols):
        sec = get_sector(sym)
        groups.setdefault(sec, []).append(i)
    return groups


def _build_constraints(
    symbols: list[str],
    cov: np.ndarray,
    spy_betas: Optional[np.ndarray],
    c: dict,
) -> list[dict]:
    """Construye lista de constraints para scipy.optimize.minimize."""
    n = len(symbols)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
    ]

    # Restriccion sectorial
    max_sec = c.get('max_sector_weight', 0.35)
    for _sec, idxs in _sector_groups(symbols).items():
        idx_arr = np.array(idxs)
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, ia=idx_arr: max_sec - float(np.sum(w[ia])),
        })

    # Restriccion beta maxima
    max_beta = c.get('max_beta')
    if max_beta is not None and spy_betas is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, b=spy_betas, mb=max_beta: mb - float(w @ b),
        })

    return constraints


def _bounds(n: int, c: dict) -> list[tuple[float, float]]:
    max_w = float(c.get('max_weight', 0.20))
    min_w = float(c.get('min_weight', 0.0))
    return [(min_w, max_w)] * n


def _estimate_betas(returns: np.ndarray, spy_rets: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Beta de cada activo respecto a SPY por regresion OLS."""
    if spy_rets is None or len(spy_rets) < 30:
        return None
    n_rows = min(len(returns), len(spy_rets))
    r = returns[-n_rows:]
    s = spy_rets[-n_rows:].reshape(-1, 1)
    var_spy = float(np.var(s, ddof=1))
    if var_spy < 1e-10:
        return None
    betas = np.array([
        float(np.cov(r[:, j], s[:, 0], ddof=1)[0, 1] / var_spy)
        for j in range(r.shape[1])
    ])
    return betas


# ── 3.1 — Mean-Variance (Ledoit-Wolf) ────────────────────────────────────────

def optimize_mv(
    returns_df: pd.DataFrame,
    scores: Optional[dict] = None,
    spy_rets: Optional[pd.Series] = None,
    constraints: Optional[dict] = None,
) -> dict:
    """
    Optimizacion Media-Varianza con shrinkage Ledoit-Wolf.

    Maximiza el Sharpe ratio sujeto a restricciones de peso, sector y beta.

    Parametros
    ----------
    returns_df : DataFrame de retornos diarios (filas=fechas, cols=simbolos).
    scores     : dict {symbol: score 0-100} para tiltar mu esperada.
    spy_rets   : Serie de retornos diarios del SPY para calcular betas.
    constraints: dict con max_weight, max_sector_weight, max_beta, min_weight.

    Retorna
    -------
    dict con keys: weights, expected_return, volatility, sharpe, method, n_assets
    """
    from scipy.optimize import minimize

    c = constraints or {}
    symbols = list(returns_df.columns)
    n = len(symbols)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 activos")

    ret_mat = returns_df.dropna(how='all').fillna(0).values.astype(float)
    mu_hist = np.mean(ret_mat, axis=0) * 252        # retorno anualizado
    cov     = _ledoit_wolf_cov(ret_mat) * 252       # cov anualizada

    # Tilt con scores (maximo +/- 2 % adicional anualizado)
    if scores:
        score_arr = np.array([scores.get(s, 50.0) for s in symbols], dtype=float)
        score_norm = (score_arr - 50.0) / 50.0       # [-1, 1]
        mu_hist = mu_hist + score_norm * 0.02

    spy_arr = spy_rets.values.astype(float) if spy_rets is not None else None
    betas   = _estimate_betas(ret_mat, spy_arr)

    cons   = _build_constraints(symbols, cov, betas, c)
    bounds = _bounds(n, c)
    w0     = np.ones(n) / n

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ mu_hist)
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol < 1e-10:
            return 0.0
        return -(port_ret - _RF_DAILY * 252) / port_vol

    result = minimize(
        neg_sharpe,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 1000, 'ftol': 1e-9},
    )

    if not result.success:
        log.warning("MV optimize no convergio: %s — usando igual ponderacion", result.message)
        w_opt = np.ones(n) / n
    else:
        w_opt = np.clip(result.x, 0.0, 1.0)
        w_opt /= w_opt.sum()

    port_ret = float(w_opt @ mu_hist)
    port_vol = float(np.sqrt(w_opt @ cov @ w_opt))
    sharpe   = (port_ret - _RF_DAILY * 252) / port_vol if port_vol > 0 else 0.0

    return {
        'weights':         {s: round(float(w_opt[i]), 6) for i, s in enumerate(symbols)},
        'expected_return': round(port_ret * 100, 2),   # %
        'volatility':      round(port_vol * 100, 2),   # %
        'sharpe':          round(sharpe, 3),
        'method':          'mean_variance_lw',
        'n_assets':        int(np.sum(w_opt > 1e-4)),
    }


# ── 3.1 — Equal Risk Contribution (ERC) ──────────────────────────────────────

def optimize_erc(
    returns_df: pd.DataFrame,
    spy_rets: Optional[pd.Series] = None,
    constraints: Optional[dict] = None,
) -> dict:
    """
    Optimizacion Equal Risk Contribution (paridad de riesgo).

    Resuelve el problema: minimizar sum_i sum_j (RC_i - RC_j)^2
    donde RC_i = w_i * (Sigma @ w)_i / (w' Sigma w).

    Retorna el mismo formato que optimize_mv.
    """
    from scipy.optimize import minimize

    c = constraints or {}
    symbols = list(returns_df.columns)
    n = len(symbols)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 activos")

    ret_mat = returns_df.dropna(how='all').fillna(0).values.astype(float)
    mu_hist = np.mean(ret_mat, axis=0) * 252
    cov     = _ledoit_wolf_cov(ret_mat) * 252

    spy_arr = spy_rets.values.astype(float) if spy_rets is not None else None
    betas   = _estimate_betas(ret_mat, spy_arr)

    cons   = _build_constraints(symbols, cov, betas, c)
    bounds = _bounds(n, c)
    w0     = np.ones(n) / n

    def erc_objective(w: np.ndarray) -> float:
        """Suma de diferencias cuadradas entre contribuciones al riesgo."""
        port_var = float(w @ cov @ w)
        if port_var < 1e-12:
            return 0.0
        rc = w * (cov @ w) / port_var   # contribucion relativa al riesgo
        target = 1.0 / n
        return float(np.sum((rc - target) ** 2))

    result = minimize(
        erc_objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 2000, 'ftol': 1e-10},
    )

    if not result.success:
        log.warning("ERC optimize no convergio: %s — usando igual ponderacion", result.message)
        w_opt = np.ones(n) / n
    else:
        w_opt = np.clip(result.x, 0.0, 1.0)
        w_opt /= w_opt.sum()

    port_ret = float(w_opt @ mu_hist)
    port_vol = float(np.sqrt(w_opt @ cov @ w_opt))
    sharpe   = (port_ret - _RF_DAILY * 252) / port_vol if port_vol > 0 else 0.0

    return {
        'weights':         {s: round(float(w_opt[i]), 6) for i, s in enumerate(symbols)},
        'expected_return': round(port_ret * 100, 2),
        'volatility':      round(port_vol * 100, 2),
        'sharpe':          round(sharpe, 3),
        'method':          'erc',
        'n_assets':        int(np.sum(w_opt > 1e-4)),
    }


# ── 3.2 — Simulacion de rebalanceo ───────────────────────────────────────────

def simulate_rebalance(
    price_df: pd.DataFrame,
    target_weights: dict,
    freq: str = 'monthly',
    cost_bps: float = 5.0,
) -> dict:
    """
    Simula el equity curve con rebalanceo periodico y costes de transaccion.

    Parametros
    ----------
    price_df      : DataFrame de precios ajustados (fechas x simbolos).
    target_weights: dict {symbol: peso} — los pesos objetivo constantes.
    freq          : 'weekly' | 'monthly'
    cost_bps      : coste de transaccion (ida) en basis points (default 5).

    Retorna
    -------
    dict con:
      equity_curve  — pd.Series indexada por fecha (valor inicial 100)
      rebalance_log — list de dicts {date, turnover_pct, cost_pct}
      total_return  — retorno total en %
      annualized    — retorno anualizado en %
      max_drawdown  — max drawdown en %
      total_cost_bps— coste total acumulado en bps
    """
    symbols = [s for s in target_weights if s in price_df.columns]
    if not symbols:
        raise ValueError("Ningun simbolo de target_weights encontrado en price_df")

    w_target = np.array([target_weights[s] for s in symbols], dtype=float)
    if w_target.sum() == 0:
        raise ValueError("Los pesos objetivo suman 0")
    w_target /= w_target.sum()

    prices = price_df[symbols].ffill().dropna(how='all')
    returns = prices.pct_change().fillna(0)
    cost_rate = cost_bps / 10_000

    # Fechas de rebalanceo
    if freq == 'weekly':
        reb_dates = set(prices.resample('W-FRI').last().index)
    else:
        reb_dates = set(prices.resample('ME').last().index)

    portfolio_val = 100.0
    w_current = w_target.copy()
    equity_curve = []
    rebalance_log = []

    for date, row in returns.iterrows():
        daily_ret = row.values.astype(float)

        # Actualizar pesos con retornos del dia
        vals = w_current * (1.0 + daily_ret)
        total = vals.sum()
        if total <= 0:
            break
        portfolio_val *= total
        w_current = vals / total

        equity_curve.append((date, portfolio_val))

        # Rebalanceo
        if date in reb_dates:
            turnover = float(np.sum(np.abs(w_current - w_target))) / 2.0
            cost = turnover * cost_rate
            portfolio_val *= (1.0 - cost)
            rebalance_log.append({
                'date':         date,
                'turnover_pct': round(turnover * 100, 2),
                'cost_pct':     round(cost * 100, 4),
            })
            w_current = w_target.copy()

    if not equity_curve:
        raise ValueError("Equity curve vacia — revisa el rango de fechas")

    curve = pd.Series(
        [v for _, v in equity_curve],
        index=[d for d, _ in equity_curve],
        name='portfolio',
    )

    total_ret  = float(curve.iloc[-1] / curve.iloc[0] - 1) * 100
    n_years    = len(curve) / 252
    ann_ret    = float((curve.iloc[-1] / curve.iloc[0]) ** (1 / max(n_years, 0.01)) - 1) * 100
    rolling_max = curve.cummax()
    drawdowns  = (curve - rolling_max) / rolling_max
    max_dd     = float(drawdowns.min()) * 100
    total_cost = sum(r['turnover_pct'] / 2 * cost_bps / 100 for r in rebalance_log)

    return {
        'equity_curve':   curve,
        'rebalance_log':  rebalance_log,
        'total_return':   round(total_ret, 2),
        'annualized':     round(ann_ret, 2),
        'max_drawdown':   round(max_dd, 2),
        'total_cost_bps': round(total_cost, 1),
        'n_rebalances':   len(rebalance_log),
    }


# ── 3.3 — Value at Risk ───────────────────────────────────────────────────────

def portfolio_var(
    weights: dict,
    returns_df: pd.DataFrame,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> dict:
    """
    VaR parametrico e historico de un portafolio.

    Parametros
    ----------
    weights     : dict {symbol: peso}
    returns_df  : DataFrame de retornos diarios
    confidence  : nivel de confianza (default 0.95)
    horizon_days: horizonte en dias de negociacion (default 1)

    Retorna
    -------
    dict con:
      var_parametric — VaR parametrico (z-score normal) en %
      var_historical — VaR historico (percentil) en %
      cvar_historical— CVaR (expected shortfall) historico en %
      port_vol_daily — volatilidad diaria del portafolio en %
      target_met     — bool, si VaR <= 1.5 % objetivo
    """
    from scipy.stats import norm

    symbols = [s for s in weights if s in returns_df.columns]
    if not symbols:
        raise ValueError("Ningun simbolo de weights encontrado en returns_df")

    w = np.array([weights[s] for s in symbols], dtype=float)
    w /= w.sum()

    ret_mat = returns_df[symbols].dropna(how='all').fillna(0).values.astype(float)
    port_rets = ret_mat @ w  # retornos del portafolio

    # Parametrico
    mu_d  = float(np.mean(port_rets))
    sig_d = float(np.std(port_rets, ddof=1))
    z     = norm.ppf(1 - confidence)
    var_param = float(-(mu_d + z * sig_d) * np.sqrt(horizon_days)) * 100

    # Historico
    alpha         = 1 - confidence
    var_hist_raw  = float(-np.percentile(port_rets, alpha * 100))
    cvar_mask     = port_rets <= -var_hist_raw
    cvar_hist     = float(-np.mean(port_rets[cvar_mask])) * 100 if cvar_mask.any() else var_hist_raw * 100
    var_hist      = var_hist_raw * np.sqrt(horizon_days) * 100

    return {
        'var_parametric':  round(var_param, 3),
        'var_historical':  round(var_hist, 3),
        'cvar_historical': round(cvar_hist, 3),
        'port_vol_daily':  round(sig_d * 100, 3),
        'confidence':      confidence,
        'horizon_days':    horizon_days,
        'target_met':      bool(var_hist <= 1.5),
    }


def suggest_var_budget(
    returns_df: pd.DataFrame,
    initial_weights: dict,
    target_var: float = 1.5,
    spy_rets: Optional[pd.Series] = None,
    constraints: Optional[dict] = None,
) -> dict:
    """
    Ajusta pesos ERC hasta que el VaR-95 1d este por debajo de target_var %.

    Retorna el resultado de optimize_erc mas el VaR resultante.
    """
    symbols = [s for s in initial_weights if s in returns_df.columns]
    if not symbols:
        raise ValueError("Ningun simbolo valido")

    result = optimize_erc(returns_df[symbols], spy_rets=spy_rets, constraints=constraints)
    var_info = portfolio_var(result['weights'], returns_df[symbols])

    result['var_info']   = var_info
    result['var_ok']     = var_info['target_met']
    result['var_hist']   = var_info['var_historical']
    return result


# ── 3.5 — Stress tests ───────────────────────────────────────────────────────

def stress_test(
    weights: dict,
    price_df: pd.DataFrame,
    periods: Optional[dict[str, tuple[str, str]]] = None,
) -> dict:
    """
    Aplica los pesos actuales a periodos historicos de estres.

    Parametros
    ----------
    weights  : dict {symbol: peso}
    price_df : DataFrame de precios ajustados (fechas x simbolos)
    periods  : dict {nombre: (fecha_inicio, fecha_fin)} en formato 'YYYY-MM-DD'
               Si None usa STRESS_PERIODS predefinidos.

    Retorna
    -------
    dict {periodo: {cum_return, max_drawdown, var_95, n_days, start, end}}
    """
    if periods is None:
        periods = STRESS_PERIODS

    symbols = [s for s in weights if s in price_df.columns]
    if not symbols:
        raise ValueError("Ningun simbolo de weights en price_df")

    w = np.array([weights[s] for s in symbols], dtype=float)
    w /= w.sum()

    results: dict = {}

    for period_name, (start, end) in periods.items():
        try:
            mask    = (price_df.index >= start) & (price_df.index <= end)
            sub     = price_df.loc[mask, symbols].ffill().dropna(how='all')
            if len(sub) < 3:
                results[period_name] = {'error': 'datos insuficientes', 'start': start, 'end': end}
                continue

            sub_ret = sub.pct_change().fillna(0).values.astype(float)
            port_rets = sub_ret @ w

            cum_ret   = float(np.prod(1.0 + port_rets) - 1) * 100
            curve     = np.cumprod(1.0 + port_rets)
            roll_max  = np.maximum.accumulate(curve)
            dd        = (curve - roll_max) / np.where(roll_max > 0, roll_max, 1)
            max_dd    = float(dd.min()) * 100
            var_95    = float(-np.percentile(port_rets, 5)) * 100

            results[period_name] = {
                'start':       start,
                'end':         end,
                'n_days':      len(port_rets),
                'cum_return':  round(cum_ret, 2),
                'max_drawdown':round(max_dd, 2),
                'var_95_1d':   round(var_95, 3),
                'daily_rets':  port_rets.tolist(),
            }
        except Exception as e:
            log.warning("stress_test '%s' error: %s", period_name, e)
            results[period_name] = {'error': str(e), 'start': start, 'end': end}

    return results
