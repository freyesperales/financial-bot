"""
ml_compare.py — Etapa 5.3: Comparación Sharpe OOS modelo ML vs reglas.

Evalúa dos estrategias long-only sobre el mismo período OOS:
  • Modelo ML  — entra cuando ml_prob > threshold_ml (default 0.5)
  • Reglas     — entra cuando total_score > threshold_rules (default 60)

Cada fecha del dataset representa una posible entrada. Si una estrategia
decide entrar en (symbol, date), el retorno realizado es fwd_ret_20d del
dataset (sin lookahead: se calculó con precios futuros al construir el dataset).

El portafolio diario es equi-ponderado entre todas las señales activas ese día.
La unidad de tiempo del equity curve es el step_days del dataset (≈ semana),
y el Sharpe se anualiza con ese factor.

Uso:
    from ml_compare import compare_strategies, print_comparison
    from ml_model import load_model, predict_proba
    from ml_dataset import load_or_build_dataset, feature_columns

    df = load_or_build_dataset()
    model, cv_info = load_model()

    # Añadir probabilidades al df
    feat_cols = cv_info["feature_cols"]
    X = df[feat_cols].values.astype("float32")
    df["ml_prob"] = model.predict(X)

    result = compare_strategies(df, oos_months=6)
    print_comparison(result)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ── Métricas de portafolio ────────────────────────────────────────────────────

def _portfolio_metrics(
    returns: pd.Series,
    step_days: int = 5,
    rf_annual: float = 0.045,
) -> dict:
    """
    Calcula métricas anualizadas a partir de una serie de retornos periódicos.

    step_days: días hábiles entre observaciones (para anualizació).
    Cada observación ya es el retorno acumulado del período (ej. 20 días).
    Para Sharpe se anualizan los retornos del período de observación.
    """
    rets = returns.dropna()
    n = len(rets)
    if n < 4:
        return {"n_trades": n, "sharpe": np.nan, "win_rate": np.nan,
                "avg_return": np.nan, "total_return": np.nan,
                "max_drawdown": np.nan, "ann_return": np.nan}

    # Factor de anualización: hay 252/step_days observaciones por año
    periods_per_year = 252 / step_days

    # Convertir retornos del período a retornos anualizados equivalentes
    rf_per_period    = rf_annual / periods_per_year
    excess           = rets - rf_per_period

    ann_return = float(rets.mean() * periods_per_year)
    ann_vol    = float(rets.std() * np.sqrt(periods_per_year)) if n > 1 else 0.0
    sharpe     = float(excess.mean() / rets.std() * np.sqrt(periods_per_year)) \
                 if rets.std() > 0 else 0.0

    # Downside / Sortino
    downside  = rets[rets < 0]
    ds_std    = float(downside.std() * np.sqrt(periods_per_year)) if len(downside) > 2 else np.nan
    sortino   = float((ann_return - rf_annual) / ds_std) if ds_std and ds_std > 0 else np.nan

    # Equity curve y drawdown
    equity    = (1 + rets / 100).cumprod()
    roll_max  = equity.cummax()
    dd_series = (equity / roll_max - 1) * 100
    max_dd    = float(dd_series.min())

    win_rate    = float((rets > 0).mean())
    avg_return  = float(rets.mean())
    total_ret   = float((equity.iloc[-1] - 1) * 100)

    return {
        "n_trades":    n,
        "sharpe":      round(sharpe, 3),
        "sortino":     round(sortino, 3) if not np.isnan(sortino) else None,
        "win_rate":    round(win_rate, 3),
        "avg_return":  round(avg_return, 2),
        "ann_return":  round(ann_return, 2),
        "ann_vol":     round(ann_vol, 2),
        "total_return": round(total_ret, 2),
        "max_drawdown": round(max_dd, 2),
    }


def _equity_curve(returns: pd.Series) -> pd.Series:
    """Equity curve (1 = inicio) desde una serie de retornos (%)."""
    return (1 + returns.fillna(0) / 100).cumprod()


# ── Simulación de estrategia ──────────────────────────────────────────────────

def _simulate_strategy(
    df: pd.DataFrame,
    signal_col: str,
    threshold: float,
    step_days: int = 5,
    rf_annual: float = 0.045,
) -> dict:
    """
    Simula una estrategia long-only sobre el dataset OOS.

    En cada fecha, selecciona símbolos donde signal_col > threshold.
    El retorno del portafolio ese período = media aritmética de fwd_ret_20d
    de los símbolos seleccionados.

    Parámetros
    ----------
    df         : DataFrame OOS con columnas [date, signal_col, fwd_ret_20d]
    signal_col : columna de señal (ml_prob o total_score)
    threshold  : umbral de entrada
    step_days  : paso entre observaciones (para anualización)
    rf_annual  : tasa libre de riesgo anual

    Retorna dict con métricas + equity_curve (pd.Series indexada por fecha).
    """
    if signal_col not in df.columns:
        raise ValueError(f"Columna '{signal_col}' no encontrada en df")

    df_work = df[["date", signal_col, "fwd_ret_20d"]].copy()
    df_work["active"] = df_work[signal_col] > threshold

    # Portafolio por fecha: media de fwd_ret_20d de señales activas
    daily_port = (
        df_work[df_work["active"]]
        .groupby("date")["fwd_ret_20d"]
        .mean()
    )
    # Fechas sin señal → retorno 0 (cash)
    all_dates  = pd.Series(df_work["date"].unique())
    all_dates  = pd.to_datetime(all_dates).sort_values()
    port_full  = daily_port.reindex(all_dates.dt.strftime("%Y-%m-%d"), fill_value=0.0)

    # Solo fechas donde hubo al menos una señal
    n_dates_with_signal = int((daily_port > -999).sum())

    metrics = _portfolio_metrics(daily_port, step_days=step_days, rf_annual=rf_annual)
    metrics["n_dates_with_signal"] = n_dates_with_signal
    metrics["signal_rate"] = round(
        n_dates_with_signal / max(len(all_dates), 1), 3
    )

    # Equity curve sobre todas las fechas (con cash)
    eq_curve = _equity_curve(port_full)
    eq_curve.index = all_dates.values

    return {**metrics, "equity_curve": eq_curve, "period_returns": daily_port}


# ── Comparación principal ─────────────────────────────────────────────────────

def compare_strategies(
    df: pd.DataFrame,
    oos_months: int = 6,
    threshold_ml: float = 0.5,
    threshold_rules: float = 60.0,
    step_days: int = 5,
    rf_annual: float = 0.045,
    ml_col: str = "ml_prob",
    rule_col: str = "total_score",
) -> dict:
    """
    Compara la estrategia ML vs la estrategia de reglas en el período OOS.

    El período OOS son los últimos `oos_months` meses del dataset.
    Requiere que df tenga columna `ml_prob` (añadida externamente con model.predict).

    Parámetros
    ----------
    df              : DataFrame completo (IS + OOS) con columna ml_prob añadida
    oos_months      : meses de hold-out OOS
    threshold_ml    : umbral de prob ML para señal de compra
    threshold_rules : umbral de total_score para señal de compra
    step_days       : frecuencia de observaciones en días hábiles
    rf_annual       : tasa libre de riesgo anual
    ml_col          : nombre de la columna de probabilidades ML
    rule_col        : nombre de la columna de score de reglas

    Retorna
    -------
    dict con claves:
        oos_start, oos_end, n_oos_obs
        ml      : métricas del modelo ML
        rules   : métricas de la estrategia de reglas
        buy_hold: métricas de buy-and-hold (todos los símbolos, sin filtro)
        sharpe_improvement: sharpe_ml - sharpe_rules
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    cutoff    = df["date"].max() - pd.DateOffset(months=oos_months)
    df_oos    = df[df["date"] > cutoff].copy()
    df_oos["date"] = df_oos["date"].dt.strftime("%Y-%m-%d")

    if df_oos.empty:
        raise ValueError(f"Sin datos OOS para los últimos {oos_months} meses")
    if ml_col not in df_oos.columns:
        raise ValueError(
            f"Columna '{ml_col}' no encontrada. "
            "Añade las probabilidades al df antes de llamar compare_strategies: "
            "df['ml_prob'] = model.predict(X)"
        )

    oos_start = df_oos["date"].min()
    oos_end   = df_oos["date"].max()
    n_oos     = len(df_oos)

    # ── Estrategia ML ─────────────────────────────────────────────────────────
    ml_metrics = _simulate_strategy(
        df_oos, signal_col=ml_col,
        threshold=threshold_ml,
        step_days=step_days, rf_annual=rf_annual,
    )

    # ── Estrategia de reglas ──────────────────────────────────────────────────
    rules_metrics = _simulate_strategy(
        df_oos, signal_col=rule_col,
        threshold=threshold_rules,
        step_days=step_days, rf_annual=rf_annual,
    )

    # ── Buy-and-hold (todos los símbolos sin filtro) ──────────────────────────
    bh_by_date = df_oos.groupby("date")["fwd_ret_20d"].mean()
    bh_metrics = _portfolio_metrics(bh_by_date, step_days=step_days, rf_annual=rf_annual)
    bh_metrics["n_dates_with_signal"] = len(bh_by_date)

    # ── Resumen comparativo ───────────────────────────────────────────────────
    sharpe_ml    = ml_metrics.get("sharpe", np.nan)
    sharpe_rules = rules_metrics.get("sharpe", np.nan)
    sharpe_bh    = bh_metrics.get("sharpe", np.nan)

    improvement = None
    if not np.isnan(sharpe_ml) and not np.isnan(sharpe_rules):
        improvement = round(float(sharpe_ml - sharpe_rules), 3)

    return {
        "oos_start":          oos_start,
        "oos_end":            oos_end,
        "n_oos_obs":          n_oos,
        "threshold_ml":       threshold_ml,
        "threshold_rules":    threshold_rules,
        "ml":                 {k: v for k, v in ml_metrics.items()
                               if k not in ("equity_curve", "period_returns")},
        "rules":              {k: v for k, v in rules_metrics.items()
                               if k not in ("equity_curve", "period_returns")},
        "buy_hold":           bh_metrics,
        "sharpe_improvement": improvement,
        # Curvas completas para graficar
        "_equity_ml":         ml_metrics["equity_curve"],
        "_equity_rules":      rules_metrics["equity_curve"],
        "_returns_ml":        ml_metrics["period_returns"],
        "_returns_rules":     rules_metrics["period_returns"],
    }


# ── Búsqueda del umbral óptimo ────────────────────────────────────────────────

def find_optimal_threshold(
    df: pd.DataFrame,
    oos_months: int = 6,
    step_days: int = 5,
    thresholds: Optional[list[float]] = None,
    ml_col: str = "ml_prob",
) -> pd.DataFrame:
    """
    Barre umbrales para la estrategia ML y retorna métricas por umbral.
    Útil para calibrar el punto de entrada óptimo.

    Retorna DataFrame ordenado por Sharpe descendente.
    """
    if thresholds is None:
        thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff  = df["date"].max() - pd.DateOffset(months=oos_months)
    df_oos  = df[df["date"] > cutoff].copy()
    df_oos["date"] = df_oos["date"].dt.strftime("%Y-%m-%d")

    rows = []
    for thr in thresholds:
        try:
            m = _simulate_strategy(df_oos, signal_col=ml_col,
                                   threshold=thr, step_days=step_days)
            rows.append({
                "threshold":   thr,
                "sharpe":      m["sharpe"],
                "win_rate":    m["win_rate"],
                "n_trades":    m["n_trades"],
                "avg_return":  m["avg_return"],
                "signal_rate": m["signal_rate"],
            })
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values("sharpe", ascending=False)
            .reset_index(drop=True))


# ── Presentación de resultados ────────────────────────────────────────────────

def print_comparison(result: dict) -> None:
    """Imprime un resumen tabulado de la comparación."""
    oos_start = result.get("oos_start", "?")
    oos_end   = result.get("oos_end",   "?")
    n_obs     = result.get("n_oos_obs", 0)
    thr_ml    = result.get("threshold_ml",    0.5)
    thr_rules = result.get("threshold_rules", 60)

    print(f"\n{'='*62}")
    print(f"  Comparación OOS: {oos_start} → {oos_end}  ({n_obs} obs)")
    print(f"  Umbral ML: {thr_ml}   |   Umbral reglas: total_score > {thr_rules}")
    print(f"{'='*62}")
    header = f"{'Métrica':<22}  {'ML':>9}  {'Reglas':>9}  {'Buy&Hold':>9}"
    print(header)
    print("-" * 62)

    ml = result.get("ml",       {})
    ru = result.get("rules",    {})
    bh = result.get("buy_hold", {})

    def _fmt(d: dict, key: str, pct: bool = False) -> str:
        v = d.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  N/A"
        suffix = "%" if pct else ""
        return f"{v:>8.3f}{suffix}" if not pct else f"{v:>7.2f}%"

    rows_display = [
        ("Sharpe (anual)",    "sharpe",      False),
        ("Sortino",           "sortino",     False),
        ("Win rate",          "win_rate",    False),
        ("Retorno promedio",  "avg_return",  True),
        ("Retorno anual",     "ann_return",  True),
        ("Vol anual",         "ann_vol",     True),
        ("Max Drawdown",      "max_drawdown",True),
        ("N trades/fechas",   "n_trades",    False),
    ]

    for label, key, is_pct in rows_display:
        print(f"  {label:<20}  {_fmt(ml, key, is_pct):>9}  {_fmt(ru, key, is_pct):>9}  {_fmt(bh, key, is_pct):>9}")

    print("-" * 62)
    imp = result.get("sharpe_improvement")
    if imp is not None:
        sign = "+" if imp >= 0 else ""
        winner = "MODELO ML" if imp > 0 else "REGLAS" if imp < 0 else "EMPATE"
        print(f"  Mejora Sharpe ML vs Reglas: {sign}{imp:.3f}  →  {winner}")
    print(f"{'='*62}\n")
