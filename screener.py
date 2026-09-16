"""
screener.py — Screener consolidado multi-factor por región.

Combina en un único `screener_score` (0-100) las señales que el sistema ya
calcula por separado:

  - total_score        (técnico 0-100)              — peso 35
  - confidence_score   (confluencia de señales)      — peso 15
  - ml_prob            (prob. calibrada del modelo)  — peso 15
  - fund_score         (fundamentales 0-100)         — peso 10
  - rs_score           (fuerza relativa vs SPY)      — peso 10
  - calidad de riesgo  (Sharpe + max drawdown)       — peso 15

Los pesos se renormalizan sobre los componentes disponibles (p.ej. si no hay
modelo ML entrenado, su peso se redistribuye). Los símbolos con datos
desactualizados se marcan y pueden excluirse.

Uso:
    # Desde el último análisis guardado (last_analysis.parquet):
    python screener.py --cache

    # Con un DataFrame en código:
    from screener import build_screener, screener_markdown
    ranked = build_screener(df_results)
    print(screener_markdown(ranked))

Salida: DataFrame rankeado + reporte Markdown con secciones Global / USA /
Europa / ETFs y lista de oportunidades en dip.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from stock_universe import get_region, is_etf, get_stock_info

# Pesos por horizonte (se renormalizan sobre los componentes disponibles).
#
# "20d": timing de corto plazo — domina el score técnico (~20 días).
# "6m" : retorno a 6 meses — domina el momentum de medio plazo (60-252d,
#        evidencia Jegadeesh-Titman: el momentum 6-12m persiste 3-12 meses),
#        con más peso en fundamentales y riesgo, y el técnico corto solo
#        como timing de entrada. rel_strength se apaga porque rs_60d/252d
#        ya están dentro de momentum_6m (evita doble conteo).
HORIZON_WEIGHTS = {
    "20d": {
        "technical":    35.0,  # total_score (corto plazo)
        "momentum_6m":   0.0,
        "confidence":   15.0,  # confidence_score
        "ml":           15.0,  # ml_prob calibrada
        "fundamental":  10.0,  # fund_score
        "rel_strength": 10.0,  # rs_score
        "risk":         15.0,  # sharpe + max_dd
    },
    "6m": {
        "technical":    15.0,
        "momentum_6m":  25.0,  # change_60d/252d + rs_60d/252d
        "confidence":   10.0,
        "ml":           15.0,  # prob calibrada de +8% en 126d (si el modelo es 6m)
        "fundamental":  15.0,
        "rel_strength":  0.0,
        "risk":         20.0,
    },
}

DEFAULT_HORIZON = "6m"

# Compatibilidad con código existente
COMPONENT_WEIGHTS = HORIZON_WEIGHTS["20d"]

# Rango observado de confidence_score (puntos de confluencia)
_CONF_MIN, _CONF_MAX = -7.0, 9.0


def _norm(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Escala lineal a 0-100 con clipping."""
    return ((series.astype(float) - lo) / (hi - lo)).clip(0, 1) * 100.0


def _component_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula los componentes normalizados 0-100 (NaN si no disponibles)."""
    comp = pd.DataFrame(index=df.index)

    comp["technical"] = df["total_score"].astype(float).clip(0, 100)

    if "confidence_score" in df.columns:
        comp["confidence"] = _norm(df["confidence_score"], _CONF_MIN, _CONF_MAX)
    else:
        comp["confidence"] = np.nan

    if "ml_prob" in df.columns:
        comp["ml"] = pd.to_numeric(df["ml_prob"], errors="coerce") * 100.0
    else:
        comp["ml"] = np.nan

    if "fund_score" in df.columns:
        comp["fundamental"] = pd.to_numeric(df["fund_score"], errors="coerce").clip(0, 100)
    else:
        comp["fundamental"] = np.nan

    if "rs_score" in df.columns:
        comp["rel_strength"] = pd.to_numeric(df["rs_score"], errors="coerce").clip(0, 100)
    else:
        comp["rel_strength"] = np.nan

    # Calidad de riesgo: mitad Sharpe (clip -1..3), mitad drawdown (clip -50..0)
    sharpe_n = _norm(pd.to_numeric(df.get("sharpe"), errors="coerce"), -1.0, 3.0) \
        if "sharpe" in df.columns else pd.Series(np.nan, index=df.index)
    dd_n = _norm(pd.to_numeric(df.get("max_dd"), errors="coerce"), -50.0, 0.0) \
        if "max_dd" in df.columns else pd.Series(np.nan, index=df.index)
    comp["risk"] = pd.concat([sharpe_n, dd_n], axis=1).mean(axis=1, skipna=True)

    # Momentum de medio plazo (horizonte 6m): retornos 60d/252d absolutos y
    # relativos a SPY, normalizados a rangos empíricos razonables.
    parts = []
    for col, lo, hi in [("change_60d", -20.0, 40.0), ("change_252d", -30.0, 60.0),
                        ("rs_60d", -15.0, 15.0), ("rs_252d", -30.0, 30.0)]:
        if col in df.columns:
            parts.append(_norm(pd.to_numeric(df[col], errors="coerce"), lo, hi))
    if parts:
        comp["momentum_6m"] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    else:
        comp["momentum_6m"] = pd.Series(np.nan, index=df.index)

    return comp


def compute_screener_score(df: pd.DataFrame, horizon: str = DEFAULT_HORIZON) -> pd.Series:
    """
    Score consolidado 0-100 por fila según el perfil de horizonte.
    Renormaliza pesos sobre los componentes no-NaN de cada símbolo, de modo
    que la ausencia de un componente (p.ej. fundamentales en un ETF) no
    penaliza injustamente.
    """
    comp = _component_frame(df)
    weights = pd.Series(HORIZON_WEIGHTS[horizon])

    w_matrix = pd.DataFrame(
        np.where(comp.notna(), weights.reindex(comp.columns).fillna(0.0).values, 0.0),
        index=comp.index, columns=comp.columns,
    )
    w_sum = w_matrix.sum(axis=1).replace(0, np.nan)
    score = (comp.fillna(0) * w_matrix).sum(axis=1) / w_sum
    return score.round(1)


def build_screener(
    df_results: pd.DataFrame,
    exclude_stale: bool = False,
    horizon: str = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """
    Construye el DataFrame consolidado rankeado por screener_score.

    df_results   : salida de StockAnalyzer.analyze_all_stocks()
    exclude_stale: si True, descarta símbolos cuyos datos no están al día
    horizon      : "6m" (retorno a 6 meses, default) o "20d" (timing corto)
    """
    df = df_results.copy()

    df["region"] = df["symbol"].map(get_region)
    df["tipo"] = df["symbol"].map(lambda s: "ETF" if is_etf(s) else "Acción")
    df["nombre"] = df["symbol"].map(lambda s: get_stock_info(s)["name"])

    # Frescura de datos: cualquier marca distinta de "al día" se señala
    if "data_freshness" in df.columns:
        fresh = df["data_freshness"].astype(str)
        df["datos_ok"] = ~fresh.str.contains("desactualizado|stale|antiguo", case=False)
    else:
        df["datos_ok"] = True

    if exclude_stale:
        df = df[df["datos_ok"]]

    comp = _component_frame(df)
    for col in comp.columns:
        df[f"c_{col}"] = comp[col].round(1)

    df["screener_score"] = compute_screener_score(df, horizon=horizon)
    df.attrs["horizon"] = horizon
    df = df.sort_values("screener_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def _fmt_row(r: pd.Series) -> str:
    ml = f"{r['c_ml']:.0f}" if pd.notna(r.get("c_ml")) else "—"
    fund = f"{r['c_fundamental']:.0f}" if pd.notna(r.get("c_fundamental")) else "—"
    mom = f"{r['c_momentum_6m']:.0f}" if pd.notna(r.get("c_momentum_6m")) else "—"
    conf = r.get("confidence", "")
    fresh = "" if r.get("datos_ok", True) else " ⚠️datos"
    return (
        f"| {r['rank']} | **{r['symbol']}** | {r['nombre'][:32]} | {r['region']} | "
        f"{r['sector']} | {r['tipo']} | **{r['screener_score']:.1f}** | "
        f"{mom} | {r['total_score']:.0f} | {ml} | {fund} | {conf}{fresh} |"
    )


_TABLE_HEADER = (
    "| # | Símbolo | Nombre | Región | Sector | Tipo | Score | Mom6m | Técnico | ML | Fund | Confianza |\n"
    "|---|---------|--------|--------|--------|------|-------|-------|---------|----|------|-----------|"
)


def _section(df: pd.DataFrame, title: str, top_n: int) -> list[str]:
    out = [f"## {title}\n", _TABLE_HEADER]
    for _, r in df.head(top_n).iterrows():
        out.append(_fmt_row(r))
    out.append("")
    return out


def screener_markdown(
    ranked: pd.DataFrame,
    top_n: int = 15,
    meta: Optional[dict] = None,
) -> str:
    """Genera el reporte Markdown consolidado a partir de build_screener()."""
    horizon = ranked.attrs.get("horizon", DEFAULT_HORIZON)
    h_label = "6 meses" if horizon == "6m" else "corto plazo (~20 días)"
    lines = [f"# 🎯 Screener Consolidado de Inversiones — horizonte {h_label}"]
    lines.append(f"\n**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Símbolos evaluados:** {len(ranked)}")
    if meta:
        vix = (meta.get("vix_data") or {}).get("vix")
        if vix:
            lines.append(f"**VIX:** {vix:.1f}")
        ts = meta.get("timestamp")
        if ts:
            lines.append(f"**Datos del análisis:** {ts[:16]}")
    stale = int((~ranked["datos_ok"]).sum())
    if stale:
        lines.append(f"**⚠️ Símbolos con datos desactualizados:** {stale}")
    w = HORIZON_WEIGHTS[horizon]
    w_txt = " · ".join(f"{k} {v:.0f}%" for k, v in w.items() if v > 0)
    lines.append(f"\n**Score consolidado ({h_label})** = {w_txt}. "
                 "Pesos renormalizados si falta un componente.\n\n---\n")

    lines += _section(ranked, "🌍 Top Global", top_n)
    lines += _section(ranked[ranked["region"] == "USA"], "🇺🇸 Top USA", top_n)
    lines += _section(ranked[ranked["region"] == "Europa"], "🇪🇺 Top Europa", top_n)
    lines += _section(ranked[ranked["tipo"] == "ETF"], "📦 Top ETFs", top_n)

    # Oportunidades en dip: buen score consolidado pero caída reciente
    if "dip_from_high_pct" in ranked.columns:
        dips = ranked[
            (ranked["screener_score"] >= ranked["screener_score"].quantile(0.6))
            & (pd.to_numeric(ranked["dip_from_high_pct"], errors="coerce") <= -8)
        ]
        if not dips.empty:
            lines += _section(dips, "📉 Oportunidades en Dip (calidad + caída ≥8%)", 10)

    lines.append("---\n")
    lines.append("## ⚠️ Advertencias\n")
    lines.append("1. Análisis cuantitativo automatizado — **no es asesoría financiera**.")
    lines.append("2. Las probabilidades ML están calibradas sobre historia OOS, pero el "
                 "futuro puede diferir del pasado.")
    lines.append("3. Verifica disponibilidad e ISIN de cada instrumento en N26 antes de operar.")
    lines.append("4. Diversifica y usa stop losses.")
    return "\n".join(lines)


def load_cached_analysis(base_dir: Optional[Path] = None):
    """Carga last_analysis.parquet + meta JSON. Devuelve (df, meta)."""
    import json
    d = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    pq = d / "last_analysis.parquet"
    if not pq.exists():
        raise FileNotFoundError(
            f"No existe {pq}. Ejecuta primero un análisis (GUI o generate_investment_report.py)."
        )
    df = pd.read_parquet(pq)
    meta_path = d / "last_analysis_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return df, meta


def main():
    parser = argparse.ArgumentParser(description="Screener consolidado multi-factor")
    parser.add_argument("--cache", action="store_true",
                        help="Usar last_analysis.parquet en lugar de analizar de nuevo")
    parser.add_argument("--horizon", choices=list(HORIZON_WEIGHTS), default=DEFAULT_HORIZON,
                        help="Perfil de pesos: 6m (retorno a 6 meses) o 20d (timing corto)")
    parser.add_argument("--top", type=int, default=15, help="Filas por sección")
    parser.add_argument("--exclude-stale", action="store_true",
                        help="Excluir símbolos con datos desactualizados")
    parser.add_argument("--output", type=str, default=None,
                        help="Ruta del Markdown de salida")
    args = parser.parse_args()

    if args.cache:
        df, meta = load_cached_analysis()
    else:
        from database import TradingDatabase
        from stock_analyzer import StockAnalyzer
        db = TradingDatabase()
        analyzer = StockAnalyzer(db)
        df = analyzer.analyze_all_stocks()
        meta = {}
        db.close()

    ranked = build_screener(df, exclude_stale=args.exclude_stale, horizon=args.horizon)
    report = screener_markdown(ranked, top_n=args.top, meta=meta)

    out = Path(args.output) if args.output else Path(
        f"SCREENER_CONSOLIDADO_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    )
    out.write_text(report, encoding="utf-8")
    print(f"Reporte guardado en: {out}")
    print(f"Top 5 global: {', '.join(ranked.head(5)['symbol'])}")


if __name__ == "__main__":
    main()
