"""
build_portfolio.py — Convierte el top del screener en una cartera con pesos.

Flujo:
  1. Carga last_analysis.parquet y rankea con el screener (horizonte 6m default).
  2. Selecciona los top N candidatos y elimina duplicados de exposición
     (pares con correlación > 0.92 — p.ej. SMH vs SOXX — conserva el mejor rankeado).
  3. Optimiza pesos con Mean-Variance (Ledoit-Wolf, tilt por score) y ERC,
     con restricciones de peso máximo por nombre y por sector.
  4. Reporta VaR-95, CVaR y stress tests (COVID Mar-2020, Bear Oct-2022).

Uso:
    python build_portfolio.py [--horizon 6m|20d] [--candidates 20] [--max-weight 0.15]
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from database import TradingDatabase
from screener import build_screener, load_cached_analysis, DEFAULT_HORIZON, HORIZON_WEIGHTS
from portfolio_optimizer import optimize_mv, optimize_erc, portfolio_var, stress_test

BASE = Path(__file__).resolve().parent
CORR_DEDUP = 0.92


def load_price_matrix(db: TradingDatabase, symbols: list[str], rows: int = 756) -> pd.DataFrame:
    """Matriz de cierres (fechas x símbolos) de los últimos `rows` días hábiles."""
    frames = {}
    for s in symbols:
        df = db.get_precios(s)
        if df.empty:
            continue
        df = df.sort_values("timestamp")
        ser = pd.Series(df["close"].values,
                        index=pd.to_datetime(df["timestamp"]).dt.normalize())
        ser = ser[~ser.index.duplicated(keep="last")]
        frames[s] = ser.iloc[-rows:]
    px = pd.DataFrame(frames).sort_index()
    # Exigir cobertura razonable: descartar símbolos con >20% de huecos
    px = px.loc[:, px.isna().mean() <= 0.2]
    return px.ffill()


def dedup_correlated(px: pd.DataFrame, ranked_order: list[str]) -> list[str]:
    """Elimina el peor rankeado de cada par con correlación > CORR_DEDUP."""
    rets = px.pct_change().dropna(how="all")
    corr = rets.corr()
    keep: list[str] = []
    for s in ranked_order:
        if s not in corr.columns:
            continue
        if any(corr.loc[s, k] > CORR_DEDUP for k in keep if k in corr.columns):
            continue
        keep.append(s)
    return keep


def main():
    parser = argparse.ArgumentParser(description="Cartera optimizada desde el screener")
    parser.add_argument("--horizon", choices=list(HORIZON_WEIGHTS), default=DEFAULT_HORIZON)
    parser.add_argument("--candidates", type=int, default=20, help="Top N candidatos del screener")
    parser.add_argument("--max-weight", type=float, default=0.15, help="Peso máximo por activo")
    parser.add_argument("--max-sector", type=float, default=0.35, help="Peso máximo por sector")
    args = parser.parse_args()

    df, meta = load_cached_analysis()
    ranked = build_screener(df, horizon=args.horizon)
    top = ranked.head(args.candidates)
    order = top["symbol"].tolist()
    print(f"Candidatos (top {args.candidates}, horizonte {args.horizon}): {', '.join(order)}")

    db = TradingDatabase()
    px = load_price_matrix(db, order + ["SPY"])
    # Matriz larga (~8 años) solo para stress tests (COVID-2020, Bear-2022)
    px_long = load_price_matrix(db, order, rows=2000)
    spy_rets = px["SPY"].pct_change().dropna() if "SPY" in px.columns else None
    px_c = px.drop(columns=["SPY"], errors="ignore")

    kept = dedup_correlated(px_c, [s for s in order if s in px_c.columns])
    dropped = [s for s in order if s in px_c.columns and s not in kept]
    if dropped:
        print(f"Descartados por solapamiento (corr > {CORR_DEDUP}): {', '.join(dropped)}")

    rets = px_c[kept].pct_change().dropna(how="all")
    scores = dict(zip(top["symbol"], top["screener_score"]))
    constraints = {"max_weight": args.max_weight, "max_sector_weight": args.max_sector}

    mv  = optimize_mv(rets, scores=scores, spy_rets=spy_rets, constraints=constraints)
    erc = optimize_erc(rets, spy_rets=spy_rets, constraints=constraints)

    lines = [f"# 💼 Cartera optimizada — horizonte {args.horizon}",
             f"\n**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"**Candidatos:** top {args.candidates} del screener, {len(kept)} tras dedup de correlación",
             f"**Restricciones:** máx {args.max_weight:.0%} por activo, {args.max_sector:.0%} por sector\n"]

    info = ranked.set_index("symbol")
    for name, res in [("Mean-Variance (Ledoit-Wolf, tilt por score)", mv),
                      ("Equal Risk Contribution", erc)]:
        var = portfolio_var(res["weights"], rets)
        stress_syms = [s for s in kept if s in px_long.columns]
        stress = stress_test(res["weights"], px_long[stress_syms])
        lines.append(f"## {name}\n")
        lines.append(f"Retorno esperado (histórico anualizado): **{res['expected_return']:.1f}%** · "
                     f"Volatilidad: **{res['volatility']:.1f}%** · Sharpe: **{res['sharpe']:.2f}**")
        lines.append(f"VaR-95 1d: **{var['var_historical']:.2f}%** · CVaR: {var['cvar_historical']:.2f}% · "
                     f"objetivo ≤1.5%: {'✅' if var['target_met'] else '⚠️ NO'}\n")
        lines.append("| Activo | Región | Sector | Peso | Score |")
        lines.append("|--------|--------|--------|------|-------|")
        ws = sorted(res["weights"].items(), key=lambda t: -t[1])
        for s, w in ws:
            if w < 0.005:
                continue
            r = info.loc[s]
            lines.append(f"| **{s}** | {r['region']} | {r['sector']} | {w:.1%} | {r['screener_score']:.1f} |")
        lines.append("\n**Stress tests:**\n")
        lines.append("| Período | Retorno | Max DD |")
        lines.append("|---------|---------|--------|")
        for pname, st in stress.items():
            if "error" in st:
                lines.append(f"| {pname} | datos insuficientes | — |")
            else:
                lines.append(f"| {pname} ({st['start']} → {st['end']}) | "
                             f"{st['cum_return']:+.1f}% | {st['max_drawdown']:.1f}% |")
        lines.append("")

    lines.append("---\n*Análisis cuantitativo automatizado — no es asesoría financiera. "
                 "Verifica cada instrumento en N26 antes de operar.*")

    out = BASE / f"PORTFOLIO_{args.horizon.upper()}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReporte guardado en: {out.name}")
    print(f"MV : ret {mv['expected_return']:.1f}% vol {mv['volatility']:.1f}% sharpe {mv['sharpe']:.2f}")
    print(f"ERC: ret {erc['expected_return']:.1f}% vol {erc['volatility']:.1f}% sharpe {erc['sharpe']:.2f}")
    db.close()


if __name__ == "__main__":
    main()
