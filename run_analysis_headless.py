"""
run_analysis_headless.py — Análisis completo sin GUI.

Flujo:
  1. (opcional --train) Construye dataset ML, entrena LightGBM con purged CV
     y calibración isotónica, y guarda el modelo en cache/ml/.
  2. Ejecuta StockAnalyzer.analyze_all_stocks() sobre todo el universo.
  3. Guarda last_analysis.parquet + last_analysis_meta.json (mismo formato que
     el GUI, para que la app y `screener.py --cache` los reutilicen).
  4. Genera el reporte del screener consolidado.

Uso:
    python run_analysis_headless.py [--train] [--top 15]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent


# Perfiles de horizonte para el modelo ML.
# 6m: target = retorno a 126 días hábiles > +8% (≈16% anualizado).
#     purge = 130 días (>= target) para evitar leakage en la CV purgada.
HORIZON_PROFILES = {
    "20d": {"target_days": 20,  "target_pct": 3.0, "purge_days": 28,  "embargo_days": 5,
            "lookback_years": 3.0},
    "6m":  {"target_days": 126, "target_pct": 8.0, "purge_days": 130, "embargo_days": 10,
            "lookback_years": 6.0},
}


def train_ml_model(horizon: str = "6m"):
    prof = HORIZON_PROFILES[horizon]
    print("=" * 60)
    print(f"  Entrenando modelo ML horizonte {horizon} "
          f"(target {prof['target_days']}d > +{prof['target_pct']}%)")
    print("=" * 60)
    from ml_dataset import load_or_build_dataset
    from ml_model import train_model, save_model
    from ml_signal import invalidate_model_cache

    df = load_or_build_dataset(
        force=True,
        target_days=prof["target_days"],
        target_pct=prof["target_pct"],
        lookback_years=prof["lookback_years"],
    )
    print(f"Dataset: {len(df)} filas | tasa positiva: {df['target_20d'].mean():.2%}")
    model, cv_info = train_model(
        df,
        purge_days=prof["purge_days"],
        embargo_days=prof["embargo_days"],
    )
    cv_info["horizon"] = horizon
    cv_info["target_days"] = prof["target_days"]
    cv_info["target_pct"] = prof["target_pct"]
    save_model(model, cv_info)
    invalidate_model_cache()
    cal = cv_info.get("calibration") or {}
    print(f"AUC medio CV: {cv_info['mean_auc']} +/- {cv_info['std_auc']}")
    if cal:
        print(f"Calibracion: Brier {cal['brier_raw']} -> {cal['brier_cal']} "
              f"({cal['n_oos']} muestras OOS)")
    return cv_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Entrenar modelo ML antes de analizar")
    parser.add_argument("--horizon", choices=list(HORIZON_PROFILES), default="6m",
                        help="Horizonte del modelo ML y del screener")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    if args.train:
        try:
            train_ml_model(args.horizon)
        except Exception as e:
            print(f"AVISO: entrenamiento ML fallido ({e}) — se continua sin ML")

    from database import TradingDatabase
    from stock_analyzer import StockAnalyzer
    from screener import build_screener, screener_markdown

    db = TradingDatabase()
    analyzer = StockAnalyzer(db)
    df_results = analyzer.analyze_all_stocks()

    if df_results is None or df_results.empty:
        print("ERROR: el analisis no devolvio resultados")
        db.close()
        return 1

    # Persistir en el mismo formato que el GUI
    df_results.to_parquet(BASE / "last_analysis.parquet", index=False)
    meta = {
        "timestamp": datetime.now().isoformat(),
        "vix_data": getattr(analyzer, "_vix_data", None) or analyzer.get_vix_level(),
        "spy_data": None,
        "portfolio": [],
    }
    (BASE / "last_analysis_meta.json").write_text(json.dumps(meta, indent=2),
                                                  encoding="utf-8")
    print(f"\nGuardado last_analysis.parquet ({len(df_results)} simbolos)")

    ranked = build_screener(df_results, horizon=args.horizon)
    report = screener_markdown(ranked, top_n=args.top, meta=meta)
    # Nombre estable por horizonte: cada corrida sobrescribe (sin acumular archivos)
    out = BASE / f"SCREENER_CONSOLIDADO_{args.horizon.upper()}.md"
    out.write_text(report, encoding="utf-8")
    print(f"Screener guardado en: {out.name}")
    print("Top 10 global:", ", ".join(ranked.head(10)["symbol"]))

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
