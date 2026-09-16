"""
update_universe_data.py — Descarga/actualización batch NO interactiva del universo.

- Símbolos sin datos en la DB: descarga el histórico completo (period=max).
- Símbolos existentes (con --refresh): añade los últimos días que falten.

Uso:
    python update_universe_data.py            # solo símbolos nuevos
    python update_universe_data.py --refresh  # además refresca los existentes
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd
import yfinance as yf

from database import TradingDatabase
from stock_universe import get_all_symbols


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [c.lower() for c in df.columns]
    return df.dropna()


def download_full(db: TradingDatabase, symbol: str) -> int:
    df = yf.download(symbol, period="max", interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty:
        return 0
    df = _clean(df)
    df["symbol"] = symbol
    df["timestamp"] = df.index
    db.bulk_insert_precios(df, symbol)
    return len(df)


def refresh_recent(db: TradingDatabase, symbol: str, last_ts) -> int:
    start = (pd.to_datetime(last_ts) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    df = yf.download(symbol, start=start, interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty:
        return 0
    df = _clean(df)
    df["symbol"] = symbol
    df["timestamp"] = df.index
    db.append_precios(df, symbol)
    return len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="Actualizar también los símbolos ya presentes en la DB")
    args = parser.parse_args()

    db = TradingDatabase()
    symbols = get_all_symbols()
    nuevos, refrescados, fallidos = [], [], []

    for i, sym in enumerate(symbols, 1):
        try:
            existing = db.get_precios(sym)
            if existing.empty:
                n = download_full(db, sym)
                if n:
                    nuevos.append(sym)
                    print(f"[{i}/{len(symbols)}] {sym}: historico completo, {n} velas")
                else:
                    fallidos.append(sym)
                    print(f"[{i}/{len(symbols)}] {sym}: SIN DATOS")
            elif args.refresh:
                last_ts = existing["timestamp"].max()
                n = refresh_recent(db, sym, last_ts)
                refrescados.append(sym)
                print(f"[{i}/{len(symbols)}] {sym}: +{n} velas recientes")
            # pausa corta para no saturar yfinance
            time.sleep(0.3)
        except Exception as e:
            fallidos.append(sym)
            print(f"[{i}/{len(symbols)}] {sym}: ERROR {e}")

    print()
    print(f"Nuevos: {len(nuevos)} | Refrescados: {len(refrescados)} | Fallidos: {len(fallidos)}")
    if fallidos:
        print("Fallidos:", ", ".join(fallidos))
    db.close()
    sys.exit(1 if len(fallidos) > len(symbols) * 0.2 else 0)


if __name__ == "__main__":
    main()
