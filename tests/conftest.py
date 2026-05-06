"""
conftest.py — Fixtures compartidos por toda la suite de tests.

Estrategia:
- La primera vez que se corre la suite, se intenta extraer fixtures
  reproducibles desde la DB local del proyecto (`trading_bot.db`).
  Los datos extraídos se guardan en `tests/fixtures/*.csv`.
- En CI o en máquinas sin DB, los CSV ya generados sirven directamente.

Esto desacopla los tests del estado vivo de la base de datos.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Permitir imports desde el directorio raíz del proyecto.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

# Símbolos que extraemos como fixture si no existen ya en disco.
_FIXTURE_SYMBOLS = ("AAPL", "SPY", "TLT")
_MAX_ROWS = 1500  # ~6 años de datos diarios; suficiente para todos los rolling.


def _ensure_fixture_csv(symbol: str) -> Path:
    """Garantiza que `tests/fixtures/{symbol}.csv` existe.

    Si no existe, intenta extraer de la DB local. Si tampoco hay DB,
    se omite el test que lo pida (skip), no se cae la suite.
    """
    csv_path = FIXTURES_DIR / f"{symbol}.csv"
    if csv_path.exists():
        return csv_path

    db_path = ROOT / "trading_bot.db"
    if not db_path.exists():
        return csv_path  # quedará no-existente; los tests deben manejar skip

    try:
        from database import TradingDatabase
    except ImportError:
        return csv_path

    db = TradingDatabase(db_path=str(db_path), auto_cleanup_snapshots=False)
    df = db.get_precios(symbol)
    db.close()
    if df.empty:
        return csv_path

    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    if len(df) > _MAX_ROWS:
        df = df.tail(_MAX_ROWS).reset_index(drop=True)

    keep_cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in df.columns]
    df[keep_cols].to_csv(csv_path, index=False)
    return csv_path


def _load_fixture(symbol: str) -> pd.DataFrame:
    csv_path = _ensure_fixture_csv(symbol)
    if not csv_path.exists():
        pytest.skip(
            f"No hay fixture para {symbol} y no se encontró DB para generarlo. "
            f"Ejecuta el GUI al menos una vez para descargar datos."
        )
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def aapl_df() -> pd.DataFrame:
    """OHLCV diario de AAPL — proxy de acción individual representativa."""
    return _load_fixture("AAPL").copy()


@pytest.fixture(scope="session")
def spy_df() -> pd.DataFrame:
    """OHLCV diario de SPY — proxy de ETF de renta variable."""
    return _load_fixture("SPY").copy()


@pytest.fixture(scope="session")
def tlt_df() -> pd.DataFrame:
    """OHLCV diario de TLT — proxy de ETF de renta fija."""
    return _load_fixture("TLT").copy()


@pytest.fixture(scope="session", autouse=True)
def _ensure_all_fixtures():
    """Pre-genera todos los CSV de fixture al arrancar la sesión.

    Si la DB no existe, no falla; cada test individual decidirá si
    skipea o no según el fixture concreto que pida.
    """
    for sym in _FIXTURE_SYMBOLS:
        _ensure_fixture_csv(sym)
    yield
