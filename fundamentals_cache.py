"""
fundamentals_cache.py — Caché en disco para yfinance.Ticker(...).info.

Problema resuelto:
  En cada análisis de 160 símbolos, calculate_fundamentals() hace una
  petición HTTP a Yahoo Finance por símbolo. Con TTL de 24h, la segunda
  ejecución del día no vuelve a pegar a la red.

Formato:
  cache/fundamentals/<SYMBOL>.json
  Cada archivo tiene un campo "_cached_at" (ISO timestamp).
  Si el archivo tiene más de TTL_HOURS horas, se descarga de nuevo.

Uso:
  from fundamentals_cache import get_info
  info = get_info("AAPL")   # dict idéntico a yf.Ticker("AAPL").info
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from logging_setup import get_logger

log = get_logger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "fundamentals"
_TTL_HOURS = 24


def _cache_path(symbol: str) -> Path:
    return _CACHE_DIR / f"{symbol.upper()}.json"


def _is_fresh(path: Path, ttl_hours: int = _TTL_HOURS) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(data.get("_cached_at", "1970-01-01"))
        return datetime.now() - cached_at < timedelta(hours=ttl_hours)
    except Exception:
        return False


def _load(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("_cached_at", None)
        return data
    except Exception:
        return None


def _save(path: Path, info: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {**info, "_cached_at": datetime.now().isoformat()}
    # JSON solo acepta tipos básicos; yfinance puede devolver tipos numpy.
    def _to_native(v):
        if hasattr(v, "item"):   # numpy scalar
            return v.item()
        if isinstance(v, dict):
            return {k: _to_native(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_to_native(vv) for vv in v]
        return v
    try:
        path.write_text(json.dumps(_to_native(payload), ensure_ascii=False),
                        encoding="utf-8")
    except Exception as e:
        log.warning("No se pudo guardar caché de fundamentals para %s: %s", path.stem, e)


def get_info(symbol: str, ttl_hours: int = _TTL_HOURS) -> dict:
    """Devuelve yf.Ticker(symbol).info usando caché en disco de TTL horas.

    Si la caché es fresca, no hay petición de red.
    Si falla la descarga y hay caché expirado, lo usa de todas formas (fallback).
    """
    path = _cache_path(symbol)

    if _is_fresh(path, ttl_hours):
        data = _load(path)
        if data is not None:
            log.debug("fundamentals cache HIT: %s", symbol)
            return data

    log.debug("fundamentals cache MISS: %s — descargando de yfinance", symbol)
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
        _save(path, info)
        return info
    except Exception as e:
        log.warning("Error descargando fundamentals de %s: %s", symbol, e)
        # Fallback: usar caché expirado si existe
        if path.exists():
            data = _load(path)
            if data is not None:
                log.info("Usando caché expirado de %s como fallback", symbol)
                return data
        return {}


def clear_cache(symbol: Optional[str] = None) -> None:
    """Borra la caché de un símbolo, o de todos si symbol=None."""
    if symbol:
        p = _cache_path(symbol)
        if p.exists():
            p.unlink()
    else:
        for p in _CACHE_DIR.glob("*.json"):
            p.unlink()
