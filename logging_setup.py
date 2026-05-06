"""
logging_setup.py — Configuración centralizada de logging.

Uso:
    from logging_setup import get_logger
    log = get_logger(__name__)
    log.info("mensaje")

Variables de entorno:
    FINBOT_LOG_LEVEL  — nivel global (DEBUG, INFO, WARNING, ERROR). Default: INFO.
    FINBOT_LOG_DIR    — directorio de archivos. Default: ./logs.
    FINBOT_LOG_FILE   — nombre base. Default: finbot.log.

El handler de consola usa el nivel global; el de archivo es siempre DEBUG (con rotación).
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("FINBOT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_dir = Path(os.environ.get("FINBOT_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / os.environ.get("FINBOT_LOG_FILE", "finbot.log")

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("finbot")
    root.setLevel(logging.DEBUG)
    root.propagate = False

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
               for h in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(fmt)
        root.addHandler(console)

    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        file_h = RotatingFileHandler(
            log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        file_h.setLevel(logging.DEBUG)
        file_h.setFormatter(fmt)
        root.addHandler(file_h)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger hijo de 'finbot.<name>' con la config global aplicada."""
    _configure_root()
    short = name.split(".")[-1] if name else "app"
    return logging.getLogger(f"finbot.{short}")
