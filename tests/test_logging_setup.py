"""Tests del módulo logging_setup."""
import logging
from pathlib import Path

from logging_setup import get_logger


def test_get_logger_returns_finbot_namespace():
    log = get_logger("test_module")
    assert log.name.startswith("finbot.")


def test_logger_writes_to_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FINBOT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("FINBOT_LOG_FILE", "test.log")

    # Forzar reconfiguración limpiando handlers previos.
    root = logging.getLogger("finbot")
    for h in list(root.handlers):
        root.removeHandler(h)
    import logging_setup
    logging_setup._CONFIGURED = False

    log = get_logger("writer")
    log.info("hola desde el test")

    # Cerrar handlers para que el archivo se libere en Windows.
    for h in list(root.handlers):
        h.flush()

    log_file = Path(tmp_path) / "test.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "hola desde el test" in content
