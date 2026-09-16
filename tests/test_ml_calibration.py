"""Tests de calibración isotónica de probabilidades ML (ml_model)."""
import numpy as np
import pytest

from ml_model import _fit_isotonic_calibration, apply_calibration


def _synthetic_overconfident(n=2000, seed=7):
    """Predicciones sobre-confiadas: prob cruda 0.9 pero frecuencia real ~0.6."""
    rng = np.random.default_rng(seed)
    preds = rng.uniform(0.05, 0.95, size=n)
    # La frecuencia real es una versión comprimida hacia 0.5 de la predicción
    true_p = 0.5 + (preds - 0.5) * 0.5
    labels = (rng.uniform(size=n) < true_p).astype(float)
    return preds, labels


def test_fit_returns_serializable_dict():
    preds, labels = _synthetic_overconfident()
    cal = _fit_isotonic_calibration(preds, labels)
    assert cal is not None
    assert cal["method"] == "isotonic"
    assert len(cal["x"]) == len(cal["y"]) >= 2
    assert cal["n_oos"] == len(preds)
    # Debe ser JSON-serializable (floats/ints nativos)
    import json
    json.dumps(cal)


def test_calibration_improves_brier():
    preds, labels = _synthetic_overconfident()
    cal = _fit_isotonic_calibration(preds, labels)
    assert cal["brier_cal"] <= cal["brier_raw"]


def test_apply_calibration_compresses_overconfidence():
    preds, labels = _synthetic_overconfident()
    cal = _fit_isotonic_calibration(preds, labels)
    cv_info = {"calibration": cal}
    # Una prob cruda de 0.9 debe quedar claramente por debajo tras calibrar
    assert apply_calibration(0.9, cv_info) < 0.85
    # Monotonía: mayor prob cruda nunca da menor prob calibrada
    grid = np.linspace(0.05, 0.95, 50)
    out = apply_calibration(grid, cv_info)
    assert np.all(np.diff(out) >= -1e-12)
    # Rango válido
    assert np.all((out >= 0) & (out <= 1))


def test_apply_calibration_passthrough_without_cal():
    assert apply_calibration(0.7, None) == 0.7
    assert apply_calibration(0.7, {}) == 0.7
    arr = np.array([0.1, 0.9])
    np.testing.assert_array_equal(apply_calibration(arr, {"calibration": {}}), arr)


def test_scalar_in_scalar_out():
    preds, labels = _synthetic_overconfident()
    cal = _fit_isotonic_calibration(preds, labels)
    out = apply_calibration(0.5, {"calibration": cal})
    assert isinstance(out, float)


def test_too_few_samples_returns_none():
    preds = np.array([0.2, 0.8, 0.5])
    labels = np.array([0.0, 1.0, 1.0])
    assert _fit_isotonic_calibration(preds, labels) is None
