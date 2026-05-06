"""
ml_model.py — Etapa 5.2: Modelo LightGBM con purged walk-forward CV.

Implementa la técnica de López de Prado (Advances in Financial Machine
Learning, Cap. 7) para evitar lookahead en datos financieros de panel:
  - Cada observación tiene una ventana de etiqueta [t, t+target_days].
  - Las observaciones de entrenamiento cuya etiqueta se solapa con el
    período de test se eliminan ("purge").
  - Se añade un embargo extra tras el test para mayor seguridad.

Uso:
    from ml_model import train_model, predict_proba, load_model, save_model
    from ml_dataset import load_or_build_dataset, feature_columns

    df    = load_or_build_dataset()
    model, cv_info = train_model(df)
    prob  = predict_proba(model, features_dict, cv_info["feature_cols"])
    save_model(model, cv_info)
    model, cv_info = load_model()

Estructura del resultado de train_model:
    cv_info = {
        "fold_results":  [{"fold", "n_train", "n_test", "auc", "best_iteration"}, ...],
        "mean_auc":      float,
        "std_auc":       float,
        "best_iteration": int,
        "feature_cols":  list[str],
        "positive_rate": float,
        "n_samples":     int,
    }
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_DIR  = Path(__file__).resolve().parent / "cache" / "ml"
MODEL_PATH = MODEL_DIR / "lgbm_model.txt"
INFO_PATH  = MODEL_DIR / "lgbm_info.json"

# Parámetros LightGBM por defecto
_LGB_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.05,
    "num_leaves":       31,
    "max_depth":        6,
    "min_child_samples": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "verbose":          -1,
}


# ── Purged Walk-Forward CV ────────────────────────────────────────────────────

class PurgedWalkForwardCV:
    """
    Validación cruzada walk-forward con purge + embargo para series temporales
    financieras de panel (múltiples activos × fechas).

    Lógica de purge (López de Prado):
      - Cada observación en date=t tiene su etiqueta calculada sobre [t, t+purge_days].
      - Si esa ventana se solapa con el período de test, la observación se elimina
        del conjunto de entrenamiento para evitar leakage.
      - En la práctica: se eliminan las últimas `purge_days` de entrenamiento antes
        del inicio del período de test.
      - Adicionalmente se aplica un embargo de `embargo_days` después del test.

    Parámetros
    ----------
    n_splits      : número de folds (test windows)
    test_months   : duración de cada ventana de test en meses calendario
    purge_days    : días de calendario que se eliminan antes del test (= target_days)
    embargo_days  : días de calendario de embargo después del test
    min_train_obs : mínimo de observaciones en el train para que el fold sea válido
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_months: int = 6,
        purge_days: int = 28,
        embargo_days: int = 5,
        min_train_obs: int = 200,
    ):
        self.n_splits      = n_splits
        self.test_months   = test_months
        self.purge_days    = purge_days
        self.embargo_days  = embargo_days
        self.min_train_obs = min_train_obs

    def split(self, dates: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Genera tuplas (train_indices, test_indices) para cada fold.

        dates: pd.Series con fechas de las observaciones (strings o Timestamps).
               El orden debe ser el mismo que las filas del dataset X.
        """
        dates_ts  = pd.to_datetime(dates).reset_index(drop=True)
        all_dates = pd.Series(np.sort(dates_ts.unique()))
        n_dates   = len(all_dates)

        total_span_days = (all_dates.iloc[-1] - all_dates.iloc[0]).days
        test_span_days  = self.test_months * 30

        if total_span_days <= test_span_days:
            log.warning("PurgedWalkForwardCV: rango de datos insuficiente para CV")
            return

        # Distribuir los n_splits folds empezando por el final (walk-forward)
        usable_span  = total_span_days - test_span_days
        step_days    = max(test_span_days // 2, usable_span // (self.n_splits + 1))

        for i in range(self.n_splits):
            # El último fold termina en el final del dataset
            days_from_end = (self.n_splits - 1 - i) * step_days
            test_end   = all_dates.iloc[-1] - pd.Timedelta(days=days_from_end)
            test_start = test_end - pd.Timedelta(days=test_span_days)

            # Límite de purge: excluir observaciones cuya etiqueta llega al test
            purge_cutoff   = test_start - pd.Timedelta(days=self.purge_days)
            embargo_cutoff = test_end   + pd.Timedelta(days=self.embargo_days)

            train_mask = (dates_ts < purge_cutoff)
            test_mask  = (dates_ts >= test_start) & (dates_ts <= test_end)

            train_idx = np.where(train_mask.values)[0]
            test_idx  = np.where(test_mask.values)[0]

            if len(train_idx) < self.min_train_obs or len(test_idx) < 10:
                log.debug("Fold %d descartado: train=%d test=%d",
                          i + 1, len(train_idx), len(test_idx))
                continue

            log.debug("Fold %d: train=%d obs [hasta %s], test=%d obs [%s → %s]",
                      i + 1, len(train_idx), purge_cutoff.date(),
                      len(test_idx), test_start.date(), test_end.date())
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    n_splits: int = 5,
    test_months: int = 6,
    purge_days: int = 28,
    embargo_days: int = 5,
    random_state: int = 42,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
) -> tuple:
    """
    Entrena LightGBM con purged walk-forward CV.

    Parámetros
    ----------
    df                   : DataFrame de ml_dataset.build_dataset()
    feature_cols         : columnas de features (None = auto-detecta)
    n_splits             : número de folds CV
    test_months          : meses por fold de test
    purge_days           : días de purge antes del test (debe ser >= target_days)
    embargo_days         : días de embargo post-test
    random_state         : semilla
    num_boost_round      : máximo de rounds LightGBM
    early_stopping_rounds: rounds sin mejora para parar

    Retorna
    -------
    (model: lgb.Booster, cv_info: dict)
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        raise ImportError(
            "lightgbm y scikit-learn son requeridos para Etapa 5. "
            "Instala con: pip install lightgbm scikit-learn"
        ) from e

    from ml_dataset import feature_columns as _feat_cols

    if feature_cols is None:
        feature_cols = _feat_cols(df)

    # Filtrar features que realmente existen en df
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        raise ValueError("No hay columnas de features en df")
    if "target_20d" not in df.columns:
        raise ValueError("df debe contener columna 'target_20d'")

    df = df.sort_values("date").reset_index(drop=True)

    X      = df[feature_cols].values.astype(np.float32)
    y      = df["target_20d"].values.astype(np.int32)
    dates  = df["date"]

    params = {**_LGB_PARAMS, "seed": random_state}

    cv = PurgedWalkForwardCV(
        n_splits=n_splits,
        test_months=test_months,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )

    fold_results: list[dict] = []
    best_iterations: list[int] = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(dates)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if y_tr.sum() < 5 or y_te.sum() < 2:
            log.warning("Fold %d: positivos insuficientes (train=%d, test=%d)",
                        fold_i + 1, int(y_tr.sum()), int(y_te.sum()))
            continue

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols,
                             free_raw_data=False)
        dvalid = lgb.Dataset(X_te, label=y_te, feature_name=feature_cols,
                             reference=dtrain, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        fold_model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=callbacks,
        )

        preds  = fold_model.predict(X_te)
        auc    = float(roc_auc_score(y_te, preds))
        best_n = fold_model.best_iteration

        fold_results.append({
            "fold":           fold_i + 1,
            "n_train":        int(len(train_idx)),
            "n_test":         int(len(test_idx)),
            "auc":            round(auc, 4),
            "best_iteration": int(best_n),
        })
        best_iterations.append(best_n)
        log.info("Fold %d: AUC=%.4f  best_iter=%d  train=%d  test=%d",
                 fold_i + 1, auc, best_n, len(train_idx), len(test_idx))

    if not fold_results:
        raise RuntimeError(
            "No se generaron folds válidos. El dataset puede ser muy pequeño "
            "o la distribución temporal no permite suficientes splits."
        )

    mean_auc  = float(np.mean([f["auc"] for f in fold_results]))
    std_auc   = float(np.std([f["auc"] for f in fold_results]))
    best_iter = int(np.median(best_iterations))

    log.info("CV completado: mean_AUC=%.4f ±%.4f  best_iter=%d",
             mean_auc, std_auc, best_iter)

    # ── Modelo final entrenado en todos los datos ─────────────────────────────
    dtrain_full = lgb.Dataset(X, label=y, feature_name=feature_cols,
                              free_raw_data=False)

    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=best_iter,
        callbacks=[lgb.log_evaluation(period=-1)],
    )

    cv_info = {
        "fold_results":   fold_results,
        "mean_auc":       round(mean_auc, 4),
        "std_auc":        round(std_auc,  4),
        "best_iteration": best_iter,
        "feature_cols":   feature_cols,
        "positive_rate":  round(float(y.mean()), 4),
        "n_samples":      int(len(df)),
        "n_features":     int(len(feature_cols)),
    }
    return final_model, cv_info


# ── Evaluación OOS ────────────────────────────────────────────────────────────

def evaluate_oos(
    model,
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    test_months: int = 6,
) -> dict:
    """
    Evalúa el modelo en el último período de test (OOS hold-out).

    Retorna métricas: auc, accuracy, precision, recall, positive_rate_pred.
    """
    try:
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    except ImportError as e:
        raise ImportError("scikit-learn requerido") from e

    if feature_cols is None:
        from ml_dataset import feature_columns as _fc
        feature_cols = _fc(df)
    feature_cols = [c for c in feature_cols if c in df.columns]

    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    cutoff = dates.max() - pd.DateOffset(months=test_months)
    test_mask = dates > cutoff

    if test_mask.sum() < 20:
        return {"error": "Insuficientes datos OOS"}

    X_te = df.loc[test_mask, feature_cols].values.astype(np.float32)
    y_te = df.loc[test_mask, "target_20d"].values

    probs   = model.predict(X_te)
    preds   = (probs >= 0.5).astype(int)

    return {
        "n_test":              int(test_mask.sum()),
        "auc":                 round(float(roc_auc_score(y_te, probs)), 4),
        "accuracy":            round(float(accuracy_score(y_te, preds)), 4),
        "precision":           round(float(precision_score(y_te, preds, zero_division=0)), 4),
        "recall":              round(float(recall_score(y_te, preds, zero_division=0)), 4),
        "positive_rate_pred":  round(float(preds.mean()), 4),
        "positive_rate_true":  round(float(y_te.mean()), 4),
        "oos_start":           str(cutoff.date()),
    }


# ── Inferencia ────────────────────────────────────────────────────────────────

def predict_proba(model, features: dict, feature_cols: list[str]) -> float:
    """
    Probabilidad de que el retorno forward 20d supere el umbral.

    features     : dict {feature_name → valor} (NaN si no disponible)
    feature_cols : orden de features que espera el modelo
    Retorna float en [0, 1].
    """
    x = np.array([[features.get(c, np.nan) for c in feature_cols]], dtype=np.float32)
    return float(model.predict(x)[0])


# ── Feature importance ────────────────────────────────────────────────────────

def feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """DataFrame con importancia de features (gain y split)."""
    gain  = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain":  gain,
        "importance_split": split,
    })
    df["importance_gain_norm"] = (df["importance_gain"] /
                                   df["importance_gain"].sum().clip(min=1e-9))
    return df.sort_values("importance_gain", ascending=False).reset_index(drop=True)


# ── Persistencia ──────────────────────────────────────────────────────────────

def save_model(model, cv_info: dict, model_dir: Optional[Path] = None) -> None:
    """Guarda el modelo LightGBM y el cv_info JSON en disco."""
    d = Path(model_dir) if model_dir else MODEL_DIR
    d.mkdir(parents=True, exist_ok=True)
    model.save_model(str(d / "lgbm_model.txt"))
    (d / "lgbm_info.json").write_text(
        json.dumps(cv_info, indent=2), encoding="utf-8"
    )
    log.info("Modelo guardado en %s", d)


def load_model(model_dir: Optional[Path] = None):
    """
    Carga el modelo guardado. Devuelve (model, cv_info) o lanza FileNotFoundError.
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm requerido") from e

    d     = Path(model_dir) if model_dir else MODEL_DIR
    mpath = d / "lgbm_model.txt"
    ipath = d / "lgbm_info.json"

    if not mpath.exists():
        raise FileNotFoundError(f"No se encontró modelo en {mpath}")

    model   = lgb.Booster(model_file=str(mpath))
    cv_info = json.loads(ipath.read_text(encoding="utf-8")) if ipath.exists() else {}
    log.info("Modelo cargado desde %s", mpath)
    return model, cv_info


def model_exists(model_dir: Optional[Path] = None) -> bool:
    d = Path(model_dir) if model_dir else MODEL_DIR
    return (d / "lgbm_model.txt").exists()
