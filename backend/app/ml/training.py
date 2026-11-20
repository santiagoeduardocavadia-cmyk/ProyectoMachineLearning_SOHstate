"""Entrenamiento y evaluación de un experimento.

Justificación del protocolo de validación
-----------------------------------------
Los ciclos consecutivos de una misma batería son casi idénticos. Un split
aleatorio fila a fila pondría el ciclo 50 en entrenamiento y el 51 en test, y
el modelo "aprobaría" memorizando la batería (fuga de información). Por eso
el split es **por batería**: las baterías de ``test_batteries`` nunca se ven
durante el entrenamiento, que es el escenario real (predecir una celda nueva).

Además, dentro del entrenamiento se hace validación cruzada *leave-one-battery-out*
(GroupKFold con grupos = batería) para estimar la variabilidad del modelo.

Métricas: MAE, RMSE, R² y MAPE en test; comparadas contra una línea base que
predice la media del entrenamiento (si el modelo no la supera, no aprende nada útil).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate

from app.data.dataset import battery_ids, target_column
from app.data.features import DEFAULT_FEATURES, FEATURE_NAMES
from app.errors import AppError
from app.ml.models import build_pipeline, resolve_hyperparameters


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = np.abs(y_true) > 1e-9
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else None
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2,
        "mape_pct": mape,
        "n": int(len(y_true)),
    }


def prepare_target_frame(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Filtra las filas válidas para el objetivo y devuelve también advertencias."""
    warnings: list[str] = []
    col = target_column(target)
    if target == "rul":
        censored = sorted(df.loc[df["rul"].isna(), "battery_id"].unique().tolist())
        if censored:
            warnings.append(f"Baterías sin fin de vida observado (RUL desconocido) excluidas: {censored}.")
        df = df[df["rul"].notna()]
        # Tras el EOL la batería está "muerta" y el RUL vale 0 constante: no aporta a la curva.
        df = df[df["cycle"] <= df["eol_cycle"]]
    df = df[df[col].notna()]
    return df, warnings


def validate_config(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    usable, _ = prepare_target_frame(df, cfg["target"])
    with_data = [f for f in FEATURE_NAMES if usable[f].notna().any()]
    features = cfg.get("features") or [f for f in DEFAULT_FEATURES if f in with_data]
    unknown = sorted(set(features) - set(FEATURE_NAMES))
    if unknown:
        raise AppError("invalid_features", f"Características desconocidas: {unknown}", 422,
                       {"available": FEATURE_NAMES})
    if len(set(features)) != len(features):
        raise AppError("invalid_features", "Hay características repetidas.", 422)
    empty = [f for f in features if f not in with_data]
    if empty:
        raise AppError("invalid_features", f"Estas características no tienen datos en el dataset: {empty}", 422,
                       {"available": with_data})
    if not features:
        raise AppError("invalid_features", "No hay características con datos para entrenar.", 422)

    all_batteries = battery_ids(df)
    test = list(dict.fromkeys(cfg["test_batteries"]))
    missing = sorted(set(test) - set(all_batteries))
    if missing:
        raise AppError("battery_not_found", f"Baterías de test inexistentes: {missing}", 422,
                       {"available": all_batteries})

    usable_ids = set(usable["battery_id"].unique())
    bad_test = sorted(set(test) - usable_ids)
    if bad_test:
        raise AppError("invalid_split", f"Las baterías {bad_test} no tienen datos válidos para '{cfg['target']}'.",
                       422, {"usable_batteries": sorted(usable_ids)})
    train = sorted(usable_ids - set(test))
    if not train:
        raise AppError("invalid_split", "Debe quedar al menos una batería para entrenamiento.", 422,
                       {"usable_batteries": sorted(usable_ids)})

    hyper = resolve_hyperparameters(cfg["model_type"], cfg.get("hyperparameters"))
    return {**cfg, "features": features, "test_batteries": test, "train_batteries": train,
            "hyperparameters": hyper}


def train_experiment(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Entrena y evalúa. Devuelve el modelo, métricas, predicciones e importancias."""
    cfg = validate_config(df, cfg)
    data, warnings = prepare_target_frame(df, cfg["target"])
    ycol = target_column(cfg["target"])
    feats = cfg["features"]
    train_df = data[data["battery_id"].isin(cfg["train_batteries"])]
    test_df = data[data["battery_id"].isin(cfg["test_batteries"])]

    pipeline = build_pipeline(cfg["model_type"], cfg["hyperparameters"], cfg["random_state"])
    t0 = time.perf_counter()
    pipeline.fit(train_df[feats], train_df[ycol])
    train_seconds = time.perf_counter() - t0

    pred_train = pipeline.predict(train_df[feats])
    pred_test = pipeline.predict(test_df[feats])
    if cfg["target"] == "rul":
        pred_train, pred_test = np.clip(pred_train, 0, None), np.clip(pred_test, 0, None)

    baseline_value = float(train_df[ycol].mean())
    metrics: dict[str, Any] = {
        "train": regression_metrics(train_df[ycol], pred_train),
        "test": regression_metrics(test_df[ycol], pred_test),
        "baseline_test": regression_metrics(test_df[ycol], np.full(len(test_df), baseline_value)),
        "per_battery": {},
        "cv": None,
        "train_seconds": round(train_seconds, 4),
    }
    for b, g in test_df.assign(_pred=pred_test).groupby("battery_id"):
        metrics["per_battery"][b] = regression_metrics(g[ycol], g["_pred"])

    groups = train_df["battery_id"]
    n_groups = groups.nunique()
    if n_groups >= 2:
        cv = cross_validate(
            build_pipeline(cfg["model_type"], cfg["hyperparameters"], cfg["random_state"]),
            train_df[feats], train_df[ycol], groups=groups, cv=GroupKFold(n_splits=n_groups),
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
        )
        metrics["cv"] = {
            "strategy": f"leave-one-battery-out ({n_groups} pliegues)",
            "mae_mean": float(-cv["test_mae"].mean()), "mae_std": float(cv["test_mae"].std()),
            "rmse_mean": float(-cv["test_rmse"].mean()), "rmse_std": float(cv["test_rmse"].std()),
            "r2_mean": float(cv["test_r2"].mean()), "r2_std": float(cv["test_r2"].std()),
        }

    perm = permutation_importance(pipeline, test_df[feats], test_df[ycol], n_repeats=10,
                                  random_state=cfg["random_state"], scoring="neg_mean_absolute_error")
    importance = sorted(
        ({"feature": f, "importance": float(m), "std": float(s)}
         for f, m, s in zip(feats, perm.importances_mean, perm.importances_std)),
        key=lambda d: d["importance"], reverse=True,
    )

    predictions = pd.concat([
        train_df[["battery_id", "cycle"]].assign(split="train", y_true=train_df[ycol].values, y_pred=pred_train),
        test_df[["battery_id", "cycle"]].assign(split="test", y_true=test_df[ycol].values, y_pred=pred_test),
    ], ignore_index=True)

    # Medianas del entrenamiento: sirven para documentar qué valor se imputa en inferencia.
    medians = {f: (None if pd.isna(v) else float(v)) for f, v in train_df[feats].median().items()}

    return {
        "config": cfg,
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_importance": importance,
        "predictions": predictions,
        "warnings": warnings,
        "train_feature_medians": medians,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
