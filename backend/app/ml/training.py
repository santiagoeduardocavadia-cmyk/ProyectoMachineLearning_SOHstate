"""Entrenamiento y evaluación de un experimento.

Justificación del protocolo de validación
-----------------------------------------
Los ciclos consecutivos de una misma batería son casi idénticos. Un split
aleatorio fila a fila pondría el ciclo 50 en entrenamiento y el 51 en test, y
el modelo "aprobaría" memorizando la batería (fuga de información). Por eso
el split es **por batería**: las baterías de ``test_batteries`` nunca se ven
durante el entrenamiento, que es el escenario real (predecir una celda nueva).

Además, dentro del entrenamiento se hace validación cruzada *leave-one-cell-out*
(GroupKFold con grupos = celda física) para estimar la variabilidad del modelo.

Desde v1.2.0 el grupo es la **celda física**, no el ``battery_id``: las baterías virtuales de un CSV
aumentado (``B0005~ruido1``) y los tramos de temperatura de una misma celda (``B0042_04C`` y
``B0042_22C``) cuentan como la misma celda. Así la CV no sale optimista y ninguna copia de una
celda de test puede quedar en entrenamiento. ``test_temperatures`` permite reservar como test
**todas** las celdas de una temperatura, para medir cómo generaliza el modelo a condiciones no vistas.

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
from app.ml.domain import build_reference, is_virtual, physical_cell
from app.ml.models import build_pipeline, resolve_hyperparameters

MAX_CV_FOLDS = 10
TEMP_MATCH_C = 3.0
NON_EXTRAPOLATING = {"random_forest", "gradient_boosting", "knn"}


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
    split_warnings: list[str] = []
    test_temps = cfg.get("test_temperatures")
    if test_temps:
        if df["ambient_temperature"].isna().all():
            raise AppError("invalid_split", "El dataset no tiene 'ambient_temperature': no se puede separar por temperatura.", 422)
        med = df.groupby("battery_id")["ambient_temperature"].median()
        test = sorted(b for b, t in med.items()
                      if not is_virtual(b) and any(abs(t - x) <= TEMP_MATCH_C for x in test_temps))
        if not test:
            raise AppError("invalid_split", f"Ninguna batería a {test_temps} °C (±{TEMP_MATCH_C:g}).", 422,
                           {"available_temperatures": sorted({float(round(t)) for t in med.dropna()})})
        split_warnings.append(f"Test = todas las baterías a {test_temps} °C: {test}.")
    else:
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
    test_cells = {physical_cell(b) for b in test}
    leaked = [b for b in train if physical_cell(b) in test_cells]
    if leaked:
        train = [b for b in train if b not in leaked]
        split_warnings.append(f"Se excluyeron del entrenamiento {len(leaked)} baterías derivadas de la misma celda "
                              f"física que el test (tramos de temperatura o copias aumentadas): {leaked[:10]}"
                              f"{' …' if len(leaked) > 10 else ''}.")
    if any(is_virtual(b) for b in test):
        split_warnings.append("El test incluye baterías virtuales (aumentadas): las métricas no miden datos reales.")
    if not train:
        raise AppError("invalid_split", "Debe quedar al menos una batería para entrenamiento.", 422,
                       {"usable_batteries": sorted(usable_ids)})

    hyper = resolve_hyperparameters(cfg["model_type"], cfg.get("hyperparameters"))
    return {**cfg, "features": features, "test_batteries": test, "train_batteries": train,
            "hyperparameters": hyper, "test_temperatures": test_temps or None, "_split_warnings": split_warnings}


def train_experiment(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Entrena y evalúa. Devuelve el modelo, métricas, predicciones e importancias."""
    cfg = validate_config(df, cfg)
    data, warnings = prepare_target_frame(df, cfg["target"])
    warnings = [*cfg.pop("_split_warnings"), *warnings]
    ycol = target_column(cfg["target"])
    feats = cfg["features"]
    train_df = data[data["battery_id"].isin(cfg["train_batteries"])]
    test_df = data[data["battery_id"].isin(cfg["test_batteries"])]

    if train_df["battery_id"].map(is_virtual).any():
        warnings.append("Dataset aumentado: el modelo se entrena con las baterías virtuales, pero las métricas de "
                        "train y la CV (agrupada por celda física) usan solo celdas reales.")
    pipeline = build_pipeline(cfg["model_type"], cfg["hyperparameters"], cfg["random_state"])
    t0 = time.perf_counter()
    pipeline.fit(train_df[feats], train_df[ycol])
    train_seconds = time.perf_counter() - t0

    # Métricas y predicciones de train solo con filas reales: las virtuales son casi copias.
    train_df = train_df[~train_df["battery_id"].map(is_virtual)]
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
    if test_df["ambient_temperature"].notna().any():
        metrics["per_temperature"] = {
            f"{t:g}": regression_metrics(g[ycol], g["_pred"])
            for t, g in test_df.assign(_pred=pred_test).groupby(test_df["ambient_temperature"].round())
        }

    fit_df = data[data["battery_id"].isin(cfg["train_batteries"])]
    groups = fit_df["battery_id"].map(physical_cell)
    n_groups = groups.nunique()
    if n_groups >= 2:
        n_splits = min(n_groups, MAX_CV_FOLDS)
        cv = cross_validate(
            build_pipeline(cfg["model_type"], cfg["hyperparameters"], cfg["random_state"]),
            fit_df[feats], fit_df[ycol], groups=groups, cv=GroupKFold(n_splits=n_splits),
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
        )
        label = "leave-one-cell-out" if n_splits == n_groups else f"GroupKFold por celda física, {n_groups} celdas"
        metrics["cv"] = {
            "strategy": f"{label} ({n_splits} pliegues)",
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

    if cfg["model_type"] in NON_EXTRAPOLATING and len(test_df) and test_df[ycol].min() < train_df[ycol].min() - 1e-9:
        warnings.append(f"El test llega a {cfg['target']} = {test_df[ycol].min():.3g}, por debajo del mínimo visto en "
                        f"entrenamiento ({train_df[ycol].min():.3g}). Los modelos de árboles y KNN no extrapolan: "
                        "en esa zona se quedan en el mínimo conocido. Ridge o SVR extrapolan mejor.")

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
        "domain": build_reference(pipeline, fit_df[feats]),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
