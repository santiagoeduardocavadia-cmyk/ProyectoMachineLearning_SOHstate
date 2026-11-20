"""Inferencia con un experimento guardado."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app import config
from app.data.dataset import get_dataset, require_battery, target_column
from app.ml.training import regression_metrics
from app.errors import AppError
from app.ml import store


def health_status(soh: float) -> str:
    """Interpretación del SOH. El fin de vida del dataset es 70 % (1.4 Ah de 2.0 Ah)."""
    if soh >= 0.85:
        return "bueno"
    if soh >= 0.70:
        return "degradado"
    return "fin_de_vida"


def _samples_frame(samples: list[dict[str, Any]], features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    errors = []
    for i, s in enumerate(samples):
        missing = [f for f in features if f not in s]
        if missing:
            errors.append({"index": i, "missing_features": missing})
        for k, v in s.items():
            if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)):
                errors.append({"index": i, "feature": k, "message": "valor no numérico o infinito"})
    if errors:
        raise AppError("invalid_samples", "Muestras incompletas o con valores inválidos.", 422,
                       {"required_features": features, "errors": errors[:50]})
    extra = sorted({k for s in samples for k in s} - set(features))
    if extra:
        warnings.append(f"Se ignoraron columnas que el modelo no usa: {extra}.")
    frame = pd.DataFrame(samples)[features].astype(float)
    n_null = int(frame.isna().sum().sum())
    if n_null:
        warnings.append(f"{n_null} valores nulos se imputaron con la mediana del entrenamiento.")
    return frame, warnings


def predict(experiment_id: str, *, samples: list[dict] | None = None, battery_id: str | None = None,
            cycles: list[int] | None = None, actuals: list[float | None] | None = None) -> dict[str, Any]:
    """``actuals`` (opcional, modo samples): valores reales del objetivo para calcular métricas."""
    record = store.load_record(experiment_id)
    features: list[str] = record["config"]["features"]
    target: str = record["config"]["target"]
    dataset_id: str = record["config"].get("dataset_id") or "nasa"
    nominal = float(record.get("dataset", {}).get("nominal_capacity_ah") or config.NOMINAL_CAPACITY_AH)
    model = store.load_model(experiment_id)

    meta: list[dict[str, Any]] = []
    y_true: list[float | None] = []
    if samples is not None:
        X, warnings = _samples_frame(samples, features)
        meta = [{"index": i} for i in range(len(X))]
        y_true = list(actuals) if actuals is not None else [None] * len(X)
    else:
        warnings = []
        sub = require_battery(get_dataset(dataset_id), battery_id)
        if cycles:
            missing = sorted(set(cycles) - set(sub["cycle"]))
            if missing:
                raise AppError("cycle_not_found", f"Ciclos inexistentes para {battery_id}: {missing[:20]}", 404,
                               {"min_cycle": int(sub["cycle"].min()), "max_cycle": int(sub["cycle"].max())})
            sub = sub[sub["cycle"].isin(cycles)]
        X = sub[features].astype(float)
        meta = [{"battery_id": battery_id, "cycle": int(c)} for c in sub["cycle"]]
        col = sub[target_column(target)]
        y_true = [None if pd.isna(v) else float(v) for v in col]
        if battery_id in record["config"]["train_batteries"]:
            warnings.append(f"{battery_id} se usó en el entrenamiento: estas predicciones no miden generalización.")

    y = model.predict(X)
    if target == "rul":
        y = np.clip(y, 0, None)

    out = []
    for m, pred, real in zip(meta, y, y_true):
        item = {**m, "prediction": float(pred), "actual": real}
        if target == "soh":
            item["capacity_ah_estimated"] = float(pred) * nominal
            item["health_status"] = health_status(float(pred))
        elif target == "capacity":
            item["soh_estimated"] = float(pred) / nominal
            item["health_status"] = health_status(float(pred) / nominal)
        out.append(item)

    pairs = [(r, float(p)) for r, p in zip(y_true, y) if r is not None]
    metrics = None
    if pairs:
        metrics = regression_metrics([r for r, _ in pairs], [p for _, p in pairs])

    return {
        "experiment_id": experiment_id,
        "metrics": metrics,
        "target": target,
        "unit": {"soh": "fracción (0-1)", "rul": "ciclos", "capacity": "Ah"}[target],
        "features_used": features,
        "predictions": out,
        "warnings": warnings,
    }
