"""Dominio de validez de un modelo: ¿se parecen los datos nuevos a los de entrenamiento? (v1.2.0)

Un modelo de regresión solo es fiable dentro de las condiciones que vio al entrenar. Con una
celda a otra temperatura o corriente, el modelo no "sabe" que no sabe: devuelve un número igual
de seguro, pero puede estar muy lejos de la realidad (el profesor lo llamó "alucinar").

Este módulo mide esa distancia sin usar nunca el valor real del objetivo:

- **Distancia** de cada muestra al entrenamiento: media de |z| de sus características, donde z es
  el valor estandarizado con la media y la desviación del entrenamiento (el ``StandardScaler`` del
  propio pipeline). Se divide por el percentil 95 de esa misma distancia en el entrenamiento:
  ``ratio <= 1`` es tan típico como el 95 % de los ciclos vistos.
- **Rangos**: percentiles 1-99 de cada característica en el entrenamiento (con un margen del 10 %).
- **Condiciones**: temperaturas ambiente vistas (±3 °C) y corrientes de descarga vistas (±0.3 A).

Cada muestra queda ``dentro``, ``borde`` o ``fuera`` del dominio.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

TEMP_TOLERANCE_C = 3.0
CURRENT_TOLERANCE_A = 0.3
RANGE_MARGIN = 0.10
_SEGMENT_RE = re.compile(r"_\d+C$")


def physical_cell(battery_id: str) -> str:
    """Celda física de origen: ``B0005~ruido1`` -> ``B0005``; ``B0042_04C`` -> ``B0042``."""
    return _SEGMENT_RE.sub("", str(battery_id).split("~", 1)[0])


def is_virtual(battery_id: str) -> bool:
    """Batería virtual creada con ``scripts/augment_dataset.py``."""
    return "~" in str(battery_id)


def _standardized(pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(pipeline[:-1].transform(X), dtype=float)


def _distance(z: np.ndarray) -> np.ndarray:
    return np.nanmean(np.abs(z), axis=1)


def _seen(values: pd.Series, step: float) -> list[float]:
    v = values.dropna()
    return sorted({float(round(x / step) * step) for x in v}) if len(v) else []


def build_reference(pipeline, X_train: pd.DataFrame) -> dict[str, Any]:
    """Resumen del dominio de entrenamiento que se guarda en ``experiment.json`` (campo ``domain``)."""
    d = _distance(_standardized(pipeline, X_train))
    ranges = {}
    for f in X_train.columns:
        col = X_train[f].astype(float).dropna()
        if len(col):
            ranges[f] = [float(np.percentile(col, 1)), float(np.percentile(col, 99))]
    ref: dict[str, Any] = {
        "distance_p95": float(np.percentile(d, 95)) if len(d) else 1.0,
        "distance_median": float(np.median(d)) if len(d) else 1.0,
        "feature_ranges": ranges,
        "temperatures": _seen(X_train["ambient_temperature"], 1.0) if "ambient_temperature" in X_train else [],
        "currents": _seen(X_train["discharge_current_a"], 0.5) if "discharge_current_a" in X_train else [],
    }
    ref["distance_p95"] = max(ref["distance_p95"], 1e-6)
    return ref


def _near(value: float, seen: list[float], tol: float) -> bool:
    return any(abs(value - s) <= tol for s in seen)


def assess(pipeline, reference: dict[str, Any], X: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Evalúa cada fila de ``X`` (sin imputar). Devuelve (por fila, resumen, advertencias)."""
    ratio = _distance(_standardized(pipeline, X)) / reference["distance_p95"]
    ranges = reference.get("feature_ranges", {})
    temps, currents = reference.get("temperatures") or [], reference.get("currents") or []

    rows: list[dict[str, Any]] = []
    out_counts: dict[str, int] = {}
    unseen_temps: set[float] = set()
    unseen_currents: set[float] = set()
    for i in range(len(X)):
        row = X.iloc[i]
        out_feats = []
        for f, (lo, hi) in ranges.items():
            v = row.get(f)
            if v is None or pd.isna(v):
                continue
            margin = RANGE_MARGIN * max(hi - lo, abs(hi) * 0.01, 1e-9)
            if v < lo - margin or v > hi + margin:
                out_feats.append(f)
                out_counts[f] = out_counts.get(f, 0) + 1
        condition_out = False
        t = row.get("ambient_temperature")
        if temps and t is not None and not pd.isna(t) and not _near(float(t), temps, TEMP_TOLERANCE_C):
            condition_out = True
            unseen_temps.add(float(round(t)))
        c = row.get("discharge_current_a")
        if currents and c is not None and not pd.isna(c) and not _near(float(c), currents, CURRENT_TOLERANCE_A):
            condition_out = True
            unseen_currents.add(float(round(c, 1)))
        r = float(ratio[i]) if np.isfinite(ratio[i]) else None
        if condition_out or (r is not None and r > 2.0):
            status = "fuera"
        elif out_feats or (r is not None and r > 1.0):
            status = "borde"
        else:
            status = "dentro"
        rows.append({"domain_status": status, "domain_distance": r, "out_of_range_features": out_feats})

    n = max(len(rows), 1)
    frac = {s: sum(x["domain_status"] == s for x in rows) / n for s in ("dentro", "borde", "fuera")}
    overall = "fuera" if frac["fuera"] >= 0.2 else "borde" if frac["fuera"] + frac["borde"] >= 0.2 else "dentro"
    finite = [x["domain_distance"] for x in rows if x["domain_distance"] is not None]
    summary = {
        "status": overall,
        "fraction": {k: round(v, 4) for k, v in frac.items()},
        "distance_ratio_median": float(np.median(finite)) if finite else None,
        "features_out_of_range": [
            {"feature": f, "n_rows": k, "train_range": ranges[f]}
            for f, k in sorted(out_counts.items(), key=lambda kv: -kv[1])
        ],
        "train_temperatures": temps,
        "unseen_temperatures": sorted(unseen_temps),
        "train_currents": currents,
        "unseen_currents": sorted(unseen_currents),
        "message": {
            "dentro": "Los datos se parecen a los del entrenamiento: la predicción es fiable dentro del error de test.",
            "borde": "Parte de los datos está en el borde de lo visto en entrenamiento: el error puede ser mayor que el de test.",
            "fuera": "Los datos están fuera de las condiciones de entrenamiento: la predicción NO es fiable. "
                     "Entrena con datos de esas condiciones (p. ej. el dataset nasa_multitemp).",
        }[overall],
    }
    warnings = []
    if unseen_temps:
        warnings.append(f"Temperaturas no vistas en entrenamiento: {sorted(unseen_temps)} °C "
                        f"(el modelo vio {temps} °C).")
    if unseen_currents:
        warnings.append(f"Corrientes de descarga no vistas en entrenamiento: {sorted(unseen_currents)} A "
                        f"(el modelo vio {currents} A).")
    if overall != "dentro":
        warnings.append(summary["message"])
    return rows, summary, warnings
