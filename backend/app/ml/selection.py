"""Selección automática del modelo para un CSV nuevo (v1.2.0, ``POST /api/v1/predict/auto``).

Criterio, en este orden:

1. Solo compiten los experimentos del objetivo pedido cuyas características están todas en el CSV.
2. Se elige el experimento cuyo dominio de entrenamiento está **más cerca** del CSV: menor mediana del
   ratio de distancia (``domain.py``), penalizando las temperaturas y corrientes que el modelo no vio.
3. Si dos candidatos quedan a menos de un 10 % de distancia, gana el de menor MAE de **validación cruzada**
   por celda física (v1.5.0; antes era el MAE de test, pero el test no debe usarse para elegir).
   Así, entre los 5 modelos entrenados con los mismos datos, se usa el que menos se equivoca en celdas
   que no vio (ver ``scripts/compare_models.py`` y ``docs/SELECCION_MODELO.md``).

La elección usa solo las características de entrada, **nunca el valor real** del CSV (sería hacer trampa).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from app.data.dataset import get_dataset
from app.errors import AppError
from app import config
from app.ml import domain, inference, life, store
from app.ml.models import MODELS

TIE_TOLERANCE = 0.10
CONDITION_PENALTY = 2.0
TARGET_COLUMNS = ("soh", "rul", "eol_cycle", "capacity_ah")
N_SIMILAR = 5


def _score(record: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    feats = record["config"]["features"]
    base = {"experiment_id": record["id"], "name": record["name"], "model_type": record["config"]["model_type"],
            "dataset_id": record["config"].get("dataset_id") or "nasa",
            "test_mae": (record.get("metrics", {}).get("test") or {}).get("mae"),
            "cv_mae": (record.get("metrics", {}).get("cv") or {}).get("mae_mean")}
    missing = [f for f in feats if f not in df.columns]
    if missing:
        return {**base, "eligible": False, "reason": f"al CSV le faltan columnas: {missing}"}
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype(float)
    try:
        model = store.load_model(record["id"])
    except AppError as exc:
        return {**base, "eligible": False, "reason": exc.message}
    ref = inference.domain_reference(record, model)
    if ref is None:
        return {**base, "eligible": False, "reason": "no se pudo reconstruir su dominio de entrenamiento"}
    _, summary, _ = domain.assess(model, ref, X)
    ratio = summary["distance_ratio_median"]
    penalty = CONDITION_PENALTY * (bool(summary["unseen_temperatures"]) + bool(summary["unseen_currents"]))
    score = (ratio if ratio is not None else np.inf) + penalty
    return {**base, "eligible": True, "score": float(score), "distance_ratio_median": ratio,
            "domain_status": summary["status"], "unseen_temperatures": summary["unseen_temperatures"],
            "unseen_currents": summary["unseen_currents"]}


def rank(df: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    records = [r for r in store.list_records() if r["config"]["target"] == target]
    scored = [_score(r, df) for r in records]
    ok = sorted((c for c in scored if c["eligible"]), key=lambda c: c["score"])
    if ok:
        best = ok[0]["score"]
        close = [c for c in ok if c["score"] <= best * (1 + TIE_TOLERANCE) + 1e-12]
        # Desempate por validación cruzada (celdas que el modelo no vio), no por el test: el test solo confirma.
        close.sort(key=lambda c: (c["cv_mae"] if c["cv_mae"] is not None else np.inf,
                                  c["test_mae"] if c["test_mae"] is not None else np.inf))
        ok = close + [c for c in ok if c not in close]
    return ok + [c for c in scored if not c["eligible"]]


def similar_cells(record: dict[str, Any], df: pd.DataFrame, n: int = N_SIMILAR) -> list[dict[str, Any]]:
    """Celdas de entrenamiento del experimento elegido que más se parecen al CSV (v1.3.0).

    Se compara el perfil medio estandarizado de las características (con el escalador del propio
    modelo) del CSV contra el de cada celda real de entrenamiento: distancia = media de |Δz|.
    No usa el valor real del objetivo. Una celda nueva no "es" ninguna de las entrenadas: esto dice
    a cuáles se parece, que es lo que justifica (o no) confiar en la predicción.
    """
    feats = record["config"]["features"]
    try:
        data = get_dataset(record["config"].get("dataset_id") or "nasa")
    except AppError:
        return []
    model = store.load_model(record["id"])
    train = data[data["battery_id"].isin(record["config"]["train_batteries"])
                 & ~data["battery_id"].map(domain.is_virtual)]
    if train.empty:
        return []
    transform = model[:-1].transform
    target = np.nanmean(np.asarray(transform(df[feats].apply(pd.to_numeric, errors="coerce").astype(float)),
                                   dtype=float), axis=0)
    out = []
    for b, g in train.groupby("battery_id"):
        prof = np.nanmean(np.asarray(transform(g[feats].astype(float)), dtype=float), axis=0)
        temp, cur = g["ambient_temperature"].median(), g["discharge_current_a"].median()
        out.append({
            "battery_id": b,
            "distance": float(np.nanmean(np.abs(target - prof))),
            "ambient_temperature": None if pd.isna(temp) else float(temp),
            "discharge_current_a": None if pd.isna(cur) else round(float(cur), 2),
            "n_cycles": int(len(g)),
            "soh_initial": float(g["soh"].iloc[0]) if g["soh"].notna().any() else None,
            "soh_final": float(g["soh"].iloc[-1]) if g["soh"].notna().any() else None,
        })
    return sorted(out, key=lambda c: c["distance"])[:n]


def _csv_median(df: pd.DataFrame, col: str) -> float | None:
    if col not in df:
        return None
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(v.median()) if len(v) else None


def _charge_hours(df: pd.DataFrame) -> float | None:
    """Duración típica de la recarga del CSV (fases CC + CV, en horas)."""
    cc, cv = _csv_median(df, "cc_charge_time"), _csv_median(df, "cv_charge_time")
    return None if cc is None else (cc + (cv or 0.0)) / 3600.0


def predict_auto(df: pd.DataFrame, target: str, predict_frame: Callable[[str, pd.DataFrame], dict]) -> dict[str, Any]:
    candidates = rank(df, target)
    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        raise AppError("no_compatible_experiment",
                       f"Ningún experimento de '{target}' puede usar las columnas de este CSV.", 422,
                       {"candidates": candidates[:50]})
    chosen = eligible[0]
    result = predict_frame(chosen["experiment_id"], df)
    reason = (f"'{chosen['name']}' es el experimento cuyo entrenamiento más se parece a estos datos "
              f"(distancia mediana {chosen['distance_ratio_median']:.2f}; 1 = típico del entrenamiento)")
    if len(eligible) > 1 and eligible[1]["score"] <= chosen["score"] * (1 + TIE_TOLERANCE) + 1e-12:
        n_tied = sum(c["score"] <= chosen["score"] * (1 + TIE_TOLERANCE) + 1e-12 for c in eligible)
        cv = f" ({chosen['cv_mae']:.4f})" if chosen.get("cv_mae") is not None else ""
        reason += (f". Empató en distancia con {n_tied - 1} experimento(s) más (mismos datos de entrenamiento) y se "
                   f"eligió {MODELS[chosen['model_type']].label} por tener el menor error de validación cruzada{cv}")
    result["selection"] = {
        "selected_experiment_id": chosen["experiment_id"],
        "criterion": "menor distancia al dominio de entrenamiento (sin usar el valor real); desempate por menor MAE de "
                     "validación cruzada por celda física",
        "reason": reason + ".",
        "candidates": candidates[:50],
    }
    record = store.load_record(chosen["experiment_id"])
    result["similar_cells"] = similar_cells(record, df)
    if result["similar_cells"]:
        best = result["similar_cells"][0]
        t = f" a {best['ambient_temperature']:g} °C" if best["ambient_temperature"] is not None else ""
        result["selection"]["similar_cells_summary"] = (
            f"Tu batería se parece más a {best['battery_id']}{t} (y a "
            f"{', '.join(c['battery_id'] for c in result['similar_cells'][1:3])}) entre las celdas de entrenamiento.")
    ds_info = record.get("dataset") or {}
    result["life_estimate"] = life.estimate(
        result["predictions"], target, result["similar_cells"],
        nominal=float(ds_info.get("nominal_capacity_ah") or config.NOMINAL_CAPACITY_AH),
        eol_capacity=float(ds_info.get("eol_capacity_ah") or config.EOL_CAPACITY_AH),
        discharge_current_a=_csv_median(df, "discharge_current_a"),
        charge_hours=_charge_hours(df))
    return result
