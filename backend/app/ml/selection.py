"""Selección automática del modelo para un CSV nuevo (v1.2.0, ``POST /api/v1/predict/auto``).

Criterio, en este orden:

1. Solo compiten los experimentos del objetivo pedido cuyas características están todas en el CSV.
2. Se elige el experimento cuyo dominio de entrenamiento está **más cerca** del CSV: menor mediana del
   ratio de distancia (``domain.py``), penalizando las temperaturas y corrientes que el modelo no vio.
3. Si dos candidatos quedan a menos de un 10 % de distancia, gana el de menor MAE de test.

La elección usa solo las características de entrada, **nunca el valor real** del CSV (sería hacer trampa).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from app.errors import AppError
from app.ml import domain, inference, store

TIE_TOLERANCE = 0.10
CONDITION_PENALTY = 2.0
TARGET_COLUMNS = ("soh", "rul", "eol_cycle", "capacity_ah")


def _score(record: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    feats = record["config"]["features"]
    base = {"experiment_id": record["id"], "name": record["name"], "model_type": record["config"]["model_type"],
            "dataset_id": record["config"].get("dataset_id") or "nasa",
            "test_mae": (record.get("metrics", {}).get("test") or {}).get("mae")}
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
        close.sort(key=lambda c: (c["test_mae"] if c["test_mae"] is not None else np.inf))
        ok = close + [c for c in ok if c not in close]
    return ok + [c for c in scored if not c["eligible"]]


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
        reason += "; empató en distancia con otro y ganó por menor MAE de test"
    result["selection"] = {
        "selected_experiment_id": chosen["experiment_id"],
        "criterion": "menor distancia al dominio de entrenamiento (sin usar el valor real); desempate por MAE de test",
        "reason": reason + ".",
        "candidates": candidates[:50],
    }
    return result
