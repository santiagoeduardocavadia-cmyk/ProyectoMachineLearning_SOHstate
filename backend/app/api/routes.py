"""Endpoints HTTP de la API v1 (contrato documentado en CONTEXTO_CHATGPT.md)."""
from __future__ import annotations

import io
import math
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import FileResponse

from app import __version__, config
from app.data.dataset import battery_ids, dataset_fingerprint, get_dataset, require_battery
from app.data.features import FEATURES, TARGETS
from app.errors import AppError
from app.ml import inference, store
from app.ml.models import MODELS
from app.ml.training import train_experiment
from app.schemas import ExperimentCreate, PredictRequest

router = APIRouter(prefix=config.API_PREFIX)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _clean(value: Any) -> Any:
    """Convierte NaN/inf en None para que el JSON sea válido."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    return _clean(df.to_dict(orient="records"))


def _summary(r: dict) -> dict:
    return {
        "id": r["id"], "name": r["name"], "created_at": r["created_at"], "status": r["status"],
        "target": r["config"]["target"], "model_type": r["config"]["model_type"],
        "test_batteries": r["config"]["test_batteries"], "metrics_test": r["metrics"]["test"],
    }


# ----------------------------------------------------------------------------- sistema
@router.get("/health", tags=["sistema"])
def health():
    try:
        df = get_dataset()
        ds = {"loaded": True, "rows": int(len(df))}
    except AppError as exc:
        ds = {"loaded": False, "error": exc.message}
    return {"status": "ok", "api_version": config.API_VERSION, "app_version": __version__, "dataset": ds}


# ----------------------------------------------------------------------------- dataset
@router.get("/dataset/summary", tags=["dataset"])
def dataset_summary():
    df = get_dataset()
    batteries = []
    for b, g in df.groupby("battery_id"):
        eol = g["eol_cycle"].iloc[0]
        batteries.append({
            "battery_id": b, "n_cycles": int(len(g)),
            "initial_capacity_ah": float(g["capacity_ah"].iloc[0]), "final_capacity_ah": float(g["capacity_ah"].iloc[-1]),
            "min_soh": float(g["soh"].min()), "eol_cycle": None if pd.isna(eol) else int(eol),
            "reaches_eol": not pd.isna(eol), "ambient_temperature": float(g["ambient_temperature"].iloc[0]),
        })
    return {
        "source": "NASA Prognostics Center of Excellence - Battery Data Set (B0005, B0006, B0007, B0018)",
        "source_url": config.NASA_DATASET_URL,
        "n_rows": int(len(df)), "n_batteries": len(batteries),
        "nominal_capacity_ah": config.NOMINAL_CAPACITY_AH, "eol_capacity_ah": config.EOL_CAPACITY_AH,
        "sha256_16": dataset_fingerprint(), "batteries": batteries,
    }


@router.get("/dataset/features", tags=["dataset"])
def dataset_features():
    df = get_dataset()
    feats = []
    for f in FEATURES:
        s = df[f.name]
        feats.append({"name": f.name, "unit": f.unit, "description": f.description, "source": f.source,
                      "min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()),
                      "missing": int(s.isna().sum())})
    targets = [{"key": k, **v} for k, v in TARGETS.items()]
    return {"features": feats, "targets": targets}


@router.get("/dataset/batteries/{battery_id}/degradation", tags=["dataset"])
def battery_degradation(battery_id: str):
    sub = require_battery(get_dataset(), battery_id)
    eol = sub["eol_cycle"].iloc[0]
    return {
        "battery_id": battery_id,
        "eol_capacity_ah": config.EOL_CAPACITY_AH,
        "eol_cycle": None if pd.isna(eol) else int(eol),
        "points": _records(sub[["cycle", "capacity_ah", "soh", "rul"]]),
    }


@router.get("/dataset/cycles", tags=["dataset"])
def dataset_cycles(battery_id: Optional[str] = Query(None), limit: int = Query(1000, ge=1, le=5000),
                   offset: int = Query(0, ge=0)):
    df = get_dataset()
    if battery_id:
        df = require_battery(df, battery_id)
    page = df.iloc[offset: offset + limit]
    return {"total": int(len(df)), "offset": offset, "limit": limit, "rows": _records(page)}


# ----------------------------------------------------------------------------- modelos
@router.get("/models", tags=["modelos"])
def list_models():
    return {"models": [m.to_dict() for m in MODELS.values()],
            "targets": [{"key": k, **v} for k, v in TARGETS.items()],
            "batteries": battery_ids(),
            "defaults": ExperimentCreate(name="ejemplo").model_dump()}


# ----------------------------------------------------------------------------- experimentos
@router.post("/experiments", status_code=201, tags=["experimentos"])
def create_experiment(body: ExperimentCreate):
    df = get_dataset()
    result = train_experiment(df, body.model_dump())
    return _clean(store.save(result, dataset_sha=dataset_fingerprint()))


@router.get("/experiments", tags=["experimentos"])
def list_experiments(target: Optional[str] = Query(None, pattern="^(soh|rul|capacity)$")):
    recs = store.list_records()
    if target:
        recs = [r for r in recs if r["config"]["target"] == target]
    return {"experiments": [_clean(_summary(r)) for r in recs]}


@router.get("/experiments/{experiment_id}", tags=["experimentos"])
def get_experiment(experiment_id: str):
    return _clean(store.load_record(experiment_id))


@router.delete("/experiments/{experiment_id}", status_code=204, tags=["experimentos"])
def delete_experiment(experiment_id: str):
    store.delete(experiment_id)


@router.get("/experiments/{experiment_id}/predictions", tags=["experimentos"])
def experiment_predictions(experiment_id: str, split: Optional[str] = Query(None, pattern="^(train|test)$"),
                           battery_id: Optional[str] = None):
    rec = store.load_record(experiment_id)
    df = store.load_predictions(experiment_id)
    if split:
        df = df[df["split"] == split]
    if battery_id:
        df = df[df["battery_id"] == battery_id]
    df = df.assign(error=df["y_pred"] - df["y_true"])
    return {"experiment_id": experiment_id, "target": rec["config"]["target"], "points": _records(df)}


@router.get("/experiments/{experiment_id}/feature-importance", tags=["experimentos"])
def experiment_importance(experiment_id: str):
    rec = store.load_record(experiment_id)
    return {"experiment_id": experiment_id, "method": "permutation_importance (test, ΔMAE)",
            "importances": _clean(rec["feature_importance"])}


@router.get("/experiments/{experiment_id}/model", tags=["experimentos"])
def download_model(experiment_id: str):
    path = store.model_path(experiment_id)
    return FileResponse(path, media_type="application/octet-stream", filename=f"{experiment_id}.joblib")


@router.post("/experiments/{experiment_id}/predict", tags=["inferencia"])
def predict(experiment_id: str, body: PredictRequest):
    return _clean(inference.predict(experiment_id, samples=body.samples, battery_id=body.battery_id,
                                    cycles=body.cycles))


@router.post("/experiments/{experiment_id}/predict/csv", tags=["inferencia"])
async def predict_csv(experiment_id: str, file: UploadFile = File(...)):
    if file.filename and not file.filename.lower().endswith(".csv"):
        raise AppError("invalid_file", "El archivo debe ser .csv", 415)
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise AppError("file_too_large", "El CSV supera 5 MB.", 413)
    if not raw.strip():
        raise AppError("invalid_file", "El CSV está vacío.", 422)
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de parseo es entrada inválida
        raise AppError("invalid_file", f"No se pudo leer el CSV: {exc}", 422) from exc
    if df.empty:
        raise AppError("invalid_file", "El CSV no tiene filas.", 422)
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    rec = store.load_record(experiment_id)
    bad = [c for c in non_numeric if c in rec["config"]["features"]]
    if bad:
        raise AppError("invalid_file", f"Columnas no numéricas: {bad}", 422)
    numeric = df.drop(columns=non_numeric)
    samples = [{k: (None if pd.isna(v) else float(v)) for k, v in row.items()} for row in numeric.to_dict("records")]
    result = inference.predict(experiment_id, samples=samples)
    if "battery_id" in df.columns:
        for item, b in zip(result["predictions"], df["battery_id"].astype(str)):
            item["battery_id"] = b
    if "cycle" in df.columns:
        for item, c in zip(result["predictions"], df["cycle"]):
            item["cycle"] = None if pd.isna(c) else int(c)
    return _clean(result)
