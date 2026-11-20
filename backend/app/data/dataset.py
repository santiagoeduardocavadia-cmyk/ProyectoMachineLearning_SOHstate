"""Acceso al dataset procesado (CSV) con caché en memoria."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app import config
from app.data.features import FEATURE_NAMES, TARGETS
from app.errors import AppError

REQUIRED_COLUMNS = ["battery_id", "cycle", "capacity_ah", "soh", "rul", *FEATURE_NAMES]


@lru_cache(maxsize=4)
def _load(path_str: str, mtime: float) -> pd.DataFrame:  # mtime invalida la caché si cambia el archivo
    df = pd.read_csv(path_str)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise AppError("dataset_invalid", f"El CSV procesado no tiene las columnas: {missing}", 500)
    return df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)


def get_dataset(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or config.PROCESSED_FILE)
    if not path.exists():
        raise AppError(
            "dataset_not_found",
            "No existe el dataset procesado. Ejecuta 'python -m scripts.prepare_dataset' en backend/.",
            503,
            {"path": str(path)},
        )
    return _load(str(path), path.stat().st_mtime).copy()


def dataset_fingerprint(path: Path | None = None) -> str:
    path = Path(path or config.PROCESSED_FILE)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def battery_ids(df: pd.DataFrame | None = None) -> list[str]:
    df = get_dataset() if df is None else df
    return sorted(df["battery_id"].unique().tolist())


def require_battery(df: pd.DataFrame, battery_id: str) -> pd.DataFrame:
    sub = df[df["battery_id"] == battery_id]
    if sub.empty:
        raise AppError("battery_not_found", f"La batería '{battery_id}' no existe.", 404,
                       {"available": battery_ids(df)})
    return sub


def target_column(target: str) -> str:
    return TARGETS[target]["column"]
