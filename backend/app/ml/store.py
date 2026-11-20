"""Persistencia de experimentos en disco.

Cada experimento vive en ``backend/experiments/<id>/``::

    experiment.json   -> configuración, métricas, importancias, entorno, dataset (huella)
    predictions.csv   -> battery_id, cycle, split, y_true, y_pred
    model.joblib      -> Pipeline de scikit-learn entrenado (imputador + escalador + modelo)

Con ``experiment.json`` + el CSV del dataset (misma huella SHA-256) + ``random_state``
el experimento se puede reproducir exactamente.
"""
from __future__ import annotations

import json
import platform
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from app import config
from app.errors import AppError

_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,100}$")


def _dir(experiment_id: str, base: Path | None = None) -> Path:
    if not _ID_RE.match(experiment_id):
        raise AppError("invalid_experiment_id", "Identificador de experimento inválido.", 400)
    return Path(base or config.EXPERIMENTS_DIR) / experiment_id


def new_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30] or "exp"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}_{slug}_{uuid.uuid4().hex[:6]}"


def environment_info() -> dict[str, str]:
    return {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}


def save(result: dict[str, Any], *, dataset_sha: str, experiment_id: str | None = None,
         base: Path | None = None) -> dict[str, Any]:
    cfg = result["config"]
    experiment_id = experiment_id or new_id(cfg["name"])
    d = _dir(experiment_id, base)
    d.mkdir(parents=True, exist_ok=False)
    record = {
        "id": experiment_id,
        "name": cfg["name"],
        "description": cfg.get("description"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "completed",
        "config": {k: cfg[k] for k in ("target", "model_type", "hyperparameters", "features",
                                        "train_batteries", "test_batteries", "random_state")},
        "metrics": result["metrics"],
        "feature_importance": result["feature_importance"],
        "train_feature_medians": result["train_feature_medians"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "warnings": result["warnings"],
        "dataset": {"file": config.PROCESSED_FILE.name, "sha256_16": dataset_sha},
        "environment": environment_info(),
        "artifacts": {"model": "model.joblib", "predictions": "predictions.csv"},
    }
    joblib.dump(result["pipeline"], d / "model.joblib", compress=3)
    result["predictions"].to_csv(d / "predictions.csv", index=False, float_format="%.6g")
    (d / "experiment.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def load_record(experiment_id: str, base: Path | None = None) -> dict[str, Any]:
    path = _dir(experiment_id, base) / "experiment.json"
    if not path.exists():
        raise AppError("experiment_not_found", f"El experimento '{experiment_id}' no existe.", 404)
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(experiment_id: str, base: Path | None = None) -> pd.DataFrame:
    load_record(experiment_id, base)
    return pd.read_csv(_dir(experiment_id, base) / "predictions.csv")


def load_model(experiment_id: str, base: Path | None = None):
    load_record(experiment_id, base)
    path = _dir(experiment_id, base) / "model.joblib"
    if not path.exists():
        raise AppError("model_missing", "El experimento no tiene modelo guardado.", 409)
    return joblib.load(path)


def model_path(experiment_id: str, base: Path | None = None) -> Path:
    load_record(experiment_id, base)
    return _dir(experiment_id, base) / "model.joblib"


def list_records(base: Path | None = None) -> list[dict[str, Any]]:
    base = Path(base or config.EXPERIMENTS_DIR)
    if not base.exists():
        return []
    out = []
    for p in base.glob("*/experiment.json"):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue  # experimento corrupto: se ignora en el listado
    return sorted(out, key=lambda r: r.get("created_at", ""), reverse=True)


def delete(experiment_id: str, base: Path | None = None) -> None:
    load_record(experiment_id, base)
    shutil.rmtree(_dir(experiment_id, base))
