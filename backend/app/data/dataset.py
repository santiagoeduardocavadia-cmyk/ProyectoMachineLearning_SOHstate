"""Acceso a los datasets (CSV de ciclos) con caché en memoria.

Hay dos tipos de dataset:
- Incluidos (versionados en el repo, no se pueden borrar):
  - ``nasa`` (por defecto): B0005, B0006, B0007 y B0018 a 24 °C.
  - ``nasa_multitemp`` (v1.2.0): las 32 celdas del ZIP oficial a 4, 22-24 y 43-44 °C, con corrientes
    de 1, 2 y 4 A. Las celdas ensayadas a dos temperaturas se parten en tramos (``B0042_04C``, ``B0042_22C``).
- Subidos por el usuario (``POST /api/v1/datasets``): se guardan en ``data/uploads/<id>.csv``
  junto con ``<id>.json`` (nombre, capacidad nominal, umbral de EOL, fecha, advertencias).

Un dataset subido debe tener una fila por ciclo con, como mínimo, ``battery_id``, ``cycle`` y
``capacity_ah``, más las características que tenga disponibles. SOH y RUL se recalculan siempre
en el backend con la capacidad nominal y el umbral de fin de vida declarados, para que todas las
fuentes usen la misma definición de los objetivos.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app import config
from app.data.features import FEATURE_NAMES
from app.data.preprocessing import add_targets, clean
from app.errors import AppError

DEFAULT_DATASET_ID = "nasa"
REQUIRED_COLUMNS = ["battery_id", "cycle", "capacity_ah", "soh", "rul"]
UPLOAD_REQUIRED = ["battery_id", "cycle", "capacity_ah"]
TARGET_COLUMNS = {"soh", "rul", "eol_cycle"}
MAX_UPLOAD_ROWS = 200_000
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,100}$")


# ----------------------------------------------------------------------------- rutas y metadatos
def _uploads_dir() -> Path:
    return Path(config.UPLOADS_DIR)


def _paths(dataset_id: str) -> tuple[Path, Path]:
    if not _ID_RE.match(dataset_id):
        raise AppError("invalid_dataset_id", "Identificador de dataset inválido.", 400)
    d = _uploads_dir()
    return d / f"{dataset_id}.csv", d / f"{dataset_id}.json"


MULTITEMP_DATASET_ID = "nasa_multitemp"
BUILTIN_DATASETS: dict[str, dict[str, Any]] = {
    DEFAULT_DATASET_ID: {"name": "NASA PCoE Battery Data Set (B0005, B0006, B0007, B0018)", "is_default": True,
                         "file": lambda: config.PROCESSED_FILE},
    MULTITEMP_DATASET_ID: {"name": "NASA PCoE completo: 32 celdas a 4, 24 y 43 °C (1, 2 y 4 A)", "is_default": False,
                           "file": lambda: config.MULTITEMP_FILE},
}


def _builtin_meta(dataset_id: str) -> dict[str, Any]:
    b = BUILTIN_DATASETS[dataset_id]
    return {
        "id": dataset_id,
        "name": b["name"],
        "source": "nasa",
        "is_default": b["is_default"],
        "created_at": None,
        "nominal_capacity_ah": config.NOMINAL_CAPACITY_AH,
        "eol_capacity_ah": config.EOL_CAPACITY_AH,
        "warnings": [],
    }


def dataset_meta(dataset_id: str | None = None) -> dict[str, Any]:
    dataset_id = dataset_id or DEFAULT_DATASET_ID
    if dataset_id in BUILTIN_DATASETS:
        return _builtin_meta(dataset_id)
    csv_path, meta_path = _paths(dataset_id)
    if not meta_path.exists() or not csv_path.exists():
        raise AppError("dataset_not_found", f"El dataset '{dataset_id}' no existe.", 404)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def dataset_path(dataset_id: str | None = None) -> Path:
    dataset_id = dataset_id or DEFAULT_DATASET_ID
    if dataset_id in BUILTIN_DATASETS:
        path = Path(BUILTIN_DATASETS[dataset_id]["file"]())
        if not path.exists():
            raise AppError(
                "dataset_not_found",
                "No existe el dataset procesado. Ejecuta 'python -m scripts.prepare_dataset --multitemp' en backend/.",
                503, {"path": str(path)},
            )
        return path
    dataset_meta(dataset_id)
    return _paths(dataset_id)[0]


# ----------------------------------------------------------------------------- lectura
@lru_cache(maxsize=8)
def _load(path_str: str, mtime: float) -> pd.DataFrame:  # mtime invalida la caché si cambia el archivo
    df = pd.read_csv(path_str, dtype={"battery_id": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise AppError("dataset_invalid", f"El CSV del dataset no tiene las columnas: {missing}", 500)
    # Datasets guardados con una versión anterior (sin las características nuevas): quedan vacías.
    for f in FEATURE_NAMES:
        if f not in df.columns:
            df[f] = np.nan
    return df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)


def get_dataset(dataset_id: str | None = None) -> pd.DataFrame:
    path = dataset_path(dataset_id)
    return _load(str(path), path.stat().st_mtime).copy()


def dataset_fingerprint(dataset_id: str | None = None) -> str:
    return hashlib.sha256(dataset_path(dataset_id).read_bytes()).hexdigest()[:16]


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
    from app.data.features import TARGETS

    return TARGETS[target]["column"]


def describe(dataset_id: str | None = None) -> dict[str, Any]:
    """Metadatos + tamaño + baterías de un dataset (lo que devuelve GET /datasets)."""
    meta = dataset_meta(dataset_id)
    df = get_dataset(meta["id"])
    batteries = []
    for b, g in df.groupby("battery_id"):
        eol = g["eol_cycle"].iloc[0] if "eol_cycle" in g else np.nan
        temp = g["ambient_temperature"].median()
        batteries.append({"battery_id": b, "n_cycles": int(len(g)), "reaches_eol": not pd.isna(eol),
                          "ambient_temperature": None if pd.isna(temp) else float(temp)})
    temps = sorted({round(t["ambient_temperature"]) for t in batteries if t["ambient_temperature"] is not None})
    return {**meta, "n_rows": int(len(df)), "n_batteries": len(batteries), "batteries": batteries,
            "temperatures": temps,
            "sha256_16": dataset_fingerprint(meta["id"])}


def list_datasets() -> list[dict[str, Any]]:
    out = []
    for builtin in BUILTIN_DATASETS:
        try:
            out.append(describe(builtin))
        except AppError:
            pass
    d = _uploads_dir()
    if d.exists():
        metas = []
        for p in d.glob("*.json"):
            try:
                metas.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        for m in sorted(metas, key=lambda m: m.get("created_at") or "", reverse=True):
            try:
                out.append(describe(m["id"]))
            except (AppError, KeyError):
                continue
    return out


# ----------------------------------------------------------------------------- subida
def _new_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", name.lower()).encode("ascii", "ignore").decode()).strip("-")[:30] or "dataset"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"up_{stamp}_{slug}_{uuid.uuid4().hex[:6]}"


def parse_upload(raw: bytes, nominal_capacity: float, eol_capacity: float) -> tuple[pd.DataFrame, list[str]]:
    """Valida y normaliza un CSV subido. Devuelve el dataframe listo y las advertencias."""
    warnings: list[str] = []
    if not raw.strip():
        raise AppError("invalid_file", "El CSV está vacío.", 422)
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype={"battery_id": str})
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de parseo es entrada inválida
        raise AppError("invalid_file", f"No se pudo leer el CSV: {exc}", 422) from exc
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in UPLOAD_REQUIRED if c not in df.columns]
    if missing:
        raise AppError("invalid_dataset", f"Faltan columnas obligatorias: {missing}", 422,
                       {"required": UPLOAD_REQUIRED, "optional_features": FEATURE_NAMES})
    if len(df) > MAX_UPLOAD_ROWS:
        raise AppError("invalid_dataset", f"El CSV supera {MAX_UPLOAD_ROWS} filas.", 422)

    dropped_targets = sorted(TARGET_COLUMNS & set(df.columns))
    if dropped_targets:
        warnings.append(f"Las columnas {dropped_targets} se recalcularon en el backend a partir de capacity_ah.")
    extra = sorted(set(df.columns) - set(UPLOAD_REQUIRED) - set(FEATURE_NAMES) - TARGET_COLUMNS)
    if extra:
        warnings.append(f"Columnas ignoradas (no son características conocidas): {extra}.")

    present = [f for f in FEATURE_NAMES if f in df.columns and f != "cycle"]
    if not present:
        raise AppError("invalid_dataset", "El CSV no tiene ninguna característica conocida además de 'cycle'.", 422,
                       {"optional_features": FEATURE_NAMES})
    absent = [f for f in FEATURE_NAMES if f not in df.columns]
    if absent:
        warnings.append(f"Características ausentes (quedan vacías y no se podrán usar para entrenar): {absent}.")

    df["battery_id"] = df["battery_id"].astype(str).str.strip()
    if (df["battery_id"].isin(["", "nan", "None"])).any():
        raise AppError("invalid_dataset", "Hay filas sin battery_id.", 422)

    numeric_cols = ["cycle", "capacity_ah", *present]
    bad = {}
    for c in numeric_cols:
        conv = pd.to_numeric(df[c], errors="coerce")
        n_bad = int((conv.isna() & df[c].notna()).sum())
        if n_bad:
            bad[c] = n_bad
        df[c] = conv
    if bad:
        raise AppError("invalid_dataset", "Hay valores no numéricos en columnas numéricas.", 422,
                       {"columns": bad})
    for c in numeric_cols:
        if np.isinf(df[c]).any():
            raise AppError("invalid_dataset", f"La columna '{c}' tiene valores infinitos.", 422)

    if df["cycle"].isna().any() or (df["cycle"] < 1).any() or (df["cycle"] % 1 != 0).any():
        raise AppError("invalid_dataset", "'cycle' debe ser un entero >= 1 en todas las filas.", 422)
    df["cycle"] = df["cycle"].astype(int)
    dups = df.duplicated(["battery_id", "cycle"])
    if dups.any():
        ex = df.loc[dups, ["battery_id", "cycle"]].head(5).to_dict("records")
        raise AppError("invalid_dataset", "Hay ciclos repetidos para una misma batería.", 422, {"examples": ex})

    n_before = len(df)
    df = df[df["capacity_ah"].notna()]
    df = clean(df, nominal_capacity)
    if len(df) < n_before:
        warnings.append(f"Se descartaron {n_before - len(df)} filas sin capacidad o con capacidad fuera de "
                        f"(0, {1.2 * nominal_capacity:.3g}] Ah.")

    for f in absent:
        df[f] = np.nan
    df = df.drop(columns=[c for c in df.columns if c in TARGET_COLUMNS], errors="ignore")
    keep = ["battery_id", "cycle", *[f for f in FEATURE_NAMES if f != "cycle"], "capacity_ah"]
    df = add_targets(df[keep], eol_capacity=eol_capacity, nominal_capacity=nominal_capacity)

    n_batt = df["battery_id"].nunique()
    if len(df) < 10 or n_batt < 2:
        raise AppError("invalid_dataset",
                       "Se necesitan al menos 10 ciclos válidos y 2 baterías (una para entrenar y otra para test).",
                       422, {"rows": int(len(df)), "batteries": int(n_batt)})
    censored = sorted(df.loc[df["rul"].isna(), "battery_id"].unique().tolist())
    if censored:
        warnings.append(f"Baterías que no llegan al fin de vida ({eol_capacity} Ah), sin RUL: {censored}.")
    return df, warnings


def save_upload(raw: bytes, name: str, nominal_capacity: float | None, eol_capacity: float | None) -> dict[str, Any]:
    nominal = float(nominal_capacity) if nominal_capacity is not None else config.NOMINAL_CAPACITY_AH
    eol = float(eol_capacity) if eol_capacity is not None else round(0.7 * nominal, 6)
    if not (0 < nominal <= 1000) or not (0 < eol < nominal):
        raise AppError("invalid_dataset", "Se requiere 0 < eol_capacity_ah < nominal_capacity_ah <= 1000.", 422,
                       {"nominal_capacity_ah": nominal, "eol_capacity_ah": eol})
    df, warnings = parse_upload(raw, nominal, eol)
    dataset_id = _new_id(name)
    csv_path, meta_path = _paths(dataset_id)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.6g")
    meta = {
        "id": dataset_id, "name": name, "source": "upload", "is_default": False,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nominal_capacity_ah": nominal, "eol_capacity_ah": eol, "warnings": warnings,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return describe(dataset_id)


def delete_dataset(dataset_id: str) -> None:
    if dataset_id in BUILTIN_DATASETS:
        raise AppError("dataset_protected", "Los datasets NASA incluidos no se pueden borrar.", 403)
    dataset_meta(dataset_id)
    for p in _paths(dataset_id):
        p.unlink(missing_ok=True)


def template_csv() -> str:
    """Plantilla con el formato esperado: 3 filas reales del dataset NASA."""
    df = get_dataset(DEFAULT_DATASET_ID)
    cols = ["battery_id", "cycle", "capacity_ah", *[f for f in FEATURE_NAMES if f != "cycle"]]
    return df[df["battery_id"] == "B0005"].head(3)[cols].to_csv(index=False, float_format="%.6g")
