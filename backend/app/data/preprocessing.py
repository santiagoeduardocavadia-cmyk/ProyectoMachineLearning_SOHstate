"""Descarga y preprocesamiento del NASA Battery Dataset.

Flujo (documentado en ``backend/docs/PREPROCESAMIENTO.md``):

1. Descargar el ZIP oficial del PCoE de la NASA (≈210 MB) a ``data/raw``.
2. Extraer los archivos ``B0005.mat``, ``B0006.mat``, ``B0007.mat``, ``B0018.mat``.
3. Recorrer los ciclos en orden cronológico. Para cada ciclo de *descarga*:
   - asociar la carga inmediatamente anterior y la última impedancia medida;
   - calcular las características (``features.py``);
   - calcular los objetivos: capacidad, SOH y RUL.
4. Limpiar: descartar ciclos con capacidad físicamente imposible y escribir
   ``data/processed/nasa_battery_cycles.csv``.
"""
from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from app import config
from app.data.features import charge_features, discharge_features, impedance_features

log = logging.getLogger(__name__)


def download_raw(raw_dir: Path = config.RAW_DIR, url: str = config.NASA_DATASET_URL) -> Path:
    """Descarga el ZIP de la NASA y extrae los .mat de las baterías 5, 6, 7 y 18."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if all((raw_dir / f"{b}.mat").exists() for b in config.DEFAULT_BATTERIES):
        log.info("Archivos .mat ya presentes en %s", raw_dir)
        return raw_dir
    outer = raw_dir / "nasa_battery_dataset.zip"
    if not outer.exists():
        log.info("Descargando %s ...", url)
        urllib.request.urlretrieve(url, outer)  # noqa: S310 (URL fija y oficial)
    with zipfile.ZipFile(outer) as zf:
        inner_name = next(n for n in zf.namelist() if n.endswith(config.NASA_INNER_ZIP))
        inner_path = raw_dir / config.NASA_INNER_ZIP
        inner_path.write_bytes(zf.read(inner_name))
    with zipfile.ZipFile(inner_path) as zf:
        for member in zf.namelist():
            if member.endswith(".mat") or member.endswith(".txt"):
                (raw_dir / Path(member).name).write_bytes(zf.read(member))
    return raw_dir


def load_mat_cycles(mat_path: Path) -> list[dict]:
    from scipy.io import loadmat

    battery_id = mat_path.stem
    mat = loadmat(mat_path, simplify_cells=True)
    if battery_id not in mat:
        raise ValueError(f"{mat_path.name} no contiene la estructura '{battery_id}'")
    return list(mat[battery_id]["cycle"])


def process_battery(battery_id: str, cycles: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    last_charge = None
    last_impedance = None
    n_discharge = 0
    for c in cycles:
        ctype = c["type"]
        if ctype == "charge":
            last_charge = c["data"]
        elif ctype == "impedance":
            imp = impedance_features(c["data"])
            if not np.isnan(imp["re_ohm"]):
                last_impedance = c["data"]
        elif ctype == "discharge":
            data = c["data"]
            capacity = float(np.asarray(data.get("Capacity", np.nan)).ravel()[0])
            n_discharge += 1
            try:
                feats = discharge_features(data)
            except ValueError as exc:
                log.warning("%s ciclo %d descartado: %s", battery_id, n_discharge, exc)
                continue
            row = {
                "battery_id": battery_id,
                "cycle": n_discharge,
                "ambient_temperature": float(c["ambient_temperature"]),
                **feats,
                **charge_features(last_charge),
                **impedance_features(last_impedance),
                "capacity_ah": capacity,
            }
            rows.append(row)
            last_charge = None  # cada carga se asocia solo a la descarga siguiente
    return pd.DataFrame(rows)


def add_targets(df: pd.DataFrame, eol_capacity: float = config.EOL_CAPACITY_AH) -> pd.DataFrame:
    """Calcula SOH y RUL por batería.

    - ``soh = capacity_ah / 2.0`` (capacidad nominal).
    - ``rul``: ciclos que faltan para el primer ciclo con capacidad < EOL.
      Si la batería nunca llega al EOL (p. ej. B0007), el RUL es desconocido
      (censurado) y se deja como NaN: esas filas no se usan para entrenar RUL.
      Después del EOL el RUL es 0.
    """
    out = []
    for _, g in df.groupby("battery_id", sort=True):
        g = g.sort_values("cycle").copy()
        g["soh"] = g["capacity_ah"] / config.NOMINAL_CAPACITY_AH
        below = g.loc[g["capacity_ah"] < eol_capacity, "cycle"]
        if below.empty:
            g["rul"] = np.nan
            g["eol_cycle"] = np.nan
        else:
            eol = int(below.iloc[0])
            g["rul"] = np.clip(eol - g["cycle"], 0, None).astype(float)
            g["eol_cycle"] = float(eol)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con capacidad ausente o físicamente imposible (> 1.2 × nominal o <= 0)."""
    mask = df["capacity_ah"].between(0.01, 1.2 * config.NOMINAL_CAPACITY_AH)
    dropped = int((~mask).sum())
    if dropped:
        log.warning("Se descartaron %d ciclos con capacidad inválida", dropped)
    return df.loc[mask].reset_index(drop=True)


def build_processed_dataset(raw_dir: Path = config.RAW_DIR, out_file: Path = config.PROCESSED_FILE,
                            batteries: list[str] | None = None) -> pd.DataFrame:
    batteries = batteries or config.DEFAULT_BATTERIES
    frames = []
    for b in batteries:
        path = raw_dir / f"{b}.mat"
        if not path.exists():
            raise FileNotFoundError(f"No existe {path}. Ejecuta primero la descarga.")
        frames.append(process_battery(b, load_mat_cycles(path)))
    df = add_targets(clean(pd.concat(frames, ignore_index=True)))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False, float_format="%.6g")
    return df
