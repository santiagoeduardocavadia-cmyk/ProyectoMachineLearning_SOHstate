"""Convierte el dataset de Kaggle "NASA Battery Dataset" (carpeta cleaned_dataset) al CSV que acepta
POST /api/v1/datasets (una fila por ciclo de descarga con las mismas 14 características).

Fuente: https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset
Es una reorganización en CSV de los mismos datos del NASA PCoE. Estructura esperada:

    cleaned_dataset/
      metadata.csv          type, start_time, ambient_temperature, battery_id, test_id, uid,
                            filename, Capacity, Re, Rct
      data/00001.csv ...    una serie temporal por operación (charge / discharge / impedance)

Uso (desde backend/):
    python -m scripts.convert_kaggle RUTA/cleaned_dataset --batteries B0029 B0030 B0031 B0032 --out mis_baterias.csv
Sin --batteries convierte todas las baterías (puede tardar varios minutos).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.features import FEATURE_NAMES, charge_features, discharge_features  # noqa: E402


def _num(x) -> float:
    """Kaggle guarda Re/Rct a veces como complejos en texto, p. ej. '(0.056+0j)'."""
    try:
        return float(np.real(complex(str(x).strip().strip("()").replace(" ", ""))))
    except (TypeError, ValueError):
        return float("nan")


def _series(path: Path) -> dict:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return {c: df[c].to_numpy(dtype=float) for c in df.columns if c in
            ("Voltage_measured", "Current_measured", "Temperature_measured", "Time")}


def convert(root: Path, batteries: list[str] | None) -> pd.DataFrame:
    meta = pd.read_csv(root / "metadata.csv")
    meta.columns = [c.strip() for c in meta.columns]
    # orden cronológico real dentro de cada batería (test_id es el índice de la operación)
    meta = meta.sort_values(["battery_id", "test_id"])
    if batteries:
        missing = sorted(set(batteries) - set(meta["battery_id"]))
        if missing:
            raise SystemExit(f"Baterías no encontradas en metadata.csv: {missing}")
        meta = meta[meta["battery_id"].isin(batteries)]

    rows = []
    for battery, g in meta.groupby("battery_id", sort=True):
        last_charge = None
        last_imp = {"re_ohm": float("nan"), "rct_ohm": float("nan")}
        n = 0
        for _, op in g.iterrows():
            path = root / "data" / str(op["filename"])
            kind = str(op["type"]).strip()
            if kind == "impedance":
                re_, rct = _num(op.get("Re")), _num(op.get("Rct"))
                if 0 < re_ < 1 and 0 < rct < 1:
                    last_imp = {"re_ohm": re_, "rct_ohm": rct}
            elif kind == "charge":
                last_charge = _series(path) if path.exists() else None
            elif kind == "discharge":
                cap = _num(op.get("Capacity"))
                if not path.exists() or np.isnan(cap):
                    continue
                n += 1
                try:
                    feats = discharge_features(_series(path))
                except (ValueError, KeyError):
                    continue
                rows.append({"battery_id": battery, "cycle": n,
                             "ambient_temperature": float(op["ambient_temperature"]),
                             **feats, **charge_features(last_charge), **last_imp, "capacity_ah": cap})
                last_charge = None
        print(f"{battery}: {n} descargas")
    cols = ["battery_id", "cycle", "capacity_ah", "ambient_temperature", *[f for f in FEATURE_NAMES if f != "cycle"]]
    return pd.DataFrame(rows)[cols]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="carpeta cleaned_dataset (contiene metadata.csv y data/)")
    p.add_argument("--batteries", nargs="*", help="p. ej. B0029 B0030 B0031 B0032")
    p.add_argument("--out", type=Path, default=Path("dataset_kaggle.csv"))
    a = p.parse_args()
    df = convert(a.root, a.batteries)
    df.to_csv(a.out, index=False, float_format="%.6g")
    print(f"{len(df)} ciclos de {df['battery_id'].nunique()} baterías -> {a.out}")


if __name__ == "__main__":
    main()
