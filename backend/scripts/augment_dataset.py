"""Genera una versión aumentada (*data augmentation*) de un CSV de ciclos.

Uso (desde backend/):
    python -m scripts.augment_dataset ENTRADA.csv --test B0006 [--out SALIDA.csv]

El resultado tiene el mismo formato de subida (``GET /api/v1/datasets/template``), así
que se sube por la página como cualquier otro dataset y se entrena con él. Las filas
sintéticas se guardan como **baterías virtuales** con su propio ``battery_id``
(el formato exige ciclos enteros y sin repetir por batería):

- ``B0005~ruido1``   copia de B0005 con ruido gaussiano en las características
                     (N(0, (nivel · σ_característica)²)); la capacidad no cambia.
                     Simula el error de medida de los sensores.
- ``B0005~interp1``  cada ciclo k es un punto aleatorio entre los ciclos k y k+1 de
                     B0005 (características y capacidad). La degradación es suave.
- ``B0005~mix1``     mezcla de B0005 con otra batería (λ ~ Beta(0.4, 0.4), λ ≥ 0.5),
                     emparejando ciclos con capacidad parecida. Crea una batería
                     "intermedia", que es lo que más falta con tan pocas celdas.

IMPORTANTE (fuga de información): las baterías de ``--test`` se copian tal cual y
NO se usan para generar baterías virtuales. Si una batería virtual se hiciera a partir
de la batería de test, el modelo vería versiones casi idénticas del test y las métricas
saldrían infladas. Por eso, al entrenar con el archivo aumentado, usa como test
exactamente las baterías indicadas en ``--test``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.features import FEATURE_NAMES  # noqa: E402

SEP = "~"  # separa la batería original del tipo de copia en el battery_id
MIXUP_ALPHA = 0.4
TARGET_COLUMNS = ["soh", "rul", "eol_cycle"]  # se recalculan en el backend al subir


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [f for f in FEATURE_NAMES if f != "cycle" and f in df.columns and df[f].notna().any()]


def _noise(g: pd.DataFrame, feats: list[str], scale: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    d = g.copy()
    d[feats] = d[feats].to_numpy() + rng.normal(size=(len(d), len(feats))) * scale
    return d


def _interpolation(g: pd.DataFrame, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    g = g.sort_values("cycle")
    a, b = g.iloc[:-1], g.iloc[1:]
    lam = rng.uniform(size=(len(a), 1))
    d = a.copy()
    d[cols] = a[cols].to_numpy() * lam + b[cols].to_numpy() * (1 - lam)
    return d


def _mixup(g: pd.DataFrame, partner: pd.DataFrame, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    partner = partner.sort_values("capacity_ah")
    idx = np.searchsorted(partner["capacity_ah"].to_numpy(), g["capacity_ah"].to_numpy()).clip(0, len(partner) - 1)
    p = partner.iloc[idx]
    lam = rng.beta(MIXUP_ALPHA, MIXUP_ALPHA)
    lam = max(lam, 1 - lam)  # un solo λ por batería virtual: es una celda "intermedia" coherente
    d = g.copy()
    d[cols] = g[cols].to_numpy() * lam + p[cols].to_numpy() * (1 - lam)
    return d


def augment(df: pd.DataFrame, test_batteries: list[str], *, noise_copies: int = 3, noise_level: float = 0.03,
            interpolation_copies: int = 2, mixup_copies: int = 2, seed: int = 42) -> pd.DataFrame:
    """Devuelve las filas reales + las baterías virtuales generadas a partir de las de entrenamiento."""
    df = df.drop(columns=[c for c in TARGET_COLUMNS if c in df.columns])
    df["battery_id"] = df["battery_id"].astype(str)
    if df["battery_id"].str.contains(SEP, regex=False).any():
        raise ValueError(f"El CSV ya contiene baterías virtuales ('{SEP}' en battery_id): no se aumenta dos veces.")
    unknown = sorted(set(test_batteries) - set(df["battery_id"]))
    if unknown:
        raise ValueError(f"Baterías de test inexistentes: {unknown}. Disponibles: {sorted(df['battery_id'].unique())}")

    feats = _feature_cols(df)
    cols = [*feats, "capacity_ah"]
    df[cols] = df[cols].astype(float)
    train = df[~df["battery_id"].isin(test_batteries)]
    if train.empty:
        raise ValueError("No quedan baterías de entrenamiento para aumentar.")
    rng = np.random.default_rng(seed)
    scale = (train[feats].std().fillna(0.0) * noise_level).to_numpy()
    groups = {b: g for b, g in train.groupby("battery_id", sort=True)}

    virtual = []
    for b, g in groups.items():
        for i in range(1, noise_copies + 1):
            virtual.append(_noise(g, feats, scale, rng).assign(battery_id=f"{b}{SEP}ruido{i}"))
        if len(g) >= 2:
            for i in range(1, interpolation_copies + 1):
                virtual.append(_interpolation(g, cols, rng).assign(battery_id=f"{b}{SEP}interp{i}"))
        others = [o for o in groups if o != b]
        for i in range(1, mixup_copies + 1 if others else 1):
            partner = groups[others[rng.integers(len(others))]]
            virtual.append(_mixup(g, partner, cols, rng).assign(battery_id=f"{b}{SEP}mix{i}"))

    out = pd.concat([df, *virtual], ignore_index=True)
    out["cycle"] = out["cycle"].astype(int)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="CSV de ciclos (formato de subida o el procesado de la NASA).")
    ap.add_argument("--test", nargs="+", required=True,
                    help="Baterías reservadas para test: se copian sin tocar y no generan baterías virtuales.")
    ap.add_argument("--out", type=Path, help="CSV de salida (por defecto <entrada>_aumentado_test-<ids>.csv).")
    ap.add_argument("--noise-copies", type=int, default=3)
    ap.add_argument("--noise-level", type=float, default=0.03, help="Fracción de la σ de cada característica.")
    ap.add_argument("--interpolation-copies", type=int, default=2)
    ap.add_argument("--mixup-copies", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    df = pd.read_csv(a.input, dtype={"battery_id": str})
    out = augment(df, a.test, noise_copies=a.noise_copies, noise_level=a.noise_level,
                  interpolation_copies=a.interpolation_copies, mixup_copies=a.mixup_copies, seed=a.seed)
    path = a.out or a.input.with_name(f"{a.input.stem}_aumentado_test-{'-'.join(a.test)}.csv")
    out.to_csv(path, index=False, float_format="%.6g")
    n_real = len(df)
    print(f"{path}: {n_real} ciclos reales + {len(out) - n_real} sintéticos "
          f"({out['battery_id'].nunique() - df['battery_id'].nunique()} baterías virtuales). "
          f"Entrena con test = {a.test}.")


if __name__ == "__main__":
    main()
