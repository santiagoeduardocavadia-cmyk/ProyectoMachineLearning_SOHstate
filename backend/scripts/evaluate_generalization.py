"""Mide cómo generaliza el modelo de SOH a condiciones no vistas (v1.2.0).

Uso (desde backend/):
    python -m scripts.evaluate_generalization [--out-dir docs/generalizacion]

Usa el dataset ``nasa_multitemp`` (32 celdas a 4, 22-24 y 43-44 °C; 1, 2 y 4 A) y tres esquemas:

1. **Solo 24 °C → otras temperaturas** (la situación de v1.1): entrenar con B0005/6/7/18 y probar en
   todas las celdas a 4 °C y a 43 °C. Es lo que vio el profesor cuando "el modelo alucina".
2. **Dejando fuera una temperatura completa**: entrenar con dos temperaturas y probar en la tercera.
   Es el caso más difícil: el modelo nunca vio esa temperatura.
3. **Dejando fuera celdas (5 pliegues por celda física)**: entrenar con todas las temperaturas y probar
   en celdas nuevas. Es el uso real del modelo multitemperatura: una celda nueva en condiciones conocidas.

Cada esquema se repite con las 14 características originales y con las 16 de v1.2 (más temperatura
ambiente y corriente de descarga), con y sin *data augmentation* (``scripts/augment_dataset.py``,
aplicado solo a las celdas de entrenamiento). Métrica: MAE de SOH en celdas reales de test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import NOMINAL_CAPACITY_AH  # noqa: E402
from app.data.dataset import get_dataset  # noqa: E402
from app.data.features import FEATURE_NAMES, LEGACY_FEATURES  # noqa: E402
from app.ml.domain import physical_cell  # noqa: E402
from app.ml.models import build_pipeline, resolve_hyperparameters  # noqa: E402
from scripts.augment_dataset import augment  # noqa: E402

MODELS = ["ridge", "random_forest", "gradient_boosting"]
FEATURE_SETS = {"14 originales": LEGACY_FEATURES, "16 (+ temperatura y corriente)": FEATURE_NAMES}
SEED = 42


def temp_group(t: float) -> int:
    return 4 if t < 14 else 24 if t < 34 else 43


def _fit_predict(model: str, feats: list[str], train: pd.DataFrame, test: pd.DataFrame, aug: bool) -> np.ndarray:
    if aug:
        train = augment(train.copy(), [])  # el test nunca entra: solo se aumentan las celdas de entrenamiento
        train["soh"] = train["capacity_ah"] / NOMINAL_CAPACITY_AH
    usable = [f for f in feats if train[f].notna().any()]
    pipe = build_pipeline(model, resolve_hyperparameters(model, {}), SEED)
    pipe.fit(train[usable], train["soh"])
    return pipe.predict(test[usable])


def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def run(out_dir: Path) -> pd.DataFrame:
    multi = get_dataset("nasa_multitemp")
    multi = multi[multi["soh"].notna()].copy()
    multi["tgroup"] = multi["ambient_temperature"].map(temp_group)
    nasa = get_dataset("nasa")
    rows = []

    def add(scheme, held, model, fset, aug, y, p, n_cells):
        rows.append({"esquema": scheme, "test": held, "modelo": model, "caracteristicas": fset,
                     "augmentation": "sí" if aug else "no", "mae_soh": _mae(y, p), "n_ciclos": len(y),
                     "n_celdas_test": n_cells})

    for model in MODELS:
        for fset, feats in FEATURE_SETS.items():
            for aug in (False, True):
                # 1. solo 24 °C (NASA original) -> otras temperaturas
                for tg in (4, 43):
                    test = multi[multi["tgroup"] == tg]
                    p = _fit_predict(model, feats, nasa, test, aug)
                    add("1. entrenado solo a 24 °C", f"{tg} °C", model, fset, aug, test["soh"], p,
                        test["battery_id"].map(physical_cell).nunique())
                # 2. dejando fuera una temperatura completa
                for tg in (4, 24, 43):
                    train, test = multi[multi["tgroup"] != tg], multi[multi["tgroup"] == tg]
                    test_cells = set(test["battery_id"].map(physical_cell))
                    train = train[~train["battery_id"].map(physical_cell).isin(test_cells)]
                    p = _fit_predict(model, feats, train, test, aug)
                    add("2. temperatura no vista", f"{tg} °C", model, fset, aug, test["soh"], p, len(test_cells))
                # 3. celdas nuevas, todas las temperaturas en entrenamiento (5 pliegues por celda)
                groups = multi["battery_id"].map(physical_cell)
                oof = pd.Series(np.nan, index=multi.index)
                for tr, te in GroupKFold(n_splits=5).split(multi, groups=groups):
                    oof.iloc[te] = _fit_predict(model, feats, multi.iloc[tr], multi.iloc[te], aug)
                for tg in (4, 24, 43):
                    m = multi["tgroup"] == tg
                    add("3. celda nueva (multitemperatura)", f"{tg} °C", model, fset, aug, multi.loc[m, "soh"],
                        oof[m], groups[m].nunique())
                print(f"{model:18s} {fset:32s} aug={aug} listo", flush=True)

    res = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "generalizacion_resultados.csv", index=False, float_format="%.5g")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("docs/generalizacion"))
    a = ap.parse_args()
    res = run(a.out_dir)
    piv = res.pivot_table(index=["esquema", "test", "modelo"], columns=["caracteristicas", "augmentation"],
                          values="mae_soh")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print((piv * 100).round(2))


if __name__ == "__main__":
    main()
