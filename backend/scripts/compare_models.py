"""¿Qué modelo de machine learning es el más adecuado? Selección por validación cruzada (v1.5.0).

Uso (desde backend/):
    python -m scripts.compare_models [--out-dir docs/seleccion_modelo] [--quick]

Protocolo (el mismo para los 5 modelos, para que la comparación sea justa):

1. Se apartan las **celdas de test** del experimento de referencia (B0006, B0031 y B0047 en el dataset
   multitemperatura; B0006 en el de NASA). No se usan para elegir nada.
2. Con las celdas restantes se hace **validación cruzada agrupada por celda física** (GroupKFold, hasta 10
   pliegues): cada pliegue deja fuera celdas completas, así se mide el error en una celda que el modelo no vio.
3. Para cada modelo se prueban unos pocos hiperparámetros y se queda el de menor MAE de CV.
4. El **modelo elegido** es el de menor MAE medio de CV: el que menos se equivoca en celdas que no vio.
   Además se marca qué modelos están **empatados** con él: diferencia no significativa (prueba de Wilcoxon
   pareada por pliegue, p ≥ 0.05) y a menos de un error estándar. Si hay empate, la elección no es clara y
   cualquiera de los empatados sirve.
5. Solo al final se entrena cada modelo con todas las celdas de entrenamiento y se mide en el test, como
   confirmación. El test nunca decide.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data import dataset as ds  # noqa: E402
from app.data.dataset import get_dataset  # noqa: E402
from app.data.features import FEATURE_NAMES, LEGACY_FEATURES  # noqa: E402
from app.ml.domain import is_virtual, physical_cell  # noqa: E402
from app.ml.models import MODELS, build_pipeline, resolve_hyperparameters  # noqa: E402

SEED = 42
MAX_FOLDS = 10
GRIDS: dict[str, dict[str, list[Any]]] = {
    "ridge": {"alpha": [0.1, 1.0, 10.0]},
    "knn": {"n_neighbors": [5, 10, 20]},
    "svr": {"C": [1.0, 10.0, 100.0]},
    "random_forest": {"min_samples_leaf": [2, 5], "max_depth": [None, 10]},
    "gradient_boosting": {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
}
SVR_EPSILON = {"soh": 0.01, "rul": 1.0}  # el margen del SVR va en las unidades del objetivo
SCENARIOS = [
    {"dataset": ds.MULTITEMP_DATASET_ID, "target": "soh", "features": FEATURE_NAMES,
     "test": ["B0006", "B0031", "B0047"], "label": "SOH, multitemperatura (29 celdas, 4/24/43 °C)"},
    {"dataset": ds.DEFAULT_DATASET_ID, "target": "soh", "features": LEGACY_FEATURES,
     "test": ["B0006"], "label": "SOH, NASA original (4 celdas a 24 °C)"},
    {"dataset": ds.DEFAULT_DATASET_ID, "target": "rul", "features": LEGACY_FEATURES,
     "test": ["B0006"], "label": "RUL, NASA original (B0007 no llega al fin de vida)"},
]


def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def _configs(model: str, target: str, quick: bool) -> list[dict[str, Any]]:
    grid = GRIDS[model]
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    if quick:
        combos = combos[:1]
    if model == "svr":
        combos = [{**c, "epsilon": SVR_EPSILON[target]} for c in combos]
    if quick and model in ("random_forest", "gradient_boosting"):
        combos = [{**c, "n_estimators": 50} for c in combos]
    return [resolve_hyperparameters(model, c) for c in combos]


def _fold_maes(model: str, hp: dict, X: pd.DataFrame, y: pd.Series, folds) -> list[float]:
    out = []
    for tr, te in folds:
        pipe = build_pipeline(model, hp, SEED)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        out.append(_mae(y.iloc[te], pipe.predict(X.iloc[te])))
    return out


def run_scenario(sc: dict[str, Any], quick: bool = False) -> dict[str, Any]:
    df = get_dataset(sc["dataset"])
    target = sc["target"]
    df = df[df[target].notna() & ~df["battery_id"].map(is_virtual)].copy()
    cells = df["battery_id"].map(physical_cell)
    test_cells = {physical_cell(b) for b in sc["test"]}
    train, test = df[~cells.isin(test_cells)], df[cells.isin(test_cells)]
    feats = [f for f in sc["features"] if f in df and train[f].notna().any()]
    X, y = train[feats].astype(float), train[target].astype(float)
    groups = train["battery_id"].map(physical_cell)
    n_splits = min(MAX_FOLDS, groups.nunique())
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))

    rows = []
    baseline = [_mae(y.iloc[te], np.full(len(te), y.iloc[tr].mean())) for tr, te in folds]
    for model in MODELS:
        best = None
        for hp in _configs(model, target, quick):
            maes = _fold_maes(model, hp, X, y, folds)
            if best is None or np.mean(maes) < np.mean(best[1]):
                best = (hp, maes)
        hp, maes = best
        pipe = build_pipeline(model, hp, SEED).fit(X, y)
        test_mae = _mae(test[target], pipe.predict(test[feats].astype(float))) if len(test) else None
        rows.append({"model": model, "label": MODELS[model].label, "hyperparameters": hp, "fold_mae": maes,
                     "cv_mae": float(np.mean(maes)), "cv_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
                     "test_mae": test_mae})
        print(f"  {sc['label'][:30]:30s} {model:18s} CV {np.mean(maes):.4f}", flush=True)

    rows.sort(key=lambda r: r["cv_mae"])
    best = rows[0]
    se = best["cv_std"] / np.sqrt(n_splits)
    for r in rows:
        r["rank"] = rows.index(r) + 1
        if r is best:
            r["p_value"], r["tied"] = None, True
            continue
        diff = np.asarray(r["fold_mae"]) - np.asarray(best["fold_mae"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
        r["p_value"] = p
        r["tied"] = p >= 0.05 and r["cv_mae"] - best["cv_mae"] <= se
    chosen = best
    for r in rows:
        r["chosen"] = r is chosen
    min_p = 1.0 / 2 ** (n_splits - 1)  # menor p posible de Wilcoxon con n pliegues
    return {**{k: sc[k] for k in ("dataset", "target", "label", "test")}, "n_folds": n_splits,
            "n_train_cells": int(groups.nunique()), "baseline_cv_mae": float(np.mean(baseline)),
            "wilcoxon_min_p": min_p, "rows": rows, "chosen": chosen["model"], "best_cv": best["model"]}


def _fmt(v: float | None, target: str) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.2f}" if target == "soh" else f"{v:.1f}"


def to_markdown(results: list[dict[str, Any]]) -> str:
    out = []
    for res in results:
        unit = "puntos de SOH" if res["target"] == "soh" else "ciclos"
        out.append(f"### {res['label']}\n")
        out.append(f"Validación cruzada: {res['n_folds']} pliegues por celda física con {res['n_train_cells']} celdas "
                   f"de entrenamiento. Test: {', '.join(res['test'])}. Errores (MAE) en {unit}; menor es mejor. "
                   f"Línea base (predecir siempre la media): {_fmt(res['baseline_cv_mae'], res['target'])}.\n")
        out.append("| Puesto | Modelo | MAE CV (media ± desv.) | p (Wilcoxon vs. el mejor) | ¿Empatado con el mejor? | "
                   "MAE test (solo confirmación) | Hiperparámetros elegidos |")
        out.append("|---|---|---|---|---|---|---|")
        for r in res["rows"]:
            name = f"**{r['label']}** ✅" if r["chosen"] else r["label"]
            p = "—" if r["p_value"] is None else f"{r['p_value']:.3f}"
            hp = ", ".join(f"{k}={v}" for k, v in r["hyperparameters"].items())
            out.append(f"| {r['rank']} | {name} | {_fmt(r['cv_mae'], res['target'])} ± {_fmt(r['cv_std'], res['target'])} "
                       f"| {p} | {'sí' if r['tied'] else 'no'} | {_fmt(r['test_mae'], res['target'])} | {hp} |")
        note = ""
        if res["wilcoxon_min_p"] >= 0.05:
            note = (f" Con solo {res['n_folds']} pliegues la prueba de Wilcoxon no puede bajar de "
                    f"p = {res['wilcoxon_min_p']:.3f}: no puede declarar diferencias significativas y el empate se decide solo "
                    "con la regla de un error estándar. El resultado es poco concluyente.")
        ties = [r["label"] for r in res["rows"] if r["tied"] and not r["chosen"]]
        tie = f" Empatado con: {', '.join(ties)}." if ties else " Ningún otro modelo empata con él."
        out.append(f"\nElegido: **{MODELS[res['chosen']].label}**.{tie}{note}\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("docs/seleccion_modelo"))
    ap.add_argument("--quick", action="store_true", help="una sola configuración por modelo (para pruebas)")
    a = ap.parse_args()
    results = [run_scenario(sc, a.quick) for sc in SCENARIOS]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    flat = [{"escenario": r["label"], **{k: v for k, v in row.items() if k != "fold_mae"},
             "fold_mae": ";".join(f"{m:.5g}" for m in row["fold_mae"])} for r in results for row in r["rows"]]
    pd.DataFrame(flat).to_csv(a.out_dir / "seleccion_modelo_resultados.csv", index=False, float_format="%.5g")
    md = to_markdown(results)
    (a.out_dir / "tablas.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
