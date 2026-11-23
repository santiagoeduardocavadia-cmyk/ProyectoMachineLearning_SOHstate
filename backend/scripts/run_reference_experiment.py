"""Genera el experimento de referencia versionado en el repositorio.

Uso (desde backend/):  python -m scripts.run_reference_experiment

Crea backend/experiments/referencia_soh_random_forest/ con la configuración por
defecto (SOH, Random Forest, test = B0006, random_state = 42). Sirve como
evidencia reproducible: al volver a ejecutarlo con el mismo dataset (misma
huella sha256_16) se obtienen las mismas métricas.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.data import dataset as ds  # noqa: E402
from app.data.dataset import dataset_fingerprint, get_dataset  # noqa: E402
from app.data.features import LEGACY_FEATURES  # noqa: E402
from app.ml import store  # noqa: E402
from app.ml.training import train_experiment  # noqa: E402

REFERENCE = {
    "referencia_soh_random_forest": {
        "name": "Referencia SOH - Random Forest (test B0006)",
        "description": "Experimento de referencia: SOH con Random Forest, entrenado en B0005/B0007/B0018 y evaluado en B0006.",
        "target": "soh", "model_type": "random_forest", "hyperparameters": {},
        "features": None, "test_batteries": ["B0006"], "random_state": 42,
    },
    "referencia_soh_ridge": {
        "name": "Referencia SOH - Ridge (test B0006)",
        "description": "Comparación lineal: extrapola mejor que los árboles por debajo del SOH mínimo visto en entrenamiento.",
        "target": "soh", "model_type": "ridge", "hyperparameters": {},
        "features": None, "test_batteries": ["B0006"], "random_state": 42,
    },
    "referencia_rul_gradient_boosting": {
        "name": "Referencia RUL - Gradient Boosting (test B0006)",
        "description": "RUL en ciclos; B0007 se excluye por no alcanzar el fin de vida.",
        "target": "rul", "model_type": "gradient_boosting", "hyperparameters": {},
        "features": None, "test_batteries": ["B0006"], "random_state": 42,
    },
}
# Los tres de NASA (24 °C) conservan las 14 características originales: así sus métricas no cambian
# respecto de v1.1 y sirven de comparación "antes" frente a los multitemperatura.
for _cfg in REFERENCE.values():
    _cfg["features"] = list(LEGACY_FEATURES)

MULTI_TEST = ["B0006", "B0031", "B0047"]  # una celda de test por temperatura: 24, 43 y 4 °C
# Los 5 modelos, con los hiperparámetros que eligió la validación cruzada por celda física
# (python -m scripts.compare_models; docs/SELECCION_MODELO.md). /predict/auto usa el de menor MAE de CV.
MULTI_MODELS = {
    "random_forest": ("Random Forest", {"min_samples_leaf": 5}),
    "svr": ("SVR", {"C": 10.0, "epsilon": 0.01}),
    "gradient_boosting": ("Gradient Boosting", {"learning_rate": 0.1, "max_depth": 2}),
    "ridge": ("Ridge", {"alpha": 0.1}),
    "knn": ("K vecinos", {"n_neighbors": 10}),
}
for _model, (_label, _hp) in MULTI_MODELS.items():
    REFERENCE[f"referencia_multitemp_soh_{_model}"] = {
        "name": f"Multitemperatura SOH - {_label} (test B0006 24 °C, B0031 43 °C, B0047 4 °C)",
        "description": "Entrenado con 29 celdas NASA a 4, 24 y 43 °C y corrientes de 1, 2 y 4 A, con temperatura "
                       "y corriente como características. Hiperparámetros elegidos por validación cruzada. "
                       "Evaluado en una celda nueva de cada temperatura.",
        "target": "soh", "model_type": _model, "hyperparameters": dict(_hp), "features": None,
        "test_batteries": MULTI_TEST, "random_state": 42, "dataset_id": ds.MULTITEMP_DATASET_ID,
    }
REFERENCE["referencia_multitemp_soh_sin_43C"] = {
    "name": "Multitemperatura SOH - Gradient Boosting, 43 °C nunca vista (test = todas las celdas a 43 °C)",
    "description": "Entrenado solo a 4 y 24 °C. Mide qué pasa con una temperatura que el modelo no vio.",
    "target": "soh", "model_type": "gradient_boosting", "hyperparameters": {}, "features": None,
    "test_batteries": ["B0006"], "test_temperatures": [43.0], "random_state": 42,
    "dataset_id": ds.MULTITEMP_DATASET_ID,
}


def main() -> None:
    for exp_id, cfg in REFERENCE.items():
        dataset_id = cfg.get("dataset_id", ds.DEFAULT_DATASET_ID)
        df = get_dataset(dataset_id)
        meta = ds.dataset_meta(dataset_id)
        info = {"name": meta["name"], "file": ds.dataset_path(dataset_id).name,
                "nominal_capacity_ah": meta["nominal_capacity_ah"], "eol_capacity_ah": meta["eol_capacity_ah"]}
        target_dir = config.EXPERIMENTS_DIR / exp_id
        if target_dir.exists():
            shutil.rmtree(target_dir)
        rec = store.save(train_experiment(df, dict(cfg)), dataset_sha=dataset_fingerprint(dataset_id),
                         experiment_id=exp_id, dataset_info=info)
        print(exp_id, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in rec["metrics"]["test"].items()}))


if __name__ == "__main__":
    main()
