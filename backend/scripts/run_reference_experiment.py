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
from app.data.dataset import dataset_fingerprint, get_dataset  # noqa: E402
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


def main() -> None:
    df = get_dataset()
    sha = dataset_fingerprint()
    for exp_id, cfg in REFERENCE.items():
        target_dir = config.EXPERIMENTS_DIR / exp_id
        if target_dir.exists():
            shutil.rmtree(target_dir)
        rec = store.save(train_experiment(df, dict(cfg)), dataset_sha=sha, experiment_id=exp_id)
        print(exp_id, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in rec["metrics"]["test"].items()}))


if __name__ == "__main__":
    main()
