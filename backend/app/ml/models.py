"""Catálogo de modelos de regresión disponibles y validación de hiperparámetros.

Cada modelo se envuelve en un ``Pipeline``:
    SimpleImputer(mediana) -> StandardScaler -> regresor
- El imputador cubre los pocos ciclos sin carga previa o sin impedancia medida.
- El escalado es necesario para Ridge, SVR y KNN; en árboles no afecta.
Imputador y escalador se ajustan SOLO con los datos de entrenamiento.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from app.errors import AppError


@dataclass(frozen=True)
class Param:
    name: str
    type: str  # "int" | "float"
    default: Any
    min: float | None = None
    max: float | None = None
    nullable: bool = False
    description: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("name", "type", "default", "min", "max", "nullable", "description")}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    description: str
    builder: Callable[..., Any]
    params: list[Param] = field(default_factory=list)
    uses_random_state: bool = False

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "description": self.description,
                "hyperparameters": [p.to_dict() for p in self.params]}


MODELS: dict[str, ModelSpec] = {
    "random_forest": ModelSpec(
        "random_forest", "Random Forest",
        "Ensamble de árboles (bagging). Robusto, captura no linealidades y da importancia de variables.",
        RandomForestRegressor,
        [Param("n_estimators", "int", 300, 10, 1000, description="Número de árboles."),
         Param("max_depth", "int", None, 1, 50, nullable=True, description="Profundidad máxima (null = sin límite)."),
         Param("min_samples_leaf", "int", 2, 1, 50, description="Muestras mínimas por hoja.")],
        uses_random_state=True,
    ),
    "gradient_boosting": ModelSpec(
        "gradient_boosting", "Gradient Boosting",
        "Árboles secuenciales que corrigen el error del anterior (boosting).",
        GradientBoostingRegressor,
        [Param("n_estimators", "int", 300, 10, 2000, description="Número de etapas."),
         Param("learning_rate", "float", 0.05, 0.001, 1.0, description="Tasa de aprendizaje."),
         Param("max_depth", "int", 3, 1, 10, description="Profundidad de cada árbol."),
         Param("subsample", "float", 0.8, 0.1, 1.0, description="Fracción de muestras por etapa.")],
        uses_random_state=True,
    ),
    "ridge": ModelSpec(
        "ridge", "Regresión lineal (Ridge)",
        "Modelo lineal con regularización L2. Línea base interpretable.",
        Ridge,
        [Param("alpha", "float", 1.0, 0.0, 1000.0, description="Fuerza de la regularización L2.")],
    ),
    "svr": ModelSpec(
        "svr", "Support Vector Regression (RBF)",
        "Regresión con vectores de soporte y kernel RBF.",
        SVR,
        [Param("C", "float", 10.0, 0.001, 1000.0, description="Penalización del error."),
         Param("epsilon", "float", 0.01, 0.0, 10.0, description="Margen sin penalización.")],
    ),
    "knn": ModelSpec(
        "knn", "K vecinos más cercanos",
        "Promedia el objetivo de los k ciclos más parecidos del entrenamiento.",
        KNeighborsRegressor,
        [Param("n_neighbors", "int", 5, 1, 50, description="Número de vecinos (k)."),
         ],
    ),
}


def resolve_hyperparameters(model_type: str, given: dict[str, Any] | None) -> dict[str, Any]:
    """Completa con valores por defecto y valida tipo y rango de cada hiperparámetro."""
    if model_type not in MODELS:
        raise AppError("unknown_model", f"Modelo '{model_type}' no soportado.", 422,
                       {"available": list(MODELS)})
    spec = MODELS[model_type]
    given = dict(given or {})
    allowed = {p.name for p in spec.params}
    unknown = sorted(set(given) - allowed)
    if unknown:
        raise AppError("invalid_hyperparameters", f"Hiperparámetros desconocidos para {model_type}: {unknown}",
                       422, {"allowed": sorted(allowed)})
    resolved: dict[str, Any] = {}
    errors = []
    for p in spec.params:
        value = given.get(p.name, p.default)
        if value is None:
            if not p.nullable:
                errors.append({"field": p.name, "message": "no puede ser null"})
            resolved[p.name] = None
            continue
        if isinstance(value, bool):
            errors.append({"field": p.name, "message": "debe ser numérico"})
            continue
        try:
            if p.type == "int":
                if isinstance(value, float) and not value.is_integer():
                    raise ValueError
                value = int(value)
            else:
                value = float(value)
        except (TypeError, ValueError):
            errors.append({"field": p.name, "message": f"debe ser de tipo {p.type}"})
            continue
        if (p.min is not None and value < p.min) or (p.max is not None and value > p.max):
            errors.append({"field": p.name, "message": f"debe estar en [{p.min}, {p.max}]"})
            continue
        resolved[p.name] = value
    if errors:
        raise AppError("invalid_hyperparameters", "Hiperparámetros fuera de rango o con tipo inválido.", 422, errors)
    return resolved


def build_pipeline(model_type: str, hyperparameters: dict[str, Any], random_state: int) -> Pipeline:
    spec = MODELS[model_type]
    kwargs = dict(hyperparameters)
    if spec.uses_random_state:
        kwargs["random_state"] = random_state
    if model_type == "random_forest":
        kwargs["n_jobs"] = -1
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", spec.builder(**kwargs)),
    ])
