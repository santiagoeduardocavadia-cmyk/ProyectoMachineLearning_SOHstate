"""Esquemas Pydantic de entrada (request) de la API.

Las respuestas se documentan en CONTEXTO_CHATGPT.md y en /docs.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Target = Literal["soh", "rul", "capacity"]


class ExperimentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str = Field(..., min_length=1, max_length=80, description="Nombre legible del experimento.")
    description: Optional[str] = Field(None, max_length=500)
    target: Target = Field("soh", description="Variable a predecir: soh, rul o capacity.")
    model_type: str = Field("random_forest", description="Clave del modelo (ver GET /api/v1/models).")
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    features: Optional[list[str]] = Field(None, description="Subconjunto de características; null = todas.")
    test_batteries: list[str] = Field(default_factory=lambda: ["B0006"], min_length=1,
                                      description="Baterías reservadas para test (split por batería).")
    test_temperatures: Optional[list[float]] = Field(
        None, min_length=1, max_length=10,
        description="(v1.2) Si se envía, el test son TODAS las baterías a estas temperaturas (±3 °C) y se ignora "
                    "test_batteries. Mide cómo generaliza el modelo a una temperatura que no vio.")
    random_state: int = Field(42, ge=0, le=2**31 - 1)
    dataset_id: str = Field("nasa", min_length=1, max_length=100,
                            description="Dataset de entrenamiento (ver GET /api/v1/datasets). Por defecto el NASA.")


class PredictRequest(BaseModel):
    """Exactamente una de las dos formas:

    - ``samples``: lista de diccionarios {característica: valor}. Un valor null se imputa con la mediana
      del entrenamiento.
    - ``battery_id`` (+ ``cycles`` opcional): usa las características del dataset para esa batería.
    """

    model_config = ConfigDict(extra="forbid")

    samples: Optional[list[dict[str, Optional[float]]]] = Field(None, min_length=1, max_length=5000)
    battery_id: Optional[str] = None
    cycles: Optional[list[int]] = Field(None, min_length=1)

    @model_validator(mode="after")
    def _one_mode(self):
        if (self.samples is None) == (self.battery_id is None):
            raise ValueError("Envía 'samples' o 'battery_id', pero no ambos ni ninguno.")
        if self.cycles is not None and self.battery_id is None:
            raise ValueError("'cycles' solo se usa junto con 'battery_id'.")
        return self
