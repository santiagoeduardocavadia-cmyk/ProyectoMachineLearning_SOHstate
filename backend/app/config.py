"""Configuración central del backend.

Todas las rutas se pueden sobrescribir con variables de entorno para que el
proyecto sea reproducible en cualquier máquina (Windows, Linux, CI).
"""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("BATT_DATA_DIR", BACKEND_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_FILE = Path(os.getenv("BATT_PROCESSED_FILE", PROCESSED_DIR / "nasa_battery_cycles.csv"))
UPLOADS_DIR = Path(os.getenv("BATT_UPLOADS_DIR", DATA_DIR / "uploads"))
EXPERIMENTS_DIR = Path(os.getenv("BATT_EXPERIMENTS_DIR", BACKEND_DIR / "experiments"))

# Fuente oficial: NASA Prognostics Center of Excellence (PCoE), "Battery Data Set".
NASA_DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
NASA_INNER_ZIP = "1. BatteryAgingARC-FY08Q4.zip"
DEFAULT_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

# Parámetros físicos del experimento NASA (README del dataset).
NOMINAL_CAPACITY_AH = 2.0      # capacidad nominal de las celdas 18650
EOL_CAPACITY_AH = 1.4          # fin de vida: 30 % de pérdida de capacidad
DISCHARGE_CURRENT_A = 2.0      # descarga en corriente constante

# Versión del contrato HTTP. Cambios incompatibles => nuevo prefijo (/api/v2).
API_VERSION = "1.1.1"
API_PREFIX = "/api/v1"

# Orígenes permitidos para CORS (separados por coma). "*" = cualquiera (desarrollo).
CORS_ORIGINS = [o.strip() for o in os.getenv("BATT_CORS_ORIGINS", "*").split(",") if o.strip()]
