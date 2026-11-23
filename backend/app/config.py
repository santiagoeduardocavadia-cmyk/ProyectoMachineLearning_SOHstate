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
# Dataset NASA completo (34 celdas a 4, 22-24 y 43-44 °C), v1.2.0.
MULTITEMP_FILE = Path(os.getenv("BATT_MULTITEMP_FILE", PROCESSED_DIR / "nasa_multitemp_cycles.csv"))
UPLOADS_DIR = Path(os.getenv("BATT_UPLOADS_DIR", DATA_DIR / "uploads"))
EXPERIMENTS_DIR = Path(os.getenv("BATT_EXPERIMENTS_DIR", BACKEND_DIR / "experiments"))

# Fuente oficial: NASA Prognostics Center of Excellence (PCoE), "Battery Data Set".
NASA_DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
NASA_INNER_ZIP = "1. BatteryAgingARC-FY08Q4.zip"
DEFAULT_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
# Todas las celdas del ZIP oficial (6 sub-ZIP). B0035 y B0037 no existen en la fuente.
ALL_BATTERIES = [*DEFAULT_BATTERIES, "B0025", "B0026", "B0027", "B0028", "B0029", "B0030", "B0031", "B0032",
                 "B0033", "B0034", "B0036", "B0038", "B0039", "B0040", "B0041", "B0042", "B0043", "B0044",
                 *[f"B00{n}" for n in range(45, 57)]]

# Parámetros físicos del experimento NASA (README del dataset).
NOMINAL_CAPACITY_AH = 2.0      # capacidad nominal de las celdas 18650
EOL_CAPACITY_AH = 1.4          # fin de vida: 30 % de pérdida de capacidad
DISCHARGE_CURRENT_A = 2.0      # descarga en corriente constante

# Versión del contrato HTTP. Cambios incompatibles => nuevo prefijo (/api/v2).
API_VERSION = "1.4.0"
API_PREFIX = "/api/v1"

# Orígenes permitidos para CORS (separados por coma). "*" = cualquiera (desarrollo).
CORS_ORIGINS = [o.strip() for o in os.getenv("BATT_CORS_ORIGINS", "*").split(",") if o.strip()]
