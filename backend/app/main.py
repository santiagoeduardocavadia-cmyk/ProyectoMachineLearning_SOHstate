"""Punto de entrada de la API.

Ejecutar desde la carpeta backend/:
    uvicorn app.main:app --reload --port 8000
Documentación interactiva: http://localhost:8000/docs
Página mínima de prueba:   http://localhost:8000/demo
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from app import __version__, config
from app.api.routes import router
from app.errors import register_error_handlers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="API de predicción de vida de baterías (SOH / RUL)",
    description="Backend del proyecto final de Fundamentos de IA + Machine Learning. "
                "Datos: NASA PCoE Battery Data Set. Contrato: CONTEXTO_CHATGPT.md.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)
app.include_router(router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/demo", include_in_schema=False)
def demo():
    """Página técnica mínima para validar un experimento sin el frontend (no es la UI final)."""
    return FileResponse(STATIC_DIR / "demo.html")
