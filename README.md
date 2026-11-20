# Predicción del estado de salud (SOH) y vida útil restante (RUL) de baterías de litio

Proyecto final de **Fundamentos de IA + Machine Learning** (Ingeniería Electrónica, Universidad del Magdalena).

Entorno web para cargar el **NASA Battery Dataset**, configurar y ejecutar entrenamientos, ver métricas y gráficos,
hacer inferencia y guardar cada experimento (configuración, métricas, predicciones y modelo) de forma reproducible.

| Parte | Tecnología | Responsable | Carpeta |
|---|---|---|---|
| Backend / ML | Python, FastAPI, scikit-learn | Claude | `backend/` |
| Frontend / visualización | a elección | ChatGPT | `frontend/` |

Contrato entre ambos y reglas de colaboración: **[CONTEXTO_CHATGPT.md](CONTEXTO_CHATGPT.md)**.
Metodología y preprocesamiento: **[backend/docs/PREPROCESAMIENTO.md](backend/docs/PREPROCESAMIENTO.md)**.

## Levantar el backend

Requisitos: Python 3.10 o superior.

```bash
cd backend
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (cmd):         .venv\Scripts\activate.bat
# Linux / macOS:         source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- API interactiva (Swagger): <http://localhost:8000/docs>
- Página técnica mínima para validar un experimento: <http://localhost:8000/demo>

El CSV procesado ya está en el repositorio, así que **no hace falta descargar nada** para arrancar.
Para regenerarlo desde los datos crudos de la NASA (≈210 MB): `python -m scripts.prepare_dataset`.

## Validar un experimento de punta a punta

**Opción A: navegador.** Abre <http://localhost:8000/demo>, deja los valores por defecto (SOH, Random Forest, test B0006)
y pulsa *Entrenar*. Verás MAE/RMSE/R² frente a la línea base, la curva predicho vs real de B0006 y, en *Predecir*,
el SOH estimado de un ciclo.

**Opción B: Swagger** (<http://localhost:8000/docs>):
1. `POST /api/v1/experiments` con `{"name": "mi primer experimento"}` → copia el `id` de la respuesta.
2. `GET /api/v1/experiments/{id}` → métricas.
3. `GET /api/v1/experiments/{id}/predictions?split=test` → datos para el gráfico predicho vs real.
4. `POST /api/v1/experiments/{id}/predict` con `{"battery_id": "B0006", "cycles": [1, 100]}` → inferencia.

**Opción C: terminal**
```bash
curl -X POST http://localhost:8000/api/v1/experiments -H "Content-Type: application/json" \
     -d "{\"name\": \"RF SOH\", \"target\": \"soh\", \"model_type\": \"random_forest\", \"test_batteries\": [\"B0006\"]}"
```

Cada experimento queda guardado en `backend/experiments/<id>/`:
`experiment.json` (configuración, métricas, importancias, versiones de librerías, huella del dataset),
`predictions.csv` y `model.joblib`.

Ya vienen tres experimentos de referencia (`referencia_soh_random_forest`, `referencia_soh_ridge`,
`referencia_rul_gradient_boosting`). Si tu versión de scikit-learn es distinta y el modelo guardado da
advertencias al cargarse, regenéralos con `python -m scripts.run_reference_experiment`.

## Resultados de referencia (test = batería B0006, nunca vista en entrenamiento)

| Experimento | MAE | RMSE | R² | MAE línea base |
|---|---|---|---|---|
| SOH – Random Forest | 0.030 | 0.037 | 0.914 | 0.113 |
| SOH – Ridge | 0.020 | 0.020 | 0.975 | 0.113 |
| RUL – Gradient Boosting | 11.4 ciclos | 13.0 | 0.830 | 27.3 |

## Tests

```bash
cd backend
python -m pytest -q
```
También corren en GitHub Actions en cada push a `backend/` (`.github/workflows/backend-ci.yml`).

## Estructura

```
backend/
  app/
    main.py              FastAPI, CORS, /demo
    config.py            rutas y constantes (variables de entorno BATT_*)
    errors.py            formato único de errores
    schemas.py           validación de entradas (Pydantic)
    api/routes.py        endpoints /api/v1
    data/features.py     14 características por ciclo
    data/preprocessing.py descarga, lectura .mat, objetivos SOH/RUL, limpieza
    data/dataset.py      carga del CSV procesado
    ml/models.py         catálogo de modelos y validación de hiperparámetros
    ml/training.py       split por batería, CV, métricas, línea base, importancias
    ml/store.py          persistencia de experimentos
    ml/inference.py      predicción con un experimento guardado
  data/processed/        CSV procesado (versionado)
  experiments/           experimentos guardados
  scripts/               prepare_dataset.py, run_reference_experiment.py
  tests/                 pytest
  docs/PREPROCESAMIENTO.md
frontend/                (ChatGPT)
```

## Frontend

_Sección reservada para el frontend (la edita ChatGPT)._
