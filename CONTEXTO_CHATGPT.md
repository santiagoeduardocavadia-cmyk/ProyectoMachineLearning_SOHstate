# Contexto para ChatGPT: frontend del proyecto "Predicción de vida de baterías"

> **Para ChatGPT:** este documento es tu contexto completo. Tú construyes **solo el frontend** (carpeta `frontend/`).
> El backend ya existe, lo mantiene Claude y se consume **únicamente** por HTTP con el contrato de abajo.
> Versión del contrato: **API v1.1.1** (prefijo `/api/v1`). Novedades de v1.1.0 en la sección 4.3. Repositorio:
> https://github.com/santiagoeduardocavadia-cmyk/ProyectoMachineLearning_SOHstate

---

## 1. El proyecto en una página

**Curso:** Fundamentos de IA + Machine Learning (Ingeniería Electrónica, Universidad del Magdalena).
**Problema:** estimar el **estado de salud (SOH)** y la **vida útil restante (RUL)** de baterías de ion-litio a partir de
mediciones de sus ciclos de carga/descarga.

**Datos:** NASA Prognostics Center of Excellence, *Battery Data Set*. Cuatro celdas 18650 (B0005, B0006, B0007, B0018)
cicladas a 24 °C hasta perder ~30 % de capacidad (de 2.0 Ah nominal a 1.4 Ah = fin de vida, EOL).
El backend convierte las señales crudas en **636 filas = 1 fila por ciclo de descarga** con 14 características.

| Concepto | Significado | Unidad |
|---|---|---|
| `soh` | capacidad / 2.0 Ah. 1.0 = nueva, 0.70 = fin de vida | fracción 0–1 (mostrar como %) |
| `rul` | ciclos que faltan hasta que la capacidad cae bajo 1.4 Ah | ciclos |
| `capacity_ah` | capacidad medida en el ciclo | Ah |
| `health_status` | `bueno` (SOH ≥ 0.85), `degradado` (0.70–0.85), `fin_de_vida` (< 0.70) | texto |

**Decisiones técnicas que la UI debe hacer visibles** (la rúbrica valora la justificación más que lo estético):
1. **Split por batería**: se entrena con unas baterías y se evalúa con otra nunca vista. Un split aleatorio por ciclo haría
   trampa porque ciclos consecutivos son casi iguales.
2. **Línea base**: cada experimento se compara con un modelo tonto que predice la media. Mostrar ambas métricas lado a lado.
3. **Validación cruzada leave-one-battery-out** dentro del entrenamiento (`metrics.cv`).
4. **Sin fuga de información**: no se usa la duración total de la descarga (es la capacidad disfrazada).
5. **B0007 no llega al EOL**, así que su RUL es desconocido y se excluye de los experimentos de RUL (`warnings`).

## 2. Arquitectura

```
ProyectoMachineLearning_SOHstate/
├── backend/            ← SOLO Claude. Python 3.10+ / FastAPI / scikit-learn
│   ├── app/            API, preprocesamiento, modelos, persistencia
│   ├── data/processed/ CSV procesado (versionado)
│   ├── experiments/    experimentos guardados (JSON + CSV + modelo .joblib)
│   ├── scripts/        descarga/preprocesamiento y experimentos de referencia
│   └── tests/          pytest
├── frontend/           ← SOLO ChatGPT. Tecnología a tu elección (recomendado: Vite + React o HTML/JS + Chart.js)
├── CONTEXTO_CHATGPT.md ← este archivo (lo mantiene Claude)
└── README.md
```

Flujo: `frontend (navegador)  --HTTP JSON-->  backend FastAPI :8000  -->  scikit-learn + archivos en backend/experiments/`

El backend ya tiene **CORS abierto** (`*`), así que el frontend puede correr en cualquier puerto (p. ej. `http://localhost:5173`).
Configura la URL base en **un solo lugar** (variable `VITE_API_URL` o constante `API_BASE_URL`), por defecto `http://localhost:8000/api/v1`.

## 3. Cómo levantar el backend (para probar tu frontend)

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- Documentación interactiva (probar cada endpoint): http://localhost:8000/docs
- Página técnica mínima del backend (NO es la UI final, no la copies): http://localhost:8000/demo
- Ya vienen 3 experimentos guardados: `referencia_soh_random_forest`, `referencia_soh_ridge`, `referencia_rul_gradient_boosting`.

## 4. Contrato de la API v1

Convenciones:
- Todo es JSON (`Content-Type: application/json`), salvo la subida de CSV (`multipart/form-data`) y la descarga del modelo.
- Los valores numéricos que no existen llegan como `null` (nunca `NaN`).
- Fechas en ISO 8601 UTC (`2026-11-20T18:12:28+00:00`).
- **Todos los errores** tienen la misma forma (muestra `message` al usuario y decide por `code`):

```json
{ "error": { "code": "invalid_hyperparameters", "message": "Hiperparámetros fuera de rango o con tipo inválido.",
             "details": [ { "field": "n_estimators", "message": "debe estar en [10, 1000]" } ] } }
```

| HTTP | `code` posibles |
|---|---|
| 400 | `invalid_experiment_id`, `invalid_dataset_id` |
| 403 | `dataset_protected` (v1.1) |
| 404 | `not_found`, `battery_not_found`, `experiment_not_found`, `cycle_not_found`, `dataset_not_found` |
| 413 / 415 | `file_too_large`, `invalid_file` |
| 422 | `validation_error`, `unknown_model`, `invalid_hyperparameters`, `invalid_features`, `invalid_split`, `invalid_samples`, `invalid_file`, `invalid_dataset` (v1.1) |
| 500 / 503 | `internal_error`, `dataset_not_found`, `dataset_invalid` |

### 4.1 Resumen de endpoints

| Método | Ruta | Para qué lo usa la UI |
|---|---|---|
| GET | `/health` | Indicador "backend conectado" |
| GET | `/dataset/summary` | Tarjetas del dataset y tabla de baterías |
| GET | `/dataset/features` | Tabla/explicación de características y objetivos |
| GET | `/dataset/batteries/{battery_id}/degradation` | **Gráfico** curva de degradación (capacidad/SOH vs ciclo) |
| GET | `/dataset/cycles?battery_id=&limit=&offset=` | Tabla de datos procesados (paginada) |
| GET | `/models` | Construir el **formulario de entrenamiento** (modelos, rangos, baterías, valores por defecto) |
| POST | `/experiments` | **Entrenar** un experimento (síncrono, tarda 0.1–3 s) |
| GET | `/experiments?target=` | Historial de experimentos / tabla comparativa |
| GET | `/experiments/{id}` | Detalle: configuración + métricas |
| DELETE | `/experiments/{id}` | Borrar experimento (204 sin cuerpo) |
| GET | `/experiments/{id}/predictions?split=&battery_id=` | **Gráficos** predicho vs real y residuales |
| GET | `/experiments/{id}/feature-importance` | **Gráfico** de barras de importancia |
| GET | `/experiments/{id}/model` | Botón "descargar modelo" (.joblib) |
| POST | `/experiments/{id}/predict` | **Inferencia** manual o por batería/ciclo |
| POST | `/experiments/{id}/predict/csv` | **Inferencia** subiendo un CSV (con métricas si trae la columna real) |
| GET | `/datasets` | **(v1.1)** Lista de datasets disponibles (NASA + subidos) |
| POST | `/datasets` | **(v1.1)** **Cargar datos**: subir un CSV propio para entrenar |
| GET | `/datasets/template` | **(v1.1)** Descargar plantilla CSV |
| GET | `/datasets/{dataset_id}` | **(v1.1)** Detalle de un dataset |
| DELETE | `/datasets/{dataset_id}` | **(v1.1)** Borrar un dataset subido (204) |

Todas las rutas llevan el prefijo `/api/v1`. Desde v1.1.0, todos los `GET /dataset/...` aceptan `?dataset_id=` (por defecto `nasa`).

### 4.2 Detalle con ejemplos

#### GET `/api/v1/health`
```json
{ "status": "ok", "api_version": "1.1.0", "app_version": "1.1.0", "dataset": { "loaded": true, "rows": 636 } }
```

#### GET `/api/v1/dataset/summary`
```json
{
  "source": "NASA Prognostics Center of Excellence - Battery Data Set (B0005, B0006, B0007, B0018)",
  "source_url": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
  "n_rows": 636, "n_batteries": 4, "nominal_capacity_ah": 2.0, "eol_capacity_ah": 1.4, "sha256_16": "34169ce7ec688b5f",
  "batteries": [
    { "battery_id": "B0005", "n_cycles": 168, "initial_capacity_ah": 1.85649, "final_capacity_ah": 1.32508,
      "min_soh": 0.643726, "eol_cycle": 125, "reaches_eol": true, "ambient_temperature": 24.0 },
    { "battery_id": "B0007", "n_cycles": 168, "initial_capacity_ah": 1.89105, "final_capacity_ah": 1.43246,
      "min_soh": 0.700228, "eol_cycle": null, "reaches_eol": false, "ambient_temperature": 24.0 }
  ]
}
```

#### GET `/api/v1/dataset/features`
```json
{
  "features": [
    { "name": "time_to_3_5v", "unit": "s", "description": "Tiempo desde el inicio de la descarga hasta que el voltaje cae a 3.5 V.",
      "source": "discharge", "min": 565.5, "max": 2314.5, "mean": 1516.42, "missing": 0 }
  ],
  "targets": [
    { "key": "soh", "column": "soh", "unit": "fracción (0-1)", "description": "Estado de salud = capacidad / capacidad nominal (2.0 Ah)." },
    { "key": "rul", "column": "rul", "unit": "ciclos", "description": "Vida útil restante = ciclos hasta que la capacidad cae por debajo de 1.4 Ah (EOL)." },
    { "key": "capacity", "column": "capacity_ah", "unit": "Ah", "description": "Capacidad descargada en el ciclo." }
  ]
}
```
`source` ∈ `meta | discharge | charge | impedance`. Las 14 características (en este orden):
`cycle, v_mean, v_std, time_to_3_8v, time_to_3_5v, v_at_500s, temp_max, time_to_temp_max, temp_rise, cc_charge_time, cv_charge_time, charge_temp_max, re_ohm, rct_ohm`.

#### GET `/api/v1/dataset/batteries/{battery_id}/degradation`
```json
{
  "battery_id": "B0005", "eol_capacity_ah": 1.4, "eol_cycle": 125,
  "points": [ { "cycle": 1, "capacity_ah": 1.85649, "soh": 0.928244, "rul": 124.0 },
              { "cycle": 2, "capacity_ah": 1.84633, "soh": 0.923164, "rul": 123.0 } ]
}
```
Gráfico sugerido: línea capacidad vs ciclo por batería + línea horizontal en `eol_capacity_ah` + marca vertical en `eol_cycle`.
`rul` es `null` en B0007.

#### GET `/api/v1/dataset/cycles?battery_id=B0018&limit=50&offset=0`
```json
{ "total": 132, "offset": 0, "limit": 50,
  "rows": [ { "battery_id": "B0018", "cycle": 1, "ambient_temperature": 24.0, "v_mean": 3.53978, "...": "resto de las 14 características",
              "capacity_ah": 1.855, "soh": 0.927502, "rul": 96.0, "eol_cycle": 97.0 } ] }
```
`battery_id` es opcional; `limit` 1–5000 (por defecto 1000).

#### GET `/api/v1/models`
```json
{
  "models": [
    { "key": "random_forest", "label": "Random Forest", "description": "...",
      "hyperparameters": [
        { "name": "n_estimators", "type": "int", "default": 300, "min": 10, "max": 1000, "nullable": false, "description": "Número de árboles." },
        { "name": "max_depth", "type": "int", "default": null, "min": 1, "max": 50, "nullable": true, "description": "Profundidad máxima (null = sin límite)." }
      ] },
    { "key": "gradient_boosting", "...": "..." }, { "key": "ridge", "...": "..." }, { "key": "svr", "...": "..." }, { "key": "knn", "...": "..." }
  ],
  "targets": [ { "key": "soh", "...": "..." } ],
  "batteries": ["B0005", "B0006", "B0007", "B0018"],
  "defaults": { "name": "ejemplo", "description": null, "target": "soh", "model_type": "random_forest",
                "hyperparameters": {}, "features": null, "test_batteries": ["B0006"], "random_state": 42 }
}
```
**Genera el formulario dinámicamente** a partir de `hyperparameters` (tipo, min, max, default, nullable). No escribas los rangos a mano.

#### POST `/api/v1/experiments` → 201
Request (solo `name` es obligatorio; el resto toma los valores por defecto):
```json
{
  "name": "RF SOH test B0006",
  "description": "opcional, máx 500 caracteres",
  "target": "soh",
  "model_type": "random_forest",
  "hyperparameters": { "n_estimators": 300, "min_samples_leaf": 2 },
  "features": null,
  "test_batteries": ["B0006"],
  "random_state": 42
}
```
Reglas que valida el backend (muestra el error si falla):
- `target` ∈ `soh | rul | capacity`. `model_type` ∈ claves de `/models`. Campos extra ⇒ `validation_error`.
- `features`: `null` = las 14; si se envía lista, solo nombres válidos y sin repetir.
- `test_batteries`: al menos 1 y debe quedar ≥ 1 batería para entrenar. Con `target = "rul"` **B0007 no puede ir en test**.

Response (es el mismo objeto que devuelve `GET /experiments/{id}`):
```json
{
  "id": "20261120T181032_rf-soh-test-b0006_15c2ea",
  "name": "RF SOH test B0006", "description": null,
  "created_at": "2026-11-20T18:10:32+00:00", "status": "completed",
  "config": {
    "target": "soh", "model_type": "random_forest",
    "hyperparameters": { "n_estimators": 300, "max_depth": null, "min_samples_leaf": 2 },
    "features": ["cycle", "v_mean", "...14 nombres..."],
    "train_batteries": ["B0005", "B0007", "B0018"], "test_batteries": ["B0006"], "random_state": 42
  },
  "metrics": {
    "train":         { "mae": 0.00107, "rmse": 0.00167, "r2": 0.9996, "mape_pct": 0.134, "n": 468 },
    "test":          { "mae": 0.03006, "rmse": 0.03691, "r2": 0.9137, "mape_pct": 3.915, "n": 168 },
    "baseline_test": { "mae": 0.11349, "rmse": 0.12785, "r2": -0.0354, "mape_pct": 15.25, "n": 168 },
    "per_battery":   { "B0006": { "mae": 0.03006, "rmse": 0.03691, "r2": 0.9137, "mape_pct": 3.915, "n": 168 } },
    "cv": { "strategy": "leave-one-battery-out (3 pliegues)", "mae_mean": 0.01099, "mae_std": 0.00113,
            "rmse_mean": 0.01324, "rmse_std": 0.00206, "r2_mean": 0.9731, "r2_std": 0.0122 },
    "train_seconds": 0.375
  },
  "feature_importance": [ { "feature": "time_to_3_5v", "importance": 0.0643, "std": 0.0042 } ],
  "train_feature_medians": { "cycle": 78.5, "v_mean": 3.52616, "...": "una por característica" },
  "n_train": 468, "n_test": 168,
  "warnings": [],
  "dataset": { "file": "nasa_battery_cycles.csv", "sha256_16": "34169ce7ec688b5f" },
  "environment": { "python": "3.13.18", "scikit_learn": "1.9.1", "numpy": "2.5.3", "pandas": "3.0.5" },
  "artifacts": { "model": "model.joblib", "predictions": "predictions.csv" }
}
```
Notas para la UI:
- `metrics.cv` puede ser `null` (si solo hay 1 batería de entrenamiento). `r2` y `mape_pct` pueden ser `null`.
- Las unidades de MAE/RMSE son las del objetivo: SOH en fracción (muéstralo también en puntos porcentuales: 0.030 → 3.0 %), RUL en ciclos.
- Muestra `warnings` si no está vacío (p. ej. "Baterías sin fin de vida observado (RUL desconocido) excluidas: ['B0007']").
- Muestra un indicador de "entrenando…" y deshabilita el botón mientras espera la respuesta.

#### GET `/api/v1/experiments?target=soh`
```json
{ "experiments": [
  { "id": "referencia_soh_random_forest", "name": "Referencia SOH - Random Forest (test B0006)",
    "created_at": "2026-11-20T18:12:28+00:00", "status": "completed", "target": "soh", "model_type": "random_forest",
    "test_batteries": ["B0006"], "metrics_test": { "mae": 0.03006, "rmse": 0.03691, "r2": 0.9137, "mape_pct": 3.915, "n": 168 } }
] }
```
Ordenados del más reciente al más antiguo. `target` es opcional.

#### GET `/api/v1/experiments/{id}/predictions?split=test&battery_id=B0006`
```json
{ "experiment_id": "referencia_soh_random_forest", "target": "soh",
  "points": [ { "battery_id": "B0006", "cycle": 1, "split": "test", "y_true": 1.01767, "y_pred": 0.941592, "error": -0.076078 } ] }
```
`split` ∈ `train | test` (opcional), `error = y_pred − y_true`. Gráficos sugeridos: (a) real y predicho vs ciclo,
(b) dispersión predicho vs real con la diagonal y = x, (c) residuales vs ciclo.

#### GET `/api/v1/experiments/{id}/feature-importance`
```json
{ "experiment_id": "referencia_soh_random_forest", "method": "permutation_importance (test, ΔMAE)",
  "importances": [ { "feature": "time_to_3_5v", "importance": 0.0643, "std": 0.0042 },
                   { "feature": "time_to_temp_max", "importance": 0.0265, "std": 0.0014 } ] }
```
Ya vienen ordenadas de mayor a menor. Puede haber importancias negativas o cercanas a 0 (variable que no aporta).

#### GET `/api/v1/experiments/{id}/model`
Descarga binaria `application/octet-stream` (`<id>.joblib`). Úsalo como enlace `<a href=... download>`.

#### POST `/api/v1/experiments/{id}/predict`
Dos formas, **exactamente una**:

a) Por batería del dataset (fácil para la demo; `cycles` es opcional, sin él predice todos los ciclos):
```json
{ "battery_id": "B0006", "cycles": [1, 100] }
```
b) Valores manuales (cada muestra debe traer **todas** las claves de `config.features`; `null` se imputa con la mediana):
```json
{ "samples": [ { "cycle": 100, "v_mean": 3.47, "v_std": 0.25, "time_to_3_8v": 250.0, "time_to_3_5v": 1200.0,
                 "v_at_500s": 3.70, "temp_max": 40.5, "time_to_temp_max": 2700.0, "temp_rise": 16.0,
                 "cc_charge_time": 2000.0, "cv_charge_time": 7500.0, "charge_temp_max": 29.5,
                 "re_ohm": null, "rct_ohm": null } ] }
```
Response:
```json
{
  "experiment_id": "referencia_soh_random_forest", "target": "soh", "unit": "fracción (0-1)",
  "features_used": ["cycle", "..."],
  "predictions": [
    { "battery_id": "B0006", "cycle": 1,   "prediction": 0.9416, "actual": 1.01767, "capacity_ah_estimated": 1.8832, "health_status": "bueno" },
    { "battery_id": "B0006", "cycle": 100, "prediction": 0.6976, "actual": 0.71561, "capacity_ah_estimated": 1.3953, "health_status": "fin_de_vida" }
  ],
  "warnings": []
}
```
- En modo `samples` cada predicción trae `"index"` en vez de `battery_id`/`cycle`, y `actual` es `null`.
- Campos extra según el objetivo: `soh` ⇒ `capacity_ah_estimated` + `health_status`; `capacity` ⇒ `soh_estimated` + `health_status`; `rul` ⇒ ninguno.
- Para el formulario manual usa `GET /dataset/features` (min/max/mean) para sugerir rangos y valores de ejemplo.

#### POST `/api/v1/experiments/{id}/predict/csv`
`multipart/form-data` con el campo **`file`** (.csv, máx 5 MB). Columnas: las de `config.features` (más `battery_id`
y `cycle` opcionales, que se devuelven en la respuesta). Respuesta igual a `/predict` en modo `samples`.
Ejemplo de CSV válido: las filas de `backend/data/processed/nasa_battery_cycles.csv` sin las columnas objetivo.

### 4.3 Novedades v1.1.0: cargar un dataset propio para entrenar

Todo es **compatible** con v1.0.0: si no envías `dataset_id`, todo funciona igual que antes con el dataset NASA.

**Formato del CSV** (una fila por ciclo de descarga, máx. 20 MB):
- Obligatorias: `battery_id` (texto), `cycle` (entero ≥ 1), `capacity_ah` (Ah medidos en ese ciclo).
- Opcionales: cualquiera de las 14 características (`v_mean`, `time_to_3_5v`, ...). Las que falten quedan vacías y no se
  pueden usar para entrenar. Debe haber al menos una además de `cycle`.
- `soh`, `rul` y `eol_cycle` **siempre se recalculan** en el backend (si vienen, se ignoran con una advertencia).
- Mínimo 10 ciclos válidos y 2 baterías. No puede haber `(battery_id, cycle)` repetidos.
- Plantilla con el formato exacto: `GET /api/v1/datasets/template` (descarga `plantilla_dataset.csv`).

#### POST `/api/v1/datasets` → 201
`multipart/form-data` con:
| Campo | Obligatorio | Descripción |
|---|---|---|
| `file` | sí | el `.csv` |
| `name` | sí | nombre visible (1–80 caracteres) |
| `nominal_capacity_ah` | no | capacidad nominal para calcular SOH (por defecto 2.0) |
| `eol_capacity_ah` | no | umbral de fin de vida para el RUL (por defecto 0.7 × nominal). Debe ser < nominal |

Response (mismo objeto que `GET /datasets/{id}` y que cada elemento de `GET /datasets`):
```json
{
  "id": "up_20261120T195705_mis-baterias_75212b",
  "name": "Mis baterías", "source": "upload", "is_default": false,
  "created_at": "2026-11-20T19:57:05+00:00",
  "nominal_capacity_ah": 2.0, "eol_capacity_ah": 1.4,
  "warnings": ["Características ausentes (quedan vacías y no se podrán usar para entrenar): ['v_std', ...]."],
  "n_rows": 468, "n_batteries": 3,
  "batteries": [ { "battery_id": "MI_B0005", "n_cycles": 168, "reaches_eol": true } ],
  "sha256_16": "7cf53852eb016ca9"
}
```
El dataset NASA aparece en `GET /datasets` con `"id": "nasa"`, `"source": "nasa"`, `"is_default": true` y `created_at: null`;
no se puede borrar (`403 dataset_protected`). Errores de validación: `422 invalid_dataset` con `details`
(p. ej. `{"columns": {"v_mean": 1}}` = una celda no numérica en `v_mean`).

#### Entrenar con un dataset subido
`POST /api/v1/experiments` acepta el campo nuevo **`dataset_id`** (por defecto `"nasa"`):
```json
{ "name": "Mi experimento", "dataset_id": "up_20261120T195705_mis-baterias_75212b",
  "target": "soh", "model_type": "random_forest", "test_batteries": ["MI_B0006"] }
```
- `test_batteries` debe usar baterías **de ese dataset** (el valor por defecto `["B0006"]` no existe en un dataset propio:
  la UI debe ofrecer las baterías de `GET /datasets/{id}` → `batteries`).
- Con `features: null` se usan solo las características que tienen datos. Pedir una sin datos da `invalid_features`.
- El experimento guarda `config.dataset_id` y `dataset.{id, name, nominal_capacity_ah, eol_capacity_ah}`.
  `GET /experiments` incluye `dataset_id` en cada resumen y acepta el filtro `?dataset_id=`.
- Para las gráficas de un dataset subido usa `GET /dataset/summary?dataset_id=...`,
  `GET /dataset/batteries/{b}/degradation?dataset_id=...` y `GET /dataset/cycles?dataset_id=...`.

#### Métricas en la inferencia
La respuesta de `POST /experiments/{id}/predict` y `/predict/csv` trae el campo nuevo **`metrics`**
(`{mae, rmse, r2, mape_pct, n}` o `null`):
- Por batería (`battery_id`) se calcula siempre contra el valor real del dataset.
- En `/predict/csv` se calcula si el CSV trae la columna del objetivo (`soh`, `rul` o `capacity_ah`). Entonces `actual` deja de ser `null`.
  Para `soh`, si falta la columna `soh` pero viene `capacity_ah`, el SOH real se calcula como `capacity_ah / nominal` (v1.1.1).
  Si no la trae, `metrics` es `null` y aparece una advertencia en `warnings`.

## 5. Qué debe tener el frontend (requisitos del curso)

1. **Datos:** resumen del dataset, tabla de baterías y curva de degradación (capacidad vs ciclo con la línea de EOL).
2. **Configurar y entrenar:** formulario generado desde `/models` (objetivo, modelo, hiperparámetros, baterías de test, características).
3. **Métricas:** MAE, RMSE, R², MAPE de test vs **línea base** y CV; tabla comparativa de experimentos del historial.
4. **Gráficos:** predicho vs real por ciclo, dispersión con diagonal, residuales, importancia de variables.
5. **Inferencia:** por batería/ciclo, manual y subiendo CSV; mostrar `health_status` con color.
6. **Persistencia:** historial de experimentos (vienen del backend), ver detalle, borrar, descargar modelo.
7. **Errores:** mostrar `error.message` (y `details` cuando sea una lista de campos) sin romper la página.
8. Una sección corta "Metodología" que explique las 5 decisiones técnicas de la sección 1.
9. **(v1.1) Cargar datos:** formulario para subir un CSV propio (con enlace a la plantilla), lista de datasets con
   advertencias, selector de dataset en Datos y en Entrenamiento, y métricas en la predicción por CSV.

## 6. Reglas de colaboración (para no cruzarnos)

### 6.1 Propiedad de carpetas
| Ruta | Dueño | El otro puede… |
|---|---|---|
| `frontend/**` | **ChatGPT** | nada (Claude no lo toca) |
| `backend/**` | **Claude** | nada (ChatGPT no lo toca, ni para "arreglar algo pequeño") |
| `CONTEXTO_CHATGPT.md` | Claude | proponer cambios (ver 6.4) |
| `README.md` (raíz) | Claude | ChatGPT solo añade/edita la sección `## Frontend` |
| `.github/workflows/backend-ci.yml` | Claude | — |
| `.github/workflows/frontend-*.yml` | ChatGPT | — |
| `.gitignore` (raíz) | Claude | ChatGPT pone sus ignores en `frontend/.gitignore` |

El frontend **nunca** lee archivos de `backend/` (ni el CSV ni `experiments/`): todo pasa por la API.

### 6.2 Ramas
- `main`: siempre funciona. Nadie hace push directo de trabajo grande.
- Backend: ramas `backend/<tema>` (p. ej. `backend/endpoint-comparacion`).
- Frontend: ramas `frontend/<tema>` (p. ej. `frontend/pagina-entrenamiento`).
- Se integra con Pull Request a `main`. Como cada rama toca solo su carpeta, no debe haber conflictos.
- Antes de empezar una rama: `git pull origin main`.

### 6.3 Commits
Formato: `<area>: <qué cambió en imperativo>`; `area` ∈ `frontend`, `backend`, `docs`, `ci`.
```
frontend: agregar gráfico de predicho vs real
frontend: manejar error invalid_split en el formulario
backend: agregar endpoint de comparación de experimentos
docs: actualizar contrato API a v1.1.0
```
Commits pequeños y frecuentes, en español, y que el proyecto corra después de cada uno.

### 6.4 Cómo pedir cambios al contrato de la API
1. El frontend **no inventa** endpoints ni campos. Si algo falta, se pide.
2. Escribe la petición en `frontend/PETICIONES_API.md` (es tuyo) con: endpoint deseado, request/response de ejemplo y para qué pantalla es.
   Santiago se la pasa a Claude.
3. Claude implementa, sube la versión y actualiza este archivo:
   - **Cambio compatible** (campo nuevo opcional, endpoint nuevo) ⇒ versión menor `1.x.0`, mismo prefijo `/api/v1`.
   - **Cambio incompatible** (renombrar/quitar campos, cambiar tipos) ⇒ `/api/v2` y se mantiene `/api/v1` mientras el frontend migra.
4. El frontend debe **ignorar campos desconocidos** (el backend puede añadir campos en versiones menores).
5. Comprueba la versión con `GET /api/v1/health` → `api_version`.

### 6.5 Historial de versiones del contrato
| Versión | Fecha | Cambios |
|---|---|---|
| 1.0.0 | 2026-11-20 | Versión inicial |
| 1.1.0 | 2026-11-20 | Subida de datasets propios (`/datasets`), `dataset_id` en experimentos y en `/dataset/*`, `metrics` en inferencia |
| 1.1.1 | 2026-11-20 | `/predict/csv` deriva el SOH real de `capacity_ah` si falta la columna `soh` |
