# Laboratorio de baterías

Interfaz en español para explorar los ciclos de las baterías NASA y evaluar modelos de estado de salud, capacidad y vida útil restante. Utiliza React, Vite y Chart.js. Obtiene los datos y los experimentos por HTTP mediante la API v1.

## Ejecutar

Usa Node.js 22.12 o superior y arranca la API siguiendo el [README del proyecto](../README.md). En otra terminal, desde la raíz del repositorio, ejecuta

```sh
cd frontend
npm ci
npm run dev
```

Abre la dirección local que muestra Vite. La URL predeterminada del backend es `http://localhost:8000/api/v1`. Para cambiarla, crea `frontend/.env.local` y define `VITE_API_URL` con la URL completa, incluido el prefijo `/api/v1`. Reinicia Vite tras cambiarla. Para producción, define esa variable antes de compilar.

```sh
npm run build
npm run preview
```

## Recorrido

- **Cargar datos** (API v1.1.0 o posterior) ofrece una plantilla CSV, carga de datasets de hasta 20 MB, advertencias y borrado de los datasets propios. El archivo y el nombre son obligatorios. La capacidad nominal y el umbral de fin de vida son opcionales.
- **Datos** muestra las curvas de degradación, el umbral de fin de vida, los ciclos paginados y las características del dataset seleccionado.
- **Entrenamiento** permite elegir el dataset, objetivo, modelo, hiperparámetros, características con datos y baterías de prueba. Para RUL solo se ofrecen baterías con fin de vida observado.
- **Experimentos** permite comparar modelos y abrir sus métricas, validación cruzada, gráficos e inferencia por batería, valores manuales o CSV. Desde el detalle también se puede descargar el modelo o borrar el experimento.
- En **Inferencia**, los resultados incluyen MAE, RMSE, R², MAPE y número de muestras cuando la API devuelve métricas. Para comparar un CSV con las predicciones, incluye la columna del objetivo (`soh`, `rul` o `capacity_ah`).
- **Metodología** explica la separación por batería, la línea base, la validación cruzada, la selección de variables y el caso de B0007.

La coordinación entre ramas y el contrato de datos se describen en [CONTEXTO_CHATGPT.md](../CONTEXTO_CHATGPT.md). Las peticiones de cambios al contrato se documentan en `frontend/PETICIONES_API.md` cuando se necesitan.

## Fiabilidad y temperatura (API v1.2.0)

En **Entrenamiento**, selecciona `nasa_multitemp` para explorar distintas condiciones. Puedes reservar una o más temperaturas de prueba, separadas por comas. La API aplica un margen de ±3 °C a cada valor. Al dejar el campo vacío se conserva la selección por batería. El detalle muestra las condiciones de entrenamiento y las métricas por temperatura cuando el experimento las incluye.

En **Inferencia**, el aviso resume la fiabilidad del lote y cada predicción indica su propio estado. **Selección automática** compara los experimentos guardados para el objetivo elegido, muestra el modelo seleccionado y explica las exclusiones. Las predicciones siguen visibles cuando las condiciones quedan fuera del dominio de entrenamiento.
