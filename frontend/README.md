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

- **Datos** muestra las curvas de degradación, el umbral de fin de vida, los ciclos paginados y el catálogo de variables.
- **Entrenamiento** permite seleccionar objetivo, modelo, hiperparámetros, características y baterías de prueba. El servidor publica los modelos y sus límites.
- **Experimentos** permite comparar modelos y abrir sus métricas, validación cruzada, gráficos e inferencia por batería, valores manuales o CSV. Desde el detalle también se puede descargar el modelo o borrar el experimento.
- **Metodología** explica la separación por batería, la línea base, la validación cruzada, la selección de variables y el caso de B0007.

La coordinación entre ramas y el contrato de datos se describen en [CONTEXTO_CHATGPT.md](../CONTEXTO_CHATGPT.md). Las peticiones de cambios al contrato se documentan en `frontend/PETICIONES_API.md` cuando se necesitan.
