# Plan para aplicar las mediciones

> Estado (2026-11-22): **Parte A ejecutada** (ver 0.5). **Parte B: fase 1 en curso** (fuentes verificadas, ver B1.1).
> Decisiones del equipo: sección 0.4.
> Parte A: cómo llevar al proyecto lo que ya se midió sobre *data augmentation*.
> Parte B: cómo ampliar las mediciones a 10 tipos de celda y a 5 temperaturas (4, 14, 24, 32 y 43 °C).

---

## 0. Punto de partida

### 0.1 Qué se midió
Se midió el MAE de test con y sin augmentation en los CSV de datos. El script es `backend/scripts/augment_dataset.py`. Las filas sintéticas van como baterías virtuales `B0005~ruido1`, `~interp1` y `~mix1`.

- **Protocolo:** se rota la batería de test y se promedia (n = número de baterías probadas). La augmentation de la API estuvo desactivada para medir solo el efecto de los datos. La batería de test nunca se usa para generar baterías virtuales.
- **Cómo leer la tabla:** negativo = mejora; positivo = empeora.

| Dataset | Temp. | SOH Ridge | SOH Random Forest | SOH Gradient Boosting | RUL Gradient Boosting | RUL Ridge |
|---|---|---|---|---|---|---|
| NASA B0005–B0018 | 24 °C | −10 % | 0 % | 0 % | **+13 %** (n=3) | **+24 %** (n=3) |
| Kaggle B0038–B0040 | 24 °C | −29 % | 0 % | **+48 %** | — | — |
| Kaggle B0029–B0032 | 43 °C | −40 % | −6 % | 0 % | — | — |
| Kaggle B0045–B0048 | 4 °C | +6 % | −42 % | −16 % | +23 % (n=1) | +1 % (n=1) |
| Kaggle B0041–B0044 | 4 °C / 22 °C | −22 % | −17 % | +11 % | −68 % (n=1) | +51 % (n=1) |

### 0.2 Conclusiones
1. **SOH:** la augmentation ayuda en la mayoría de los casos, sobre todo a Ridge (entre −10 % y −40 %) y a Random Forest en frío (4 °C). No es universal: Gradient Boosting empeora en 2 de 5 datasets.
2. **RUL:** no se aplica. En NASA empeora de forma consistente (n=3). En Kaggle solo una batería permite medir RUL (n=1), así que esos números no son concluyentes.
3. **Ruido alto:** con un ruido mayor al 3 % de σ el test empeoró en todos los modelos.

### 0.3 Problemas pendientes del código
- La augmentation que se agregó **dentro de la API/página** (campo `augmentation` en `POST /experiments`, `app/ml/augmentation.py` y el bucle manual de CV en `training.py`) se va a **retirar**: la decisión es aumentar los datos, no la página.
- Mientras esté, entrenar con un CSV aumentado **aumenta dos veces**.
- El bucle manual de CV tiene un **error 500**: con RUL, Gradient Boosting y pliegues muy pequeños falla con `ZeroDivisionError`.

### 0.4 Decisiones tomadas (2026-11-22)
| # | Pregunta | Decisión |
|---|---|---|
| 1 | ¿Sirven 15/25/35 °C (Sandia) como 14/24/32 °C? | **Sí.** Cada celda se asigna a la temperatura objetivo más cercana (tolerancia ±3 °C) y se reporta la temperatura real. |
| 2 | ¿Se aceptan datos sintéticos para los huecos? | **Sí.** Siempre marcados como sintéticos y **nunca en test**. |
| 3 | ¿El plomo-ácido es obligatorio? | **No.** Pasa a extensión opcional; los 10 tipos se completan con químicas de litio (tabla B1). |
| 4 | Batería de test de cada dataset de Kaggle | La eligió Claude (tabla A3), con el criterio de la sección A0. |

### 0.5 Mediciones con datos limpios (Parte A ejecutada)
Repetición de la tabla 0.1 tras la limpieza A0:
- **Qué se probó:** 111 combinaciones con 5 modelos, rotando la celda física de test. Cada vez se genera el CSV aumentado **sin** esa celda, se sube por la API y se entrena con `random_state = 42`.
- **Datos crudos:** `datos_aumentados/mediciones_augmentation.csv`.
- **Cómo leer la tabla:** cambio del MAE de test medio (negativo = mejora). Entre corchetes, el peor cambio en una batería individual.
- ✅ = cumple la regla de decisión: mejora la media, ninguna batería empeora más de un 5 % y n ≥ 2.

| Dataset (n celdas) | Ridge | Random Forest | Gradient Boosting | SVR | KNN |
|---|---|---|---|---|---|
| NASA (4) | −10 % [+84 %] | 0 % [+35 %] | 0 % [+20 %] | **−5 % [+2 %] ✅** | +2 % |
| B0029–B0032, 43 °C (4) | **−40 % [−12 %] ✅** | −6 % [+6 %] | 0 % | **−3 % [+1 %] ✅** | +4 % |
| B0045–B0048, 4 °C (4) | **−2 % [+1 %] ✅** | −6 % [+73 %] | 0 % [+49 %] | 0 % | −2 % [+11 %] |
| B0041–B0044, 4/22 °C (4) | −15 % [+11 %] | −13 % [+50 %] | **−5 % [+4 %] ✅** | +4 % | **−2 % [+3 %] ✅** |
| B0038–B0040, 44 °C (3) ⚠ | **−18 % [−10 %] ✅** | −2 % [+36 %] | +23 % | **−1 % [+4 %] ✅** | +8 % |
| **RUL** (todos) | +14 % a +24 % | — | +13 % a +57 % | — | — |

⚠ B0038–B0040 es el dataset de baja calidad: se reporta aparte y no cuenta para la regla (A3).

**Archivos entregados** (test fijo de la tabla A3; cambio del MAE de SOH en esa batería):

| Archivo | Ridge | Random Forest | Gradient Boosting | SVR | KNN |
|---|---|---|---|---|---|
| `nasa_aumentado_test-B0006` | −30 % | 0 % | 0 % | −3 % | −1 % |
| `…B0029_B0032_aumentado_test-B0032` | −42 % | −12 % | −8 % | 0 % | +9 % |
| `…B0045_B0048_aumentado_test-B0047` | −10 % | +73 % | +49 % | +11 % | −1 % |
| `…B0041_B0044_aumentado_test-B0043` | +11 % | −80 % | −6 % | −12 % | −1 % |
| `…B0038_B0040_aumentado_test-B0039_44C` | −10 % | −16 % | −23 % | +4 % | −5 % |

**Conclusiones:**
1. **RUL:** queda confirmado que **no se usa augmentation**. Empeora en todos los datasets y modelos.
2. **SOH:** la regla estricta se cumple en 7 de 20 combinaciones (sin contar B0038–B0040).
   - Ridge es el que más mejora en promedio, pero suele fallar la regla en una sola batería.
   - Ejemplo: en NASA, B0005 pasa de 0.0027 a 0.0051. Es un error ya muy pequeño, donde el cambio relativo exagera.
   - Propuesta para el equipo: medir la regla también con el cambio absoluto, por ejemplo "ninguna batería empeora más de 0.005 de SOH". Con ese criterio Ridge pasaría en NASA.
3. **Árboles (Random Forest y Gradient Boosting):** son los más inestables. En B0045–B0048 el archivo entregado empeora un 73 % con Random Forest. Se mantiene la regla A2: con árboles, usar el CSV aumentado solo donde la tabla lo marque.
4. **Regla A2 actualizada (por dataset):** usar el CSV aumentado solo en las combinaciones ✅ de la tabla de arriba. En el resto, usar el CSV limpio sin aumentar.

---

## Parte A: aplicar las mediciones de augmentation

### A0. Calidad de los datos de Kaggle (nuevo, antes de medir)
Una revisión de las capacidades, sin contar los ciclos de menos de 0.01 Ah que ya descarta la subida, encontró problemas.
La pérdida se calcula como la mediana de los 3 primeros ciclos frente a la de los 3 últimos:

| Dataset | Batería | Capacidad inicial → final | Problema | Acción |
|---|---|---|---|---|
| B0038–B0040 | B0038 | 1.09 → 1.53 Ah | Gana un 40 % de capacidad: no es degradación. | Revisar; probablemente excluir. |
| B0038–B0040 | B0039 | 0.16 → 1.32 Ah | 12 ciclos por debajo de 0.5 Ah: datos corruptos. | **Excluir.** |
| B0041–B0044 | B0041 | 0.06 → 0.85 Ah | 42 de 67 ciclos por debajo de 0.5 Ah. | **Excluir.** |
| B0041–B0044 | B0042–B0044 | ~1.7 → ~1.3 Ah | 46 ciclos por debajo de 0.5 Ah cada una (mezclan 4 °C y 22 °C). | Separar por temperatura y revisar esos ciclos. |
| B0045–B0048 | B0045 | 0.93 → 0.62 Ah | Empieza muy por debajo de sus compañeras (~1.5 Ah). | Mantener en entrenamiento, no como test. |
| B0029–B0032 | todas | ~1.8 → ~1.6 Ah | Sin problemas; pierden entre 8 % y 12 %. | — |

- [x] Limpieza implementada en `backend/scripts/clean_cycles.py`, también disponible como `convert_kaggle --clean`. Al revisar las series se vio que el problema no era solo corrupción, así que las reglas finales son distintas a las previstas:
  - Separar por temperatura: B0038–B0040 tienen 12 ciclos a 24 °C y luego 44 °C, y por eso parecían "ganar" capacidad.
  - Quitar descargas cortadas: menos del 25 % del percentil 90 del tramo.
  - Quitar caídas aisladas: menos del 70 % de la mediana móvil.
  - Descartar tramos con menos de 15 ciclos.

  Sobre NASA no cambia ni un ciclo. Resultado: B0041 conserva 25 ciclos válidos en lugar de excluirse entera, y B0042–B0044 quedan en tramos `_04C` y `_22C`. Salidas en `datos_aumentados/*_limpio.csv`.
- [x] Repetir la tabla 0.1 con los datos limpios. Las cifras actuales de B0038–B0040 y B0041–B0044 están contaminadas por estos ciclos.

### A1. Limpiar el código (requisito previo)
- [x] Revertir los cambios de la API: `git checkout -- CONTEXTO_CHATGPT.md README.md backend/app backend/docs/PREPROCESAMIENTO.md backend/scripts/run_reference_experiment.py`.
- [x] Borrar `backend/app/ml/augmentation.py` y `backend/tests/test_augmentation.py`.
- [x] Conservar `backend/scripts/augment_dataset.py` y `backend/tests/test_augment_dataset.py`.
- [x] Correr `pytest` y comprobar que la versión de la API vuelve a ser 1.1.1.

### A2. Regla de uso (sale de la tabla 0.1)

| Objetivo | Modelo | ¿Usar el CSV aumentado? |
|---|---|---|
| SOH / capacidad | Ridge | **Sí**, salvo en B0045–B0048 (+6 %) |
| SOH / capacidad | Random Forest | **Sí** (nunca empeora; hasta −42 % en frío) |
| SOH / capacidad | Gradient Boosting | **No** por defecto; solo si se mide una mejora en ese dataset |
| RUL | cualquiera | **No** |

### A3. Generar los CSV aumentados
Uno por dataset, cada uno con su batería de test fija:

Criterio, el mismo que en NASA con B0006: **la batería que más se degrada entre las que tienen datos válidos** (sección A0).

| Dataset | `--test` | Pérdida de capacidad | Motivo |
|---|---|---|---|
| NASA | **B0006** | la mayor del set | Referencia actual del proyecto. |
| B0029–B0032 (43 °C) | **B0032** | 12.3 % | La que más se degrada. |
| B0045–B0048 (4 °C, 1 A) | **B0047** | 24.0 % | La que más se degrada sin contar B0045, que es atípica. |
| B0041–B0044 (4/22 °C) | **B0043** (`B0043_04C` + `B0043_22C`) | 16.5 % a 4 °C | Tras limpiar, es la celda completa que más se degrada a 4 °C; B0041 solo tiene 25 ciclos. **Los dos tramos de temperatura de la misma celda van juntos a test**, para no filtrar información. |
| B0038–B0040 (44 °C) | **B0039_44C** | 25.2 % | Tras limpiar, es la que más se degrada; antes se eligió B0040, que en realidad perdía capacidad por dos descargas cortadas. Dataset de baja calidad: se reporta aparte. |

> Las elecciones de B0041–B0044 y B0038–B0040 cambiaron respecto de la primera versión del plan, porque la limpieza A0 cambió las curvas.

```bash
cd backend
python -m scripts.augment_dataset data/processed/nasa_battery_cycles.csv --test B0006
python -m scripts.augment_dataset ../baterias_43C_B0029_B0032.csv --test B0032
python -m scripts.augment_dataset ../baterias_4C_1A_B0045_B0048.csv --test B0047
python -m scripts.augment_dataset ../datos_aumentados/baterias_4C_B0041_B0044_limpio.csv --test B0043_04C B0043_22C
python -m scripts.augment_dataset ../datos_aumentados/baterias_24y44C_B0038_B0040_limpio.csv --test B0039_44C
```

### A4. Entrenar y registrar en la página
Por cada dataset:
1. Subir el CSV **original** y el **aumentado** en *Datasets*.
2. Entrenar el mismo modelo, con la misma semilla (`random_state = 42`) y **la misma batería de test** que se puso en `--test`.
3. Guardar ambos experimentos con nombres comparables, por ejemplo `SOH RF 43C test B0032 – original` y `… – aumentado`.
4. Comparar **solo la métrica de test**. La CV y el error de train del CSV aumentado salen optimistas: las baterías virtuales caen en otros pliegues y el train las incluye.

### A5. Mejoras opcionales al backend (sin augmentation en la página)
- [ ] **CV agrupada por batería de origen**: tratar `B0005~ruido1` como del grupo `B0005`, para que la CV del CSV aumentado deje de ser optimista. El mixup tiene dos baterías de origen: excluir esas filas del pliegue de su pareja.
- [ ] **Train solo con filas reales**: calcular `metrics.train` sin las baterías virtuales.
- [ ] **Avisar** en `warnings` si la batería de test elegida tiene baterías virtuales derivadas, que es fuga de información.

### A6. Documentar
- [x] En `backend/docs/PREPROCESAMIENTO.md`, una sección "Aumento de datos" con las técnicas, la tabla 0.1 y la regla A2.
- [ ] En `docs/INFORME_TECNICO.md` y el guion: explicar por qué no se aplica a RUL y por qué la CV del CSV aumentado no es comparable.

---

## Parte B: ampliar a 10 tipos de celda y 5 temperaturas

### B1. Matriz objetivo y lo que hay hoy

| Tipo de celda | 4 °C | 14 °C | 24 °C | 32 °C | 43 °C | Fuente candidata (por verificar) |
|---|---|---|---|---|---|---|
| 1. Li-ion LCO 18650 (NASA) | ✅ | ❌ | ✅ | ❌ | ✅ | NASA PCoE / Kaggle (ya disponible) |
| 2. Li-ion LCO prismática | | | ~25 | | | CALCE (U. Maryland) |
| 3. Li-ion NMC 18650 | | ~15 | ~25 | ~35 | | Sandia (batteryarchive.org) |
| 4. Li-ion NCA 18650 | | ~15 | ~25 | ~35 | | Sandia |
| 5. Li-ion LFP 18650 | | ~15 | ~25 | ~35 | | Sandia |
| 6. Li-ion LFP A123 | | | | ~30 | | MIT–Stanford–TRI (Severson 2019) |
| 7. Li-ion pouch (Kokam) | | | | | ~40 | Oxford Battery Degradation |
| 8. Li-ion LTO | | | | | | por buscar |
| 9. Li-ion NMC 18650 de alta energía (Samsung INR18650-20R) | | | ~25 | | | CALCE |
| 10. Li-ion NMC/LMO 18650 | | | ~25 | | | HNEI (Hawaii Natural Energy Institute) |
| *Ext. 1. Plomo-ácido VRLA/AGM (opcional)* | | | | | | solo sintético |
| *Ext. 2. Plomo-ácido inundado (opcional)* | | | | | | solo sintético |

✅ = hay datos · ~ = temperatura cercana; cuenta como la temperatura objetivo (decisión 1) · vacío = sin datos, se completa con datos sintéticos (decisión 2)

Asignación de temperaturas (decisión 1): 15 °C → 14 · 22–25 °C → 24 · 30–35 °C → 32 · 40–44 °C → 43.
Así, la celda LCO a 22 °C (B0042–B0044) cuenta como 24 °C, y la de Oxford a 40 °C como 43 °C.

### B1.1 Fuentes verificadas (fase 1, 2026-11-22)
Se confirmó que existen. **Todavía no se descargaron** ni se revisó la licencia de cada una.

| Tipo | Fuente | Qué se confirmó |
|---|---|---|
| 3–5. NMC, NCA y LFP 18650 | Sandia, Preger et al. 2020 ([OSTI 1650174](https://www.osti.gov/biblio/1650174), batteryarchive.org) | LFP (A123), NCA (Panasonic) y NMC (LG Chem), con variación de tasa de descarga, profundidad de descarga y temperatura (15/25/35 °C). |
| 2 y 9. LCO prismática y NMC INR18650-20R | [CALCE](https://web.calce.umd.edu/batteries/data/) | Hay datos de CS2, CX2 (con temperaturas variables en CX2_4), INR 18650-20R (2000 mAh) y A123. Tiene términos de uso propios. |
| 7. Pouch Kokam | [Oxford Battery Degradation Dataset 1](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac) | 8 celdas de 740 mAh (LCO/NCO) a 40 °C. **Solo hay caracterización cada 100 ciclos**: pocos puntos por celda. |
| 8. LTO | [IEEE DataPort, "LTO Battery Capacity Fading"](https://ieee-dataport.org/documents/lto-battery-capacity-fading) | 8 celdas, capacidad cada 250 ciclos hasta 3500, a 1 A entre 1.30 y 2.75 V. Sin temperatura variable. IEEE DataPort puede pedir cuenta. |
| 10. NMC/LMO (HNEI) | — | **No confirmado** en esta búsqueda; buscar de nuevo o reemplazarlo. |
| 6. LFP A123 (MIT–Stanford) | — | Pendiente de verificar. |

**Implicación:** Oxford y LTO solo dan capacidad cada 100–250 ciclos y **no tienen las curvas por ciclo** que usan las 14 características actuales. Para esas celdas solo se podría modelar capacidad frente a ciclos y temperatura, salvo que se rediseñen las características (fase 3).

### B2. Fases
1. **Conseguir los datos.** Descargar cada fuente y verificar licencia, temperaturas exactas, número de celdas y si llegan al fin de vida. Con eso se actualiza la matriz B1 con datos reales.
2. **Convertir.** Un conversor por fuente hacia el formato de subida, como el que ya existe para Kaggle (`convert_kaggle.py`).
3. **Backend para varias químicas.**
   - Cambiar las características atadas al LCO (`time_to_3_8v` y `time_to_3_5v` no sirven para LFP, LTO ni plomo) por características normalizadas a la ventana de voltaje de cada química.
   - Guardar metadatos por batería: química, voltajes, capacidad nominal y temperatura.
   - Usar la temperatura como característica.
   - Definir el fin de vida por química.
4. **Huecos (aprobado, decisión 2).** Generar datos sintéticos con un modelo semiempírico de envejecimiento:
   pérdida de capacidad = B · exp(−Eₐ / (R·T)) · Nᶻ.
   - **Calibración:** por química, con las temperaturas reales disponibles, por ejemplo LCO con 4/24/43 °C. Sirve para interpolar 14 y 32 °C.
   - **Tipos sin datos reales** (LTO y, si se hace, plomo): parámetros tomados de la literatura, citados en el informe.
   - **Marcado:** columna `synthetic = 1` y prefijo `SYN~` en el `battery_id`.
   - **Uso:** solo en entrenamiento. Las métricas se reportan siempre en celdas reales.
   - **Validación del generador:** calibrarlo dejando fuera una temperatura real (por ejemplo 24 °C en LCO), predecirla y comparar con los datos reales. Si el error supera un 5 % de SOH, el generador no se usa para esa química.
5. **Medir.** Con el mismo protocolo de 0.1, en tres esquemas:
   - **Por batería** (el actual): una celda nueva de un tipo y temperatura ya vistos.
   - **Dejando fuera una temperatura**: por ejemplo, entrenar sin 32 °C y probar en 32 °C.
   - **Dejando fuera una química**: es el caso más difícil.

   Cada esquema se mide con y sin CSV aumentado. El mixup se hace solo entre celdas de la misma química.
6. **Reportar.** Un mapa de calor del MAE por química × temperatura y una tabla con y sin augmentation.

### B3. Decisiones
Resueltas en la sección 0.4. Queda pendiente confirmar en la fase 1 que las fuentes de la tabla B1 existen y tienen
licencia de uso. Si alguna falla, se reemplaza por otra química de litio documentada.

---

## Protocolo común de medición (Partes A y B)
- **Split por batería.** El test nunca se aumenta ni genera baterías virtuales.
- **Semilla fija** (`random_state = 42`), tanto en el script como en el entrenamiento.
- **Rotar la batería de test** y reportar media y n. Con n = 1 el resultado se marca como "no concluyente".
- **Métricas:** MAE, RMSE y R² de test, comparados con la línea base (la media del entrenamiento).
- **Regla de decisión:** la augmentation se adopta en una combinación dataset × modelo solo si mejora el MAE de test medio y no empeora más de un 5 % en ninguna batería individual.
