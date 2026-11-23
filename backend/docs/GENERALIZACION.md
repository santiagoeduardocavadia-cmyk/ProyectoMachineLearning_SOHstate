# Generalización a condiciones no vistas (v1.2.0)

**Problema (observación del profesor):** el modelo entrenado con B0005/6/7/18 (todas a 24 °C y 2 A) "alucina" con CSV
de otras condiciones. Por ejemplo, Ridge da un MAE de 0.42 de SOH en B0031 a 43 °C, frente a 0.02 en B0006.
No avisa: devuelve un número igual de seguro.

**Qué se cambió en el backend:**
1. **Más datos reales y diversos.** Dataset `nasa_multitemp` con las 32 celdas del ZIP oficial de la NASA:
   - 4 °C: 16 celdas.
   - 22-24 °C: 14 celdas o tramos.
   - 43-44 °C: 7 celdas.
   - Corrientes de 1, 2 y 4 A, más onda cuadrada de 4 A.

   Se genera con `python -m scripts.prepare_dataset --multitemp`. La limpieza es la de `scripts/clean_cycles.py`:
   tramos por temperatura, descargas cortadas y caídas aisladas. B0050 y B0052 se descartan enteras por datos corruptos,
   y los tramos a 24 °C de B0038-B0040 por tener menos de 15 ciclos.
2. **Condiciones de operación como características:** `ambient_temperature` y `discharge_current_a`.
3. **Validación honesta:** la CV se agrupa por celda física, de modo que las copias aumentadas y los tramos de la misma celda
   van al mismo pliegue. `test_temperatures` permite dejar fuera una temperatura completa.
4. **Dominio de validez** en cada predicción (`dentro`/`borde`/`fuera`) y **selección automática** del experimento
   más parecido al CSV (`POST /api/v1/predict/auto`). Ver `app/ml/domain.py` y `app/ml/selection.py`.

## Resultados (MAE de SOH en puntos porcentuales; menor es mejor)

Script: `python -m scripts.evaluate_generalization`. Datos crudos: [`generalizacion/generalizacion_resultados.csv`](generalizacion/generalizacion_resultados.csv).
Semilla 42, hiperparámetros por defecto. "aug." es el CSV aumentado con `scripts/augment_dataset.py`
(ruido, interpolación y mixup), aplicado solo a las celdas de entrenamiento.

| Esquema | Test | Modelo | 14 car. | 14 car. + aug. | 16 car. | 16 car. + aug. |
|---|---|---|---|---|---|---|
| 1. entrenado solo a 24 °C | 4 °C | Gradient Boosting | 22.7 | 22.8 | 22.6 | 22.8 |
| 1. entrenado solo a 24 °C | 4 °C | Random Forest | 22.4 | 23.0 | 22.1 | 23.0 |
| 1. entrenado solo a 24 °C | 4 °C | Ridge | 27.5 | 27.9 | 10.1 | 9.5 |
| 1. entrenado solo a 24 °C | 43 °C | Gradient Boosting | 18.8 | 18.8 | 17.9 | 17.9 |
| 1. entrenado solo a 24 °C | 43 °C | Random Forest | 19.2 | 21.3 | 18.8 | 21.2 |
| 1. entrenado solo a 24 °C | 43 °C | Ridge | 51.9 | 52.5 | 39.7 | 42.2 |
| 2. temperatura no vista | 24 °C | Gradient Boosting | 7.0 | 8.2 | 5.9 | 11.8 |
| 2. temperatura no vista | 24 °C | Random Forest | 6.4 | 8.6 | 5.3 | 8.5 |
| 2. temperatura no vista | 24 °C | Ridge | 4.6 | 5.8 | 8.7 | 10.3 |
| 2. temperatura no vista | 4 °C | Gradient Boosting | 22.0 | 21.4 | 20.7 | 22.0 |
| 2. temperatura no vista | 4 °C | Random Forest | 20.4 | 22.6 | 18.6 | 22.6 |
| 2. temperatura no vista | 4 °C | Ridge | 20.8 | 21.2 | 24.0 | 24.9 |
| 2. temperatura no vista | 43 °C | Gradient Boosting | 4.2 | 3.6 | 4.6 | 3.4 |
| 2. temperatura no vista | 43 °C | Random Forest | 4.2 | 6.8 | 4.2 | 4.9 |
| 2. temperatura no vista | 43 °C | Ridge | 5.5 | 4.9 | 19.5 | 20.5 |
| 3. celda nueva (multitemperatura) | 24 °C | Gradient Boosting | 3.5 | 4.0 | 2.8 | 4.0 |
| 3. celda nueva (multitemperatura) | 24 °C | Random Forest | 2.4 | 2.9 | 2.4 | 2.8 |
| 3. celda nueva (multitemperatura) | 24 °C | Ridge | 4.4 | 4.6 | 4.5 | 4.8 |
| 3. celda nueva (multitemperatura) | 4 °C | Gradient Boosting | 4.4 | 4.8 | 3.7 | 4.6 |
| 3. celda nueva (multitemperatura) | 4 °C | Random Forest | 5.8 | 5.7 | 5.2 | 5.8 |
| 3. celda nueva (multitemperatura) | 4 °C | Ridge | 2.9 | 3.0 | 2.5 | 2.5 |
| 3. celda nueva (multitemperatura) | 43 °C | Gradient Boosting | 2.0 | 2.4 | 2.0 | 2.1 |
| 3. celda nueva (multitemperatura) | 43 °C | Random Forest | 1.5 | 1.9 | 1.4 | 1.8 |
| 3. celda nueva (multitemperatura) | 43 °C | Ridge | 3.2 | 3.4 | 3.2 | 3.4 |

Esquemas:
1. **Entrenado solo a 24 °C** (lo que había en v1.1): 4 celdas NASA → todas las celdas a 4 °C (14) o a 43 °C (7).
2. **Temperatura no vista:** entrenar con dos grupos de temperatura y probar en todas las celdas del tercero.
3. **Celda nueva (multitemperatura):** 5 pliegues por celda física. Se entrena con todas las temperaturas y se prueba en celdas
   que el modelo no vio, en condiciones que sí conoce. Es el uso real del modelo.

## Conclusiones

1. **Lo que resuelve el problema es entrenar con celdas reales de varias condiciones.**
   - El error en celdas nuevas a 4 °C baja de unos 22 a unos 2.5–5 puntos de SOH.
   - A 43 °C baja de 19–52 a 1.4–3.4 puntos (esquema 1 frente a 3), es decir, entre 5 y 15 veces menos error.
2. **El data augmentation no ayuda a generalizar a otras condiciones.**
   - En casi todas las combinaciones empeora entre 0.1 y 6 puntos.
   - El ruido, la interpolación y el mixup solo recombinan las condiciones que ya existen. No enseñan cómo cambia la
     química con la temperatura.
   - Se mantiene como herramienta opcional, con la regla de uso de `docs/PLAN_MEDICIONES.md`, que dice dónde sí mejora
     dentro de un mismo dataset.
3. **Una temperatura nunca vista sigue siendo difícil**, y eso hay que decirlo en la sustentación.
   - Extrapolar a 4 °C, que está fuera del rango 22–44 °C del entrenamiento, deja unos 19–25 puntos de error.
   - Interpolar o extrapolar cerca, como 43 °C desde 4 y 24 °C, deja unos 3.4–5.5 puntos con árboles.
   - Ridge con la temperatura como característica extrapola mal (19.5): aprende una recta en la temperatura que no
     se cumple físicamente.
   - Por eso el backend **avisa** con `domain.status = "fuera"` cuando la temperatura o la corriente no se vieron.
4. **Las características de condición (16 frente a 14)** mejoran a los árboles y, sobre todo, a Ridge cuando se entrena
   solo a 24 °C y se prueba a 4 °C (de 27.5 a 10.1). En el esquema 3 la mejora es pequeña, porque el modelo ya ve
   todas las condiciones.
5. **Modelo recomendado para la web:**
   - `referencia_multitemp_soh_random_forest`, con MAE de test de 0.020 en B0006 (24 °C), B0031 (43 °C) y B0047 (4 °C).
   - En el esquema 3 Gradient Boosting y Random Forest son los más estables. Ridge es el mejor a 4 °C.

## Datos a 14 y 34 °C
El dataset NASA solo tiene 4, 22–24 y 43–44 °C. Las fuentes con 15/25/35 °C, como Sandia (batteryarchive.org, Preger et al. 2020),
usan otras químicas (LFP, NCA, NMC) y otros protocolos.
- No se pudieron descargar desde el entorno de desarrollo: el proxy bloquea batteryarchive.org, OSTI, CALCE, Zenodo y Mendeley.
- Siguiente paso: descargarlas en un PC local y escribir un conversor al formato de subida, como `convert_kaggle.py`.
- Hasta entonces, la web detecta y avisa con `domain` cuando llega una temperatura así. No inventa una predicción "segura".
