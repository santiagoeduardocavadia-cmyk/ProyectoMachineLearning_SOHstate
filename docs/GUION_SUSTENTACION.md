# Guion de sustentación (unos 15 minutos)

Sirve para la sustentación oral o para grabar el video. Antes de empezar, dejen el backend y el frontend corriendo
(ver [README](../README.md)) y prueben que se entrena un experimento.

| Min | Quién | Qué mostrar | Qué decir (idea clave) | Criterio de la rúbrica |
|---|---|---|---|---|
| 0–2 | _[integrante]_ | Diapositiva o README | El problema: estimar SOH y RUL sin hacer una descarga completa. Usuario: analista de confiabilidad. Dataset NASA, 4 celdas | Planteamiento (10 %) |
| 2–4 | _[integrante]_ | Diagrama de arquitectura (informe, §4) | Backend y frontend separados por una API versionada; persistencia en archivos; `Pipeline` sin fuga. Por qué no Streamlit | Arquitectura (15 %) |
| 4–6 | _[integrante]_ | Pestaña **Datos** | Las curvas cruzan 1.4 Ah. B0007 no llega al EOL. Las 14 variables y por qué **no** se usa la duración de la descarga | Integración ML (15 %) |
| 6–9 | _[integrante]_ | **Entrenamiento** → **Experimentos** (en vivo) | Entrenar SOH con Random Forest y test B0006. Señalar la línea base y la CV. Luego Ridge: gana porque los árboles no extrapolan | Implementación (20 %) y ML |
| 9–11 | _[integrante]_ | Detalle del experimento → inferencia | Predecir un ciclo y un CSV. Leer el resultado: SOH en % y estado. Mostrar un error: `n_estimators = 5` | Implementación y UX (10 %) |
| 11–13 | _[integrante]_ | Tabla §8.2 y pruebas (`pytest -q`) | El ranking cambia según la batería de test. 11 pruebas en CI. Sobreajuste detectado (train contra test) | Evaluación (15 %) |
| 13–15 | _[integrante]_ | Informe §10 y §11 | Solo 4 celdas a 24 °C: no generaliza. Sobrestimar el SOH es peligroso: es apoyo, no un BMS. Riesgo de pickle. Uso de IA en el desarrollo. Trabajo futuro | Ética (5 %) y documentación (10 %) |

## Preguntas probables y respuesta corta

**¿Por qué no un split aleatorio?**
Ciclos consecutivos de una misma batería son casi iguales. Si quedan a ambos lados del split, el modelo "memoriza"
y las métricas salen infladas. Separar por batería mide lo que importa: si el modelo sirve para una celda nueva.

**¿Qué es la fuga de información y dónde había riesgo?**
Es cuando el modelo recibe, directa o indirectamente, la respuesta que debe predecir. A corriente constante,
`capacidad = corriente × duración de la descarga`; si se incluyera la duración, el modelo solo multiplicaría.
También se evita imputando y escalando solo con datos de entrenamiento, dentro del `Pipeline`.

**¿Por qué Ridge le gana al Random Forest si es más simple?**
B0006 llega a un SOH de 0.58, por debajo del mínimo del entrenamiento (0.64). Un árbol solo puede predecir valores que vio,
así que se aplana. Un modelo lineal sigue la tendencia. Ojo: con B0018 como test, Random Forest gana; no hay un ganador
absoluto.

**¿Qué significa el R² negativo de la línea base?**
Que en el test predice peor que la media del propio test. Es esperable: la línea base usa la media del *entrenamiento*,
y la batería de test tiene otra distribución.

**¿Por qué el MAPE de RUL es tan alto si el MAE es bueno?**
Cerca del fin de vida la RUL real es casi 0. Un error de 5 ciclos sobre una RUL de 2 es un 250 %. Por eso para RUL
se reporta el MAE en ciclos.

**¿Por qué la CV de RUL tiene solo 2 pliegues?**
B0007 no llega al EOL, así que su RUL es desconocida (censurada). Con B0006 en test, solo quedan B0005 y B0018 para la CV.

**¿Sirve para la batería de mi celular o de un carro eléctrico?**
No sin volver a entrenar y validar: cambian la química, la temperatura, la corriente y el fabricante. Fuera de las
condiciones de entrenamiento no hay garantía, y sobrestimar el SOH tiene riesgo de seguridad.

**¿Qué variable es la más importante?**
`time_to_3_5v`: una celda degradada llega antes a 3.5 V. Le sigue `time_to_temp_max`: una mayor resistencia interna hace que
la celda se caliente antes. Ambas tienen sentido físico.

**¿Qué harían con más tiempo?**
Más baterías y condiciones, intervalos de predicción, avisos de entrada fuera de rango y modelos secuenciales (LSTM).

## Lista previa

- [ ] Backend y frontend corriendo, y `pytest -q` pasando (11 pruebas)
- [ ] Borrar o renombrar los experimentos de prueba (p. ej. `entrenamineto-1`)
- [ ] Capturas en `docs/capturas/`, enlazadas en el informe (§5)
- [ ] Completar los nombres de los integrantes y quién presenta cada parte
- [ ] Verificar la tabla de flujos manuales (informe, §8.3)
