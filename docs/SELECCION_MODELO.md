# ¿Qué modelo de machine learning se usa y por qué? (v1.5.0)

**Respuesta corta:** para el SOH se usa **Random Forest**, entrenado con las 29 celdas multitemperatura. Se eligió con
**validación cruzada por celda física**: es el modelo que menos se equivoca, en promedio, en celdas que no vio
(2.8 puntos de SOH). SVR queda empatado con él, y los otros tres modelos quedan peor.

## 1. Los 5 modelos que se compararon

| Modelo | Idea en una frase |
|---|---|
| Regresión lineal (Ridge) | Una suma ponderada de las características (una "recta" en muchas dimensiones), con regularización L2. |
| K vecinos más cercanos (KNN) | Promedia el SOH de los k ciclos de entrenamiento más parecidos. |
| SVR (kernel RBF) | Ajusta una curva suave e ignora los errores menores que un margen ε. |
| Random Forest | Promedio de cientos de árboles de decisión entrenados con muestras al azar. |
| Gradient Boosting | Árboles pequeños en cadena; cada uno corrige el error del anterior. |

## 2. Cómo se eligió (validación cruzada)

1. Se apartan las **celdas de prueba** (B0006 a 24 °C, B0031 a 43 °C y B0047 a 4 °C). No se usan para elegir nada.
2. Las 29 celdas restantes se dividen en **10 pliegues por celda física**. En cada vuelta se entrena con 9 pliegues y se
   mide el error en el pliegue que quedó fuera. Así cada celda se evalúa una vez como "celda nueva". Los tramos de una
   misma celda a distintas temperaturas van siempre juntos, para que no haya fuga de información.
3. Para cada modelo se prueban algunos hiperparámetros (por ejemplo k = 5, 10 y 20 vecinos) y se queda el mejor.
4. Se elige el modelo con **menor MAE medio** en los 10 pliegues.
5. Para saber si la diferencia es real o casualidad, se usa la **prueba de Wilcoxon pareada**: compara pliegue a pliegue
   cada modelo con el mejor. Si p ≥ 0.05 y la diferencia es menor que un error estándar, se considera **empate**.
6. Solo al final se mide cada modelo en las celdas de prueba, **como confirmación**.

¿Por qué no se elige por el resultado de la prueba? Porque así la prueba dejaría de ser una medida honesta: se estaría
escogiendo el modelo que tuvo suerte con esas 3 celdas. Por ejemplo, SVR tiene el menor error en la prueba (1.48 frente
a 2.01), pero en 10 pliegues es menos estable que Random Forest (desviación 2.13 frente a 1.13).

Script reproducible: `python -m scripts.compare_models` (unos 2 minutos), desde `backend/`. Datos crudos:
[`backend/docs/seleccion_modelo/`](../backend/docs/seleccion_modelo/).

## 3. Resultados

Los errores de SOH están en puntos porcentuales: 2.80 significa que el modelo se equivoca en 2.8 puntos de SOH en promedio.

### SOH, multitemperatura (29 celdas, 4/24/43 °C)

Validación cruzada: 10 pliegues por celda física con 29 celdas de entrenamiento. Test: B0006, B0031, B0047. Errores (MAE) en puntos de SOH; menor es mejor. Línea base (predecir siempre la media): 12.92.

| Puesto | Modelo | MAE CV (media ± desv.) | p (Wilcoxon vs. el mejor) | ¿Empatado con el mejor? | MAE test (solo confirmación) | Hiperparámetros elegidos |
|---|---|---|---|---|---|---|
| 1 | **Random Forest** ✅ | 2.80 ± 1.13 | — | sí | 2.01 | n_estimators=300, max_depth=None, min_samples_leaf=5 |
| 2 | Support Vector Regression (RBF) | 3.07 ± 2.13 | 1.000 | sí | 1.48 | C=10.0, epsilon=0.01 |
| 3 | Gradient Boosting | 3.33 ± 1.60 | 0.275 | no | 3.22 | n_estimators=300, learning_rate=0.1, max_depth=2, subsample=0.8 |
| 4 | Regresión lineal (Ridge) | 3.95 ± 1.92 | 0.131 | no | 2.10 | alpha=0.1 |
| 5 | K vecinos más cercanos | 4.31 ± 2.49 | 0.002 | no | 3.03 | n_neighbors=10 |

Elegido: **Random Forest**. Empatado con: Support Vector Regression (RBF).

### SOH, NASA original (4 celdas a 24 °C)

Validación cruzada: 3 pliegues por celda física con 3 celdas de entrenamiento. Test: B0006. Errores (MAE) en puntos de SOH; menor es mejor. Línea base (predecir siempre la media): 7.73.

| Puesto | Modelo | MAE CV (media ± desv.) | p (Wilcoxon vs. el mejor) | ¿Empatado con el mejor? | MAE test (solo confirmación) | Hiperparámetros elegidos |
|---|---|---|---|---|---|---|
| 1 | **Regresión lineal (Ridge)** ✅ | 0.50 ± 0.39 | — | sí | 1.95 | alpha=1.0 |
| 2 | Gradient Boosting | 0.82 ± 0.22 | 0.500 | no | 3.14 | n_estimators=300, learning_rate=0.1, max_depth=3, subsample=0.8 |
| 3 | Random Forest | 1.10 ± 0.14 | 0.250 | no | 3.01 | n_estimators=300, max_depth=None, min_samples_leaf=2 |
| 4 | K vecinos más cercanos | 2.48 ± 0.52 | 0.250 | no | 4.33 | n_neighbors=20 |
| 5 | Support Vector Regression (RBF) | 4.40 ± 3.10 | 0.250 | no | 7.65 | C=1.0, epsilon=0.01 |

Elegido: **Regresión lineal (Ridge)**. Ningún otro modelo empata con él. Con solo 3 pliegues la prueba de Wilcoxon no puede bajar de p = 0.250: no puede declarar diferencias significativas y el empate se decide solo con la regla de un error estándar. El resultado es poco concluyente.

### RUL, NASA original (B0007 no llega al fin de vida)

Validación cruzada: 2 pliegues por celda física con 2 celdas de entrenamiento. Test: B0006. Errores (MAE) en ciclos; menor es mejor. Línea base (predecir siempre la media): 33.3.

| Puesto | Modelo | MAE CV (media ± desv.) | p (Wilcoxon vs. el mejor) | ¿Empatado con el mejor? | MAE test (solo confirmación) | Hiperparámetros elegidos |
|---|---|---|---|---|---|---|
| 1 | **K vecinos más cercanos** ✅ | 14.0 ± 3.0 | — | sí | 11.5 | n_neighbors=20 |
| 2 | Random Forest | 15.5 ± 0.1 | 1.000 | sí | 9.4 | n_estimators=300, max_depth=None, min_samples_leaf=2 |
| 3 | Gradient Boosting | 16.5 ± 1.4 | 0.500 | no | 8.5 | n_estimators=300, learning_rate=0.1, max_depth=2, subsample=0.8 |
| 4 | Regresión lineal (Ridge) | 16.7 ± 12.5 | 1.000 | no | 9.9 | alpha=10.0 |
| 5 | Support Vector Regression (RBF) | 30.9 ± 5.3 | 0.500 | no | 21.4 | C=1.0, epsilon=1.0 |

Elegido: **K vecinos más cercanos**. Empatado con: Random Forest. Con solo 2 pliegues la prueba de Wilcoxon no puede bajar de p = 0.500: no puede declarar diferencias significativas y el empate se decide solo con la regla de un error estándar. El resultado es poco concluyente.

## 4. Qué modelo usa la web en cada caso

| Situación | Modelo | Por qué |
|---|---|---|
| **Predecir** con un CSV de cualquier temperatura (4, 24 o 43 °C) | Random Forest multitemperatura | Los 5 modelos multitemperatura se entrenaron con las mismas celdas, así que se parecen igual a tus datos. El empate se resuelve con el menor error de validación cruzada, que es el de Random Forest. |
| Celdas a 24 °C con el dataset NASA original | Ridge | Gana en validación cruzada. Los árboles no pueden predecir un SOH menor que el mínimo que vieron (0.64), y B0006 llega a 0.58. |
| RUL (ciclos restantes) | Sin ganador claro | Solo 2 celdas de entrenamiento llegan al fin de vida, así que la validación cruzada tiene 2 pliegues y no puede separar KNN, Random Forest y Gradient Boosting. Hay que decirlo así en la sustentación. |

En Predecir, el desplegable **"Modelos considerados"** muestra ahora el MAE de validación cruzada de cada candidato.
El texto de "Modelo elegido" explica que se desempató por validación cruzada.

## 5. Qué decir en la sustentación

- *"Comparamos 5 modelos con el mismo protocolo: validación cruzada de 10 pliegues agrupada por celda, sin tocar las celdas
  de prueba. Elegimos el de menor error en celdas no vistas: Random Forest, con 2.8 puntos de SOH, frente a 12.9 de la línea
  base."*
- *"Con la prueba de Wilcoxon vimos que SVR está empatado con Random Forest, mientras que KNN es significativamente peor
  (p = 0.002). Entre los empatados nos quedamos con el de menor error medio y menor variabilidad."*
- *"No hay un modelo mejor para todo: con pocas celdas a una sola temperatura gana Ridge, porque extrapola; con muchas
  celdas y condiciones gana Random Forest. Por eso la web elige el modelo según los datos."*

## 6. Limitaciones

- La búsqueda de hiperparámetros usa los mismos pliegues que la comparación, así que los errores de CV son ligeramente
  optimistas. Lo es por igual para los 5 modelos, de modo que el orden entre ellos se mantiene.
- Con 2 o 3 pliegues (dataset NASA original) ninguna prueba estadística puede declarar diferencias. Ese resultado es solo
  orientativo.
