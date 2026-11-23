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
