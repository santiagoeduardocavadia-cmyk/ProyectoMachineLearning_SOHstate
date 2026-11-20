# Preprocesamiento y metodología

## 1. Fuente
NASA Prognostics Center of Excellence (PCoE), *Battery Data Set*
(<https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip>, subconjunto `BatteryAgingARC-FY08Q4`).
Celdas 18650 de 2.0 Ah: B0005, B0006, B0007 y B0018, a 24 °C.

- **Carga:** CC 1.5 A hasta 4.2 V, luego CV hasta 20 mA.
- **Descarga:** CC 2 A hasta 2.7 / 2.5 / 2.2 / 2.5 V (B5 / B6 / B7 / B18).
- **Impedancia:** EIS de 0.1 Hz a 5 kHz (Re, Rct).
- **Fin de vida (EOL):** 30 % de pérdida, de 2.0 Ah a 1.4 Ah.

Reproducir: `cd backend && python -m scripts.prepare_dataset` (descarga ≈210 MB y regenera
`data/processed/nasa_battery_cycles.csv`). El CSV procesado ya está versionado (huella `sha256_16` visible en
`GET /api/v1/dataset/summary` y en cada experimento).

## 2. Unidad de análisis
Una fila = **un ciclo de descarga** de una batería (636 filas). Para cada descarga se asocian:
- la **carga inmediatamente anterior** (cada carga se usa una sola vez);
- la **última medición de impedancia válida** (Re y Rct en (0, 1) Ω; valores corruptos se descartan).

## 3. Características (14)
| Nombre | Fuente | Por qué indica degradación |
|---|---|---|
| `cycle` | uso | Envejecimiento por uso acumulado |
| `v_mean`, `v_std` | descarga | La curva de voltaje se desplaza hacia abajo y cambia de forma |
| `time_to_3_8v`, `time_to_3_5v` | descarga | Una celda degradada cae antes a un mismo voltaje |
| `v_at_500s` | descarga | Voltaje a tiempo fijo, sensible a la resistencia interna |
| `temp_max`, `time_to_temp_max`, `temp_rise` | descarga | Mayor resistencia ⇒ más calor; el pico térmico llega antes |
| `cc_charge_time`, `cv_charge_time`, `charge_temp_max` | carga previa | Menos capacidad ⇒ fase CC más corta |
| `re_ohm`, `rct_ohm` | impedancia | Crecen con el envejecimiento (SEI, pérdida de litio) |

**Descarga efectiva:** se corta en el voltaje mínimo. Después, la carga se desconecta y el voltaje "rebota", lo cual no es
información de degradación.

**Fuga de información evitada:** no se incluyen la duración total de la descarga ni la energía descargada, porque con corriente
constante `capacidad = 2 A × duración`: sería entregarle la respuesta al modelo.

## 4. Objetivos
- `soh = capacity_ah / 2.0`.
- `rul = eol_cycle − cycle` (recortado a ≥ 0), donde `eol_cycle` es el primer ciclo con capacidad < 1.4 Ah.
  - B0007 nunca baja de 1.4 Ah ⇒ RUL **censurado** (`NaN`), se excluye de los experimentos de RUL.
  - Los ciclos posteriores al EOL se excluyen del entrenamiento/evaluación de RUL (RUL constante 0, la batería ya "murió").

## 5. Limpieza
- Se descartan ciclos con capacidad ≤ 0 o > 1.2 × nominal (ninguno en este subconjunto).
- Valores faltantes (3 ciclos sin carga previa, 57 sin impedancia válida previa) se imputan con la **mediana del
  entrenamiento** dentro del `Pipeline` (sin mirar el conjunto de test).

## 6. Protocolo de evaluación
- **Split por batería** (por defecto test = B0006, la que más se degrada): evita que ciclos casi idénticos de la misma celda
  queden a ambos lados del split.
- **Validación cruzada leave-one-battery-out** en las baterías de entrenamiento.
- **Línea base:** predecir la media del entrenamiento. Un modelo útil debe superarla con claridad.
- **Importancia por permutación** sobre el test (aumento del MAE al desordenar cada variable).

## 7. Resultados de referencia (test = B0006, `random_state = 42`)
| Experimento | MAE | RMSE | R² | Línea base MAE |
|---|---|---|---|---|
| SOH – Random Forest | 0.030 (3.0 pp) | 0.037 | 0.914 | 0.113 |
| SOH – Ridge | 0.020 (2.0 pp) | 0.020 | 0.975 | 0.113 |
| RUL – Gradient Boosting | 11.4 ciclos | 13.0 | 0.830 | 27.3 |

**Observación para el análisis:** B0006 llega a un SOH de 0.58, menor que el mínimo visto en entrenamiento (0.64).
Los modelos de árboles **no extrapolan** (su predicción se aplana al final de la curva), mientras que Ridge sí sigue la tendencia.
Por eso Ridge gana en este split pese a ser más simple: es una limitación conocida de los árboles, no un error.

Variable más importante en SOH: `time_to_3_5v`, seguida de `time_to_temp_max`.
