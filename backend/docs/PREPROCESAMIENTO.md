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

## 8. Datasets propios (API v1.1)
Un CSV subido con `POST /api/v1/datasets` pasa por el mismo tratamiento que el dataset NASA:
validación de tipos y ciclos repetidos, descarte de capacidades imposibles (`clean`) y cálculo de SOH/RUL (`add_targets`)
con la capacidad nominal y el umbral de EOL declarados al subirlo (por defecto 2.0 Ah y 70 % de la nominal).
Los objetivos que traiga el archivo se ignoran: así todas las fuentes usan exactamente la misma definición de SOH y RUL.
Las características ausentes quedan vacías y el entrenamiento solo usa las que tienen datos.

## 9. Limpieza y aumento de datos (scripts, fuera de la API)
El aumento se aplica a los **archivos de datos**, no dentro de la página: la API no cambia (v1.1.1).
Plan y decisiones: [docs/PLAN_MEDICIONES.md](../../docs/PLAN_MEDICIONES.md).

### 9.1 Limpieza: `python -m scripts.clean_cycles ENTRADA.csv`
Necesaria para los CSV de Kaggle B0038–B0044 (sobre NASA no cambia nada):
1. **Separar por temperatura.** Una batería ensayada a varias temperaturas se parte en tramos (`B0038_24C`, `B0038_44C`):
   el salto de capacidad al cambiar de 24 °C a 44 °C es efecto de la temperatura, no degradación.
2. **Descargas cortadas.** Se quitan los ciclos con capacidad < 25 % del percentil 90 del tramo
   (los ~0.05–0.1 Ah de B0041 y del tramo a 4 °C de B0042–B0044).
3. **Caídas aisladas.** Se quitan los ciclos con capacidad < 70 % de la mediana móvil de 5 ciclos (p. ej. los 0.43 Ah de B0040).
   Las subidas pequeñas tras un reposo (recuperación de capacidad) se conservan.
4. **Tramos cortos.** Se descartan los tramos con < 15 ciclos (los 12 ciclos a 24 °C de B0038–B0040).

`convert_kaggle --clean` aplica la misma limpieza al convertir.

### 9.2 Aumento: `python -m scripts.augment_dataset ENTRADA.csv --test B0006`
Añade **baterías virtuales** a partir de las de entrenamiento:
- `~ruidoN`: ruido gaussiano del 3 % de la σ de cada característica.
- `~interpN`: interpolación entre ciclos consecutivos.
- `~mixN`: mezcla con otra batería de capacidad parecida.

Reglas para no hacer trampa:
- Las baterías de `--test` se copian sin tocar y no generan baterías virtuales. Al entrenar se usa **exactamente** ese test.
- Si una celda física quedó partida en tramos de temperatura, **todos sus tramos** van a `--test`.
- Con un CSV aumentado, la **validación cruzada y el error de train salen optimistas**, porque las baterías virtuales caen
  en otros pliegues. La métrica válida es la de **test**.
