# Conclusiones para la sustentación (v1.4.0)

Este documento resume qué se probó, qué funcionó y qué hay que decir en la sustentación. Los números detallados están en
[`backend/docs/GENERALIZACION.md`](../backend/docs/GENERALIZACION.md). Todos los errores son MAE de SOH en puntos
porcentuales (pp): un MAE de 3 significa que, en promedio, el modelo se equivoca en 3 puntos de SOH (por ejemplo, dice 82 %
cuando es 85 %).

## 1. El problema que señaló el profesor

El modelo de v1.1 se entrenaba con 4 celdas de la NASA, todas a 24 °C y 2 A. Con un CSV de otra temperatura "alucinaba":
devolvía un número con la misma seguridad, pero muy lejos de la realidad. Por ejemplo, con Ridge, la celda B0031 a 43 °C
tenía un error de 42 pp de SOH, frente a 2 pp en B0006 a 24 °C.

## 2. Qué se hizo

1. **Más celdas reales y de varias condiciones.** El dataset `nasa_multitemp` tiene 32 celdas físicas a 4, 22–24 y 43–44 °C,
   con corrientes de 1, 2 y 4 A. Cumple el pedido de "al menos 10 celdas a diferentes temperaturas".
2. **La temperatura y la corriente como variables del modelo** (16 características en lugar de 14).
3. **Validación honesta**: la validación cruzada agrupa por celda física, así que una celda nunca está a la vez en
   entrenamiento y en prueba, ni siquiera como copia aumentada o como otro tramo de temperatura.
4. **Aviso de "fuera de dominio"**: cada predicción dice si los datos se parecen a los del entrenamiento
   (`dentro`, `borde` o `fuera`).
5. **Selección automática del modelo** en la pestaña **Predecir**: se sube el CSV y el sistema escoge el experimento
   entrenado en condiciones más parecidas, sin mirar el valor real.
6. **Celdas parecidas**: se muestran las 5 celdas de entrenamiento con un comportamiento más parecido al del CSV.
7. **Vida restante en ciclos y en días** según el uso (laptop, celular, dron, bicicleta, taladro, linterna).

## 3. Resultados principales

| Situación | Error típico (MAE SOH) | Lectura |
|---|---|---|
| Entrenado solo a 24 °C, probado a 4 °C | 10–28 pp | No sirve: el modelo nunca vio frío |
| Entrenado solo a 24 °C, probado a 43 °C | 18–52 pp | No sirve: el modelo nunca vio calor |
| Entrenado con todas las temperaturas, celda nueva | **1.4–5.8 pp** | Sirve: es el uso real de la web |
| Una temperatura completa nunca vista: 4 °C | 19–25 pp | Sigue siendo difícil (extrapolar) |
| Una temperatura completa nunca vista: 43 °C | 3.4–5.5 pp (árboles) | Aceptable (está cerca de lo visto) |

Prueba con la pestaña **Predecir** y el modelo `referencia_multitemp_soh_random_forest` (las tres celdas están en el test, el
modelo no las vio):

| CSV | Temperatura | Dominio | MAE SOH | R² | Celdas parecidas |
|---|---|---|---|---|---|
| `inferencia_24C_B0006.csv` (168 ciclos) | 24 °C | dentro | 2.5 pp | 0.94 | B0005, B0018, B0044_22C |
| `inferencia_43C_B0031.csv` (40 ciclos) | 43 °C | borde | 0.5 pp | 0.82 | B0029, B0030, B0032 |
| `inferencia_4C_B0047.csv` (69 ciclos) | 4 °C | dentro | 1.7 pp | 0.90 | B0048, B0046, B0042_04C |
| `inferencia_24C_B0006_primeros40.csv` (celda sana) | 24 °C | dentro | 3.2 pp | −0.20 | B0043_22C, B0042_22C, B0007 |

El R² negativo de la celda sana no es un error grave: con solo 40 ciclos casi iguales, la varianza del SOH real es muy
pequeña y cualquier error la supera. El MAE (3.2 pp) es el número que importa. Además, al inicio el SOH real pasa de 100 %
y un Random Forest no puede predecir por encima de lo que vio, así que lo subestima un poco.

## 3.1 ¿Qué modelo se usa y por qué?

Se compararon 5 modelos con validación cruzada de 10 pliegues por celda física. **Random Forest** tiene el menor error
en celdas no vistas (2.8 pp), empatado con SVR (Wilcoxon p = 1.0). Gradient Boosting, Ridge y KNN quedan peor, y KNN de
forma significativa (p = 0.002). En Predecir, si varios experimentos se parecen igual a tus datos, gana el de menor
error de validación cruzada. Detalle en [SELECCION_MODELO.md](SELECCION_MODELO.md).

## 4. ¿Sirvió el data augmentation?

**Respuesta corta: no para el problema del profesor.**

- En casi todas las combinaciones de modelo y esquema, entrenar con datos aumentados **empeoró** el error entre 0.1 y 6 pp.
- El ruido, la interpolación entre celdas y el mixup solo **recombinan** las condiciones que ya existen. No pueden enseñarle al
  modelo cómo cambia la química de la celda a 4 °C si nunca vio una celda a 4 °C.
- Lo que sí redujo el error entre 5 y 15 veces fue **añadir celdas reales** de otras temperaturas.

Conclusión para decir: *"Probamos data augmentation como sugirió el profesor y lo medimos con validación por celda física.
No mejoró la generalización a otras temperaturas. Lo que funcionó fue entrenar con datos reales de esas temperaturas. Dejamos el
augmentation como herramienta opcional para datasets muy pequeños de una sola condición."*

## 5. Vida restante en días

**Cómo se calcula.** El modelo predice el SOH de cada ciclo. Con el ritmo de pérdida de SOH por ciclo se calcula cuántos ciclos
faltan para llegar al 70 % (fin de vida, el criterio de la NASA). Por prudencia, se usa el **mayor** de dos ritmos: la tendencia
del propio CSV y el ritmo de las 3 celdas parecidas, que ya vivieron su curva completa. Los ciclos se pasan a días con un uso
típico:

| Uso | Corriente por celda | Una carga dura (celda de 1.78 Ah) | Ciclos por día |
|---|---|---|---|
| Uso continuo sin parar (laboratorio) | la del CSV (2 A) | 54 min de descarga + 2 h 46 min de recarga = **3 h 40 min por ciclo** | ≈ 6.6 |
| Dron (3 vuelos al día) | 4 A | 27 min | 3 |
| Laptop | 0.5 A | 3 h 34 min | 1 |
| Celular o power bank | 1 A | 1 h 47 min | 1 |
| Bicicleta o patineta eléctrica | 2 A | 53 min | 0.5 |
| Herramienta eléctrica (taladro) | 5 A | 22 min | 0.3 |
| Linterna recargable | 0.5 A | 3 h 34 min | 0.1 |

**Cuánto dura un ciclo.** Una carga dura capacidad actual ÷ corriente: una celda con 1.78 Ah a 0.5 A dura unas 3.6 h.
En **uso constante** (descargar y recargar sin parar, como en el laboratorio) el ciclo completo dura la descarga más la
recarga, medidas en el propio CSV (fases CC y CV de la carga). Para B0006 son ≈ 3 h 40 min por ciclo, ≈ 6.6 ciclos al día,
así que con sus 92 ciclos restantes le quedarían ≈ 14 días de uso continuo. Las corrientes de cada aparato son típicas por
celda 18650 y son un supuesto.

Las pilas AA del control remoto del televisor no aparecen porque son alcalinas y no recargables: no tienen ciclos de carga.
Las celdas de la NASA son Li-ion 18650, como las de laptops, power banks, drones, bicicletas eléctricas y linternas.

**Qué tan buena es la estimación.**
- Con las 6 celdas multitemperatura que llegan al fin de vida, cortando la curva a la mitad, el error mediano fue de unos
  12 ciclos (≈ 24 %).
- Ejemplo: B0006 con sus primeros 40 ciclos estima **92 ciclos** restantes; en la realidad le quedaban **69** (llegó al 70 % en
  el ciclo 109). Con un dron son ≈ 31 días; con una laptop, ≈ 3 meses; sin parar, ≈ 14 días.
- La estimación tiende a ser **optimista**: las celdas se degradan más rápido al final de la vida (la "rodilla"). Por eso se
  usa el ritmo mayor y se dice el margen (±25–50 %).

**Supuestos que hay que decir.** Un ciclo es una descarga completa equivalente (dos medias descargas = un ciclo). No incluye el
envejecimiento por el paso del tiempo ni cambios de temperatura o de corriente respecto a los del CSV. Es una estimación
orientativa, no una garantía.

## 6. Cómo leer la tabla de resultados de Predecir

Cada fila es un ciclo de la celda, **en orden de tiempo** (ciclo 1, 2, 3…). Por eso al principio todo sale **bueno** y, al ir
pasando páginas, aparece **degradado** y luego **fin de vida**: es la celda envejeciendo ciclo a ciclo.

| Estado | SOH | Qué significa |
|---|---|---|
| Bueno | ≥ 85 % | Conserva casi toda su capacidad |
| Degradado | 70 – 85 % | Funciona, pero dura menos por carga |
| Fin de vida | < 70 % | Debe reemplazarse |

La columna "Ciclos restantes" baja fila a fila, y la de "Días de vida" la convierte según el uso elegido. Cerca del 70 % la
etiqueta puede saltar entre "degradado" y "fin de vida" porque el error del modelo (2–3 pp) es del mismo tamaño que la distancia
al umbral.

## 7. Limitaciones (decirlas evita preguntas difíciles)

1. **No hay datos a 14 °C ni a 34 °C.** La NASA solo midió a 4, 22–24 y 43–44 °C. Otras fuentes (Sandia, CALCE) usan otras
   químicas y no se pudieron descargar desde el entorno de desarrollo. La web **avisa** cuando llega una temperatura no vista,
   en lugar de inventar una predicción segura.
2. **Una temperatura nunca vista sigue siendo difícil**, sobre todo lejos del rango (4 °C desde 22–44 °C: unos 20 pp de error).
3. **Una sola química (Li-ion 18650, 2 Ah).** Para otra batería hay que volver a entrenar y validar.
4. **La vida en días es aproximada** (±25–50 %) y optimista al final de la vida.
5. **Es apoyo a la decisión, no un BMS.** Sobrestimar el SOH tiene riesgo de seguridad.

## 8. Mensaje final para la sustentación

1. Un modelo solo es fiable en las condiciones que vio. Con 4 celdas a 24 °C fallaba en frío y en calor.
2. La solución fue **más datos reales y diversos** (32 celdas, 3 temperaturas, 3 corrientes), no datos sintéticos.
3. Cuando no hay datos de una condición, lo correcto es **avisar**, y la web lo hace.
4. La pestaña **Predecir** escoge sola el modelo adecuado, explica con qué celdas se parece la nueva y traduce el resultado a algo
   que entiende cualquiera: cuántos días de uso le quedan a la batería.
