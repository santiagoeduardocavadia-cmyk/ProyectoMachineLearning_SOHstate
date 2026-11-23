"""Extracción de características por ciclo de descarga.

Cada fila del dataset procesado representa UN ciclo de descarga de UNA batería.
Las características se calculan a partir de las señales medidas (voltaje,
corriente, temperatura) de la descarga y de la carga inmediatamente anterior,
y de la última medición de impedancia disponible.

Decisión de diseño importante (fuga de información / *data leakage*):
la capacidad descargada se calcula como ``Q = I * t_descarga``. Con corriente
constante de 2 A, la duración total de la descarga ES la capacidad dividida
por 2, así que usarla como entrada haría trivial el problema. Por eso **no**
se incluye la duración total ni la energía descargada como característica; en
su lugar se usan indicadores de salud parciales (tiempo hasta 3.8 V, tiempo
hasta 3.5 V, forma de la curva de voltaje, temperatura, carga CC/CV,
resistencias internas), que son los que se usan en la literatura de
pronóstico de baterías.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    unit: str
    description: str
    source: str  # discharge | charge | impedance | meta


FEATURES: list[FeatureInfo] = [
    FeatureInfo("cycle", "ciclos", "Número de ciclo de descarga (1 = primera descarga de la batería).", "meta"),
    FeatureInfo("ambient_temperature", "°C", "Temperatura ambiente del ensayo (condición de operación).", "meta"),
    FeatureInfo("discharge_current_a", "A", "Corriente media de descarga, en valor absoluto, durante la descarga efectiva "
                "(condición de operación: 1, 2 o 4 A en el dataset NASA).", "discharge"),
    FeatureInfo("v_mean", "V", "Voltaje medio durante la descarga efectiva (hasta el voltaje mínimo).", "discharge"),
    FeatureInfo("v_std", "V", "Desviación estándar del voltaje durante la descarga efectiva.", "discharge"),
    FeatureInfo("time_to_3_8v", "s", "Tiempo desde el inicio de la descarga hasta que el voltaje cae a 3.8 V.", "discharge"),
    FeatureInfo("time_to_3_5v", "s", "Tiempo desde el inicio de la descarga hasta que el voltaje cae a 3.5 V.", "discharge"),
    FeatureInfo("v_at_500s", "V", "Voltaje interpolado a los 500 s de descarga.", "discharge"),
    FeatureInfo("temp_max", "°C", "Temperatura máxima de la celda durante la descarga.", "discharge"),
    FeatureInfo("time_to_temp_max", "s", "Tiempo hasta alcanzar la temperatura máxima en la descarga.", "discharge"),
    FeatureInfo("temp_rise", "°C", "Incremento de temperatura (máx − inicial) en la descarga.", "discharge"),
    FeatureInfo("cc_charge_time", "s", "Duración de la fase de corriente constante (CC) de la carga previa (hasta 4.2 V).", "charge"),
    FeatureInfo("cv_charge_time", "s", "Duración de la fase de voltaje constante (CV) de la carga previa.", "charge"),
    FeatureInfo("charge_temp_max", "°C", "Temperatura máxima durante la carga previa.", "charge"),
    FeatureInfo("re_ohm", "Ω", "Resistencia del electrolito (Re) de la última impedancia (EIS) medida.", "impedance"),
    FeatureInfo("rct_ohm", "Ω", "Resistencia de transferencia de carga (Rct) de la última impedancia medida.", "impedance"),
]

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
# Condiciones de operación (v1.2.0). Sin ellas el modelo no puede distinguir "celda degradada"
# de "celda sana a otra temperatura o corriente", que es por lo que fallaba con datos de otras condiciones.
CONDITION_FEATURES: list[str] = ["ambient_temperature", "discharge_current_a"]
# Las 14 características originales (v1.0-v1.1); los experimentos de referencia NASA las conservan.
LEGACY_FEATURES: list[str] = [f for f in FEATURE_NAMES if f not in CONDITION_FEATURES]
DEFAULT_FEATURES: list[str] = list(FEATURE_NAMES)

TARGETS = {
    "soh": {"column": "soh", "unit": "fracción (0-1)", "description": "Estado de salud = capacidad / capacidad nominal (2.0 Ah)."},
    "rul": {"column": "rul", "unit": "ciclos", "description": "Vida útil restante = ciclos hasta que la capacidad cae por debajo de 1.4 Ah (EOL)."},
    "capacity": {"column": "capacity_ah", "unit": "Ah", "description": "Capacidad descargada en el ciclo."},
}


def _time_to_threshold(t: np.ndarray, v: np.ndarray, threshold: float) -> float:
    """Primer instante en que ``v`` cae por debajo de ``threshold`` (interpolado)."""
    below = np.nonzero(v <= threshold)[0]
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(t[0])
    # interpolación lineal entre la muestra anterior y la actual
    v0, v1, t0, t1 = v[i - 1], v[i], t[i - 1], t[i]
    if v0 == v1:
        return float(t1)
    return float(t0 + (threshold - v0) * (t1 - t0) / (v1 - v0))


def discharge_features(data: dict) -> dict:
    t = np.asarray(data["Time"], dtype=float)
    v = np.asarray(data["Voltage_measured"], dtype=float)
    temp = np.asarray(data["Temperature_measured"], dtype=float)
    if t.size < 5:
        raise ValueError("ciclo de descarga con muy pocas muestras")
    current = data.get("Current_measured")
    # Descarga efectiva: hasta el voltaje mínimo. Después la carga se desconecta
    # y el voltaje "rebota", lo cual no aporta información de degradación.
    end = int(np.argmin(v)) + 1
    t_e, v_e = t[:end] - t[0], v[:end]
    t_rel = t - t[0]
    i_tmax = int(np.argmax(temp))
    return {
        "v_mean": float(np.mean(v_e)),
        "v_std": float(np.std(v_e)),
        "time_to_3_8v": _time_to_threshold(t_e, v_e, 3.8),
        "time_to_3_5v": _time_to_threshold(t_e, v_e, 3.5),
        "v_at_500s": float(np.interp(500.0, t_e, v_e)) if t_e[-1] >= 500 else float("nan"),
        "temp_max": float(temp[i_tmax]),
        "time_to_temp_max": float(t_rel[i_tmax]),
        "temp_rise": float(temp[i_tmax] - temp[0]),
        "discharge_current_a": (float(np.mean(np.abs(np.asarray(current, dtype=float)[:end])))
                                if current is not None and np.size(current) == t.size else float("nan")),
    }


def charge_features(data: dict | None) -> dict:
    nan = float("nan")
    if data is None:
        return {"cc_charge_time": nan, "cv_charge_time": nan, "charge_temp_max": nan}
    t = np.asarray(data["Time"], dtype=float)
    v = np.asarray(data["Voltage_measured"], dtype=float)
    i = np.asarray(data["Current_measured"], dtype=float)
    temp = np.asarray(data["Temperature_measured"], dtype=float)
    if t.size < 5:
        return {"cc_charge_time": nan, "cv_charge_time": nan, "charge_temp_max": nan}
    t = t - t[0]
    reach = np.nonzero(v >= 4.195)[0]
    cc = float(t[reach[0]]) if reach.size else nan
    cv = nan
    if reach.size:
        # fin de CV: la corriente cae por debajo de 20 mA después de llegar a 4.2 V
        after = np.nonzero((np.arange(t.size) > reach[0]) & (np.abs(i) < 0.02))[0]
        cv = float((t[after[0]] if after.size else t[-1]) - t[reach[0]])
    return {"cc_charge_time": cc, "cv_charge_time": cv, "charge_temp_max": float(np.max(temp))}


def impedance_features(data: dict | None) -> dict:
    nan = float("nan")
    if data is None:
        return {"re_ohm": nan, "rct_ohm": nan}

    def _real(x):
        try:
            val = float(np.real(np.asarray(x).ravel()[0]))
        except (TypeError, ValueError, IndexError):
            return nan
        # algunos registros de EIS vienen corruptos (valores negativos o absurdos)
        return val if 0 < val < 1 else nan

    return {"re_ohm": _real(data.get("Re")), "rct_ohm": _real(data.get("Rct"))}
