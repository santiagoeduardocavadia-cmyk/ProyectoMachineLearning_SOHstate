"""Vida restante en ciclos y en días según el uso (v1.4.0, ``POST /api/v1/predict/auto``).

Idea: el modelo predice el SOH de cada ciclo. Con el **ritmo de degradación** (cuánto SOH se pierde
por ciclo) se calcula cuántos ciclos faltan para llegar al fin de vida (SOH = 70 %, el criterio de la
NASA y de la industria). Los ciclos se convierten en días con un perfil de uso típico de una celda
Li-ion 18650 recargable (laptop, dron, linterna…).

Ritmo de degradación (se usa el MAYOR de los dos, por prudencia):
1. **Tendencia del propio CSV**: pendiente de una recta sobre el SOH *predicho* de todos los ciclos
   del CSV (al menos ``MIN_TREND_CYCLES``). No usa el valor real. Se probó también con solo los
   últimos 15 o 30 ciclos y fue peor (más sensible a la recuperación de capacidad tras reposos).
2. La mediana del ritmo de las 3 **celdas parecidas** de entrenamiento (``similar_cells``), que ya
   vivieron su curva completa. Si el CSV tiene pocos ciclos o no trae ``cycle``, es el único disponible.

Validación (``docs/CONCLUSIONES.md`` en la raíz del repositorio): con la curva real de las 6 celdas multitemperatura que llegan al
fin de vida, cortando a la mitad de su vida, el error mediano fue de unos 12 ciclos (≈ 24 %). Las celdas
se degradan más rápido al final (la "rodilla"), así que la estimación tiende a ser optimista.

Supuestos (deben decirse al presentar): un "ciclo" del laboratorio es una descarga completa a 1–4 A;
en el uso real hay descargas parciales (se cuentan como ciclos equivalentes) y envejecimiento por el
paso del tiempo, que aquí no se modela. Es una estimación orientativa, no una garantía.
"""
from __future__ import annotations

from typing import Any

import numpy as np

MIN_TREND_CYCLES = 10
MIN_FADE = 1e-5

# Ciclos completos equivalentes por día y corriente típica que pide cada aparato a UNA celda 18650.
# Con la corriente se calcula cuánto dura una carga: horas = capacidad actual (Ah) / corriente (A).
# (Las pilas AA alcalinas de un control remoto NO son recargables: no tienen ciclos.)
PROFILES: list[dict[str, Any]] = [
    {"key": "laptop", "name": "Laptop (uso diario)", "cycles_per_day": 1.0, "current_a": 0.5},
    {"key": "celular", "name": "Celular o power bank", "cycles_per_day": 1.0, "current_a": 1.0},
    {"key": "dron", "name": "Dron (3 vuelos al día)", "cycles_per_day": 3.0, "current_a": 4.0},
    {"key": "bicicleta", "name": "Bicicleta o patineta eléctrica", "cycles_per_day": 0.5, "current_a": 2.0},
    {"key": "herramienta", "name": "Herramienta eléctrica (taladro)", "cycles_per_day": 0.3, "current_a": 5.0},
    {"key": "linterna", "name": "Linterna recargable", "cycles_per_day": 0.1, "current_a": 0.5},
]
# Uso continuo: descargar y recargar sin parar, como en el laboratorio de la NASA.
DEFAULT_LAB_CURRENT_A = 2.0
DEFAULT_CHARGE_HOURS = 3.0


def _median_last(values, n: int = 5) -> float | None:
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v[-n:])) if v else None


def _soh_series(predictions: list[dict[str, Any]], target: str, nominal: float) -> list[float | None]:
    if target == "soh":
        return [p.get("prediction") for p in predictions]
    if target == "capacity":
        return [None if p.get("prediction") is None else p["prediction"] / nominal for p in predictions]
    return [None] * len(predictions)


def _trend_fade(cycles: list, soh: list) -> tuple[float | None, int]:
    pts = sorted((c, s) for c, s in zip(cycles, soh) if c is not None and s is not None)
    if len(pts) < MIN_TREND_CYCLES:
        return None, len(pts)
    x, y = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
    if np.ptp(x) == 0:
        return None, len(pts)
    fade = -float(np.polyfit(x, y, 1)[0])
    return (fade if fade > MIN_FADE else None), len(pts)


def _similar_fade(similar_cells: list[dict[str, Any]]) -> tuple[float | None, list[str]]:
    rates, used = [], []
    for c in similar_cells[:3]:
        si, sf, n = c.get("soh_initial"), c.get("soh_final"), c.get("n_cycles") or 0
        if si is not None and sf is not None and n > 1 and si - sf > 0:
            rates.append((si - sf) / (n - 1))
            used.append(c["battery_id"])
    return (float(np.median(rates)) if rates else None), used


def _hours(capacity_ah: float | None, current_a: float) -> float | None:
    return None if capacity_ah is None or current_a <= 0 else capacity_ah / current_a


def _profiles(remaining: float, capacity_ah: float | None, lab: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    continuous = dict(lab)
    if continuous.get("cycle_hours"):
        continuous["cycles_per_day"] = 24.0 / continuous["cycle_hours"]
    for p in ([continuous] if continuous.get("cycles_per_day") else []) + PROFILES:
        days = remaining / p["cycles_per_day"]
        h = _hours(capacity_ah, p["current_a"])
        out.append({**p, "cycles_per_day": round(p["cycles_per_day"], 2), "days": round(days, 1),
                    "years": round(days / 365.0, 2), "hours_per_charge": None if h is None else round(h, 2),
                    "use_hours_per_day": None if h is None else round(h * p["cycles_per_day"], 2)})
    return out


def _lab_profile(capacity_ah: float | None, current_a: float | None, charge_hours: float | None) -> dict[str, Any]:
    current = current_a if current_a and current_a > 0 else DEFAULT_LAB_CURRENT_A
    charge = charge_hours if charge_hours and charge_hours > 0 else DEFAULT_CHARGE_HOURS
    discharge = _hours(capacity_ah, current)
    return {"key": "continuo", "name": "Uso continuo sin parar (laboratorio)",
            "current_a": round(current, 2), "charge_hours": round(charge, 2),
            "discharge_hours": None if discharge is None else round(discharge, 2),
            "cycle_hours": None if discharge is None else round(discharge + charge, 2)}


def estimate(predictions: list[dict[str, Any]], target: str, similar_cells: list[dict[str, Any]],
             nominal: float, eol_capacity: float, discharge_current_a: float | None = None,
             charge_hours: float | None = None) -> dict[str, Any] | None:
    """Añade ``remaining_cycles`` a cada predicción y devuelve el bloque ``life_estimate``.

    ``discharge_current_a`` y ``charge_hours`` (medianas del CSV) definen el perfil de uso continuo.
    """
    eol_soh = eol_capacity / nominal
    if target == "rul":
        # El modelo ya predice ciclos restantes: solo se convierten a días.
        for p in predictions:
            p["remaining_cycles"] = None if p.get("prediction") is None else round(max(p["prediction"], 0.0), 1)
        last = [p["remaining_cycles"] for p in predictions if p["remaining_cycles"] is not None]
        if not last:
            return None
        remaining = float(np.median(last[-5:]))
        lab = _lab_profile(None, discharge_current_a, charge_hours)
        return {"method": "predicción directa del modelo de RUL", "eol_soh": eol_soh, "soh_current": None,
                "capacity_current_ah": None, "fade_per_cycle": None, "remaining_cycles": round(remaining, 1),
                "past_end_of_life": remaining <= 0, "continuous_use": lab,
                "profiles": _profiles(remaining, None, lab), "assumptions": _ASSUMPTIONS}

    soh = _soh_series(predictions, target, nominal)
    cycles = [p.get("cycle") for p in predictions]
    trend, n_used = _trend_fade(cycles, soh)
    similar, used = _similar_fade(similar_cells)
    # Conservador: si hay ambos ritmos se usa el mayor. Al inicio de la vida la tendencia del CSV es
    # casi plana y subestima la "rodilla" final; las celdas parecidas ya la vivieron completa.
    if trend and similar:
        fade = max(trend, similar)
        method = (f"el mayor entre la tendencia del CSV ({trend * 100:.3f} %/ciclo, {n_used} ciclos) y el ritmo de "
                  f"las celdas parecidas {', '.join(used)} ({similar * 100:.3f} %/ciclo)")
    elif trend:
        fade, method = trend, f"tendencia del SOH predicho en los {n_used} ciclos del CSV"
    elif similar:
        fade, method = similar, f"ritmo medio de las celdas parecidas ({', '.join(used)})"
    else:
        return None

    for p, s in zip(predictions, soh):
        p["remaining_cycles"] = None if s is None else round(max((s - eol_soh) / fade, 0.0), 1)
    ordered = [s for c, s in sorted(zip(cycles, soh), key=lambda t: (t[0] is None, t[0] or 0)) if s is not None]
    soh_now = float(np.median(ordered[-5:]))
    remaining = max((soh_now - eol_soh) / fade, 0.0)
    capacity_now = soh_now * nominal
    lab = _lab_profile(capacity_now, discharge_current_a, charge_hours)
    return {
        "method": method,
        "eol_soh": eol_soh,
        "soh_current": soh_now,
        "capacity_current_ah": round(capacity_now, 3),
        "fade_per_cycle": fade,
        "fade_trend_csv": trend,
        "fade_similar_cells": similar,
        "remaining_cycles": round(remaining, 1),
        "past_end_of_life": soh_now < eol_soh,
        "continuous_use": lab,
        "profiles": _profiles(remaining, capacity_now, lab),
        "assumptions": _ASSUMPTIONS,
    }


_ASSUMPTIONS = ("Estimación orientativa: un ciclo es una descarga completa equivalente; las descargas parciales "
                "cuentan como fracciones de ciclo. Una carga dura capacidad actual ÷ corriente típica de cada uso; a "
                "corrientes altas (dron, taladro) la celda entrega algo menos de capacidad y dura un poco menos. No "
                "incluye el envejecimiento por el paso del tiempo ni cambios de temperatura o corriente respecto a los "
                "del CSV. Fin de vida = 70 % del SOH. Error típico ±25-50 %, y tiende a ser optimista porque la "
                "degradación se acelera al final de la vida.")
