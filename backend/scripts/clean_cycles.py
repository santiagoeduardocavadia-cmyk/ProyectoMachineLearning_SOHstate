"""Limpia un CSV de ciclos (formato de subida) antes de entrenar o aumentar.

Uso (desde backend/):
    python -m scripts.clean_cycles ENTRADA.csv [--out SALIDA.csv]

Reglas (definidas tras revisar los CSV convertidos de Kaggle, ver docs/PLAN_MEDICIONES.md §A0):

1. **Separar por temperatura.** Si una batería se ensayó a varias temperaturas ambiente
   (p. ej. B0038: 12 ciclos a 24 °C y luego 44 °C), cada tramo pasa a ser una batería distinta,
   ``B0038_24C`` y ``B0038_44C``: la capacidad cambia por la temperatura, no por la degradación.
2. **Descargas cortadas.** Se descartan los ciclos con capacidad < 25 % del percentil 90 de su
   tramo (p. ej. los ~0.05 Ah de B0041 cuando la celda da ~1 Ah).
3. **Caídas aisladas.** Se descartan los ciclos con capacidad < 70 % de la mediana móvil
   (ventana centrada de 5 ciclos), como los 0.43 Ah de B0040 después de 1.64 Ah.
   La recuperación de capacidad tras un reposo (subidas pequeñas) NO se toca: es real.
4. **Tramos cortos.** Se descartan los tramos con menos de ``--min-cycles`` ciclos válidos.

``cycle`` conserva la numeración original (es la edad de la celda). Las columnas de objetivos
(soh, rul, eol_cycle) se eliminan: el backend las recalcula al subir el CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DROPOUT_FRACTION = 0.25
DROP_FRACTION = 0.70
ROLLING_WINDOW = 5
TARGET_COLUMNS = ["soh", "rul", "eol_cycle"]


def _segment_id(battery: str, temp: float, multi: bool) -> str:
    return f"{battery}_{int(round(temp)):02d}C" if multi else battery


def clean_cycles(df: pd.DataFrame, min_cycles: int = 15) -> tuple[pd.DataFrame, list[dict]]:
    """Devuelve el CSV limpio y un informe por tramo (filas antes/después y motivo)."""
    df = df.drop(columns=[c for c in TARGET_COLUMNS if c in df.columns]).copy()
    df["battery_id"] = df["battery_id"].astype(str)
    df = df[df["capacity_ah"].notna() & (df["capacity_ah"] > 0.01)]
    has_temp = "ambient_temperature" in df and df["ambient_temperature"].notna().any()
    out, report = [], []
    for battery, g in df.sort_values(["battery_id", "cycle"]).groupby("battery_id", sort=True):
        temps = g["ambient_temperature"].round() if has_temp else pd.Series(0, index=g.index)
        multi = temps.nunique() > 1
        for temp, seg in g.groupby(temps, sort=True):
            seg_id = _segment_id(battery, temp, multi)
            n0 = len(seg)
            seg = seg[seg["capacity_ah"] >= DROPOUT_FRACTION * seg["capacity_ah"].quantile(0.9)]
            n_dropout = n0 - len(seg)
            rolling = seg["capacity_ah"].rolling(ROLLING_WINDOW, center=True, min_periods=1).median()
            seg = seg[seg["capacity_ah"] >= DROP_FRACTION * rolling]
            n_drop = n0 - n_dropout - len(seg)
            kept = len(seg) >= min_cycles
            report.append({"battery_id": seg_id, "temperature": float(temp) if has_temp else None,
                           "cycles_in": n0, "dropouts": n_dropout, "isolated_drops": n_drop,
                           "cycles_out": len(seg) if kept else 0,
                           "status": "ok" if kept else f"descartado (< {min_cycles} ciclos)"})
            if kept:
                out.append(seg.assign(battery_id=seg_id))
    if not out:
        raise ValueError("No queda ningún tramo válido después de limpiar.")
    return pd.concat(out, ignore_index=True), report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, help="Por defecto <entrada>_limpio.csv")
    ap.add_argument("--min-cycles", type=int, default=15)
    a = ap.parse_args(argv)
    df = pd.read_csv(a.input, dtype={"battery_id": str})
    clean, report = clean_cycles(df, a.min_cycles)
    path = a.out or a.input.with_name(f"{a.input.stem}_limpio.csv")
    clean.to_csv(path, index=False, float_format="%.6g")
    print(pd.DataFrame(report).to_string(index=False))
    print(f"{path}: {len(df)} -> {len(clean)} ciclos, {clean['battery_id'].nunique()} baterías")


if __name__ == "__main__":
    main()
