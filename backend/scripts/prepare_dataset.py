"""Descarga el NASA Battery Dataset y genera el CSV procesado.

Uso (desde la carpeta backend/):
    python -m scripts.prepare_dataset            # descarga (si falta) y procesa
    python -m scripts.prepare_dataset --no-download   # solo procesa los .mat de data/raw
    python -m scripts.prepare_dataset --multitemp     # además, las 34 celdas a 4/24/43 °C (v1.2.0)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.data.preprocessing import build_processed_dataset, download_raw  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-download", action="store_true", help="no descargar, usar data/raw")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out", type=Path, default=config.PROCESSED_FILE)
    parser.add_argument("--multitemp", action="store_true",
                        help="generar también el dataset multitemperatura (34 celdas)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    batteries = config.ALL_BATTERIES if args.multitemp else config.DEFAULT_BATTERIES
    if not args.no_download:
        download_raw(args.raw_dir, batteries=batteries)
    df = build_processed_dataset(args.raw_dir, args.out)
    if args.multitemp:
        multi = build_processed_dataset(args.raw_dir, config.MULTITEMP_FILE, config.ALL_BATTERIES,
                                        split_conditions=True)
        print(f"Dataset multitemperatura: {len(multi)} ciclos, {multi['battery_id'].nunique()} baterías "
              f"-> {config.MULTITEMP_FILE}")
        print(multi.groupby("battery_id").agg(T=("ambient_temperature", "first"),
                                              I=("discharge_current_a", "median"), ciclos=("cycle", "count"),
                                              cap_ini=("capacity_ah", "first"), cap_fin=("capacity_ah", "last")))
    print(f"Dataset procesado: {len(df)} ciclos -> {args.out}")
    print(df.groupby("battery_id").agg(ciclos=("cycle", "count"), cap_ini=("capacity_ah", "first"),
                                       cap_fin=("capacity_ah", "last"), eol=("eol_cycle", "first")))


if __name__ == "__main__":
    main()
