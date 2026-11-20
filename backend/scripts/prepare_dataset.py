"""Descarga el NASA Battery Dataset y genera el CSV procesado.

Uso (desde la carpeta backend/):
    python -m scripts.prepare_dataset            # descarga (si falta) y procesa
    python -m scripts.prepare_dataset --no-download   # solo procesa los .mat de data/raw
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.no_download:
        download_raw(args.raw_dir)
    df = build_processed_dataset(args.raw_dir, args.out)
    print(f"Dataset procesado: {len(df)} ciclos -> {args.out}")
    print(df.groupby("battery_id").agg(ciclos=("cycle", "count"), cap_ini=("capacity_ah", "first"),
                                       cap_fin=("capacity_ah", "last"), eol=("eol_cycle", "first")))


if __name__ == "__main__":
    main()
