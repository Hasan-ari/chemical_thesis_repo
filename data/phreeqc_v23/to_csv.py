"""Convert a PHREEQC run's input/output text files to CSV for inspection.

Originals under ``input/`` and ``output/`` are read-only; CSVs are written
to ``preview/`` next to them.

Usage:
    python to_csv.py            # converts run 1
    python to_csv.py 42         # converts run 42
    python to_csv.py --all      # converts every run found in input/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
PREVIEW_DIR = ROOT / "preview"


def convert_input(run: int) -> Path:
    src = INPUT_DIR / f"{run}_Input.txt"
    run_dir = PREVIEW_DIR / str(run)
    run_dir.mkdir(parents=True, exist_ok=True)
    dst = run_dir / "Input.csv"
    with src.open() as f, dst.open("w", newline="") as g:
        writer = csv.writer(g)
        writer.writerow(["field", "value"])
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                writer.writerow(parts)
    return dst


def convert_output(run: int) -> Path:
    src = OUTPUT_DIR / f"{run}_Output.txt"
    run_dir = PREVIEW_DIR / str(run)
    run_dir.mkdir(parents=True, exist_ok=True)
    dst = run_dir / "Output.csv"
    with src.open() as f, dst.open("w", newline="") as g:
        writer = csv.writer(g)
        for line in f:
            parts = line.split()
            if parts:
                writer.writerow(parts)
    return dst


def discover_runs() -> list[int]:
    return sorted(
        int(p.stem.split("_")[0])
        for p in INPUT_DIR.glob("*_Input.txt")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("run", type=int, nargs="?", default=None, help="run number (default: 1)")
    group.add_argument("--all", action="store_true", help="convert every run in input/")
    args = parser.parse_args()

    PREVIEW_DIR.mkdir(exist_ok=True)
    runs = discover_runs() if args.all else [args.run or 1]

    for run in runs:
        inp = convert_input(run)
        out = convert_output(run)
        print(f"{inp.relative_to(ROOT)}  +  {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
