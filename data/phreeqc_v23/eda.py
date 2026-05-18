"""Per-column EDA on PHREEQC output trajectories.

Loads every ``output/*_Output.txt`` file as-is (101 rows including t=0),
stacks them into a ``(n_runs, 101, 13)`` array, and produces:

* ``eda/summary.csv``           -- per-column min/max/mean/std and per-run range
* ``eda/trajectories.png``      -- 12-panel overlay of all runs (median in red)
* ``eda/trajectories_zoomed.png`` -- same overlay but each y-axis zoomed to
                                     the post-t=0 range so the across-run
                                     spread is visible on narrow columns

Originals are untouched.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
EDA_DIR = ROOT / "eda"

COLUMNS = [
    "time_d", "pH", "Ptot_atm", "pH2_atm", "pCO2_atm", "pCH4_atm",
    "CH4_g_mol", "H2_g_mol", "CO2_g_mol",
    "SO4", "Formate", "Acetate", "Ca",
]


def load_all() -> np.ndarray:
    files = sorted(
        OUTPUT_DIR.glob("*_Output.txt"),
        key=lambda p: int(p.stem.split("_")[0]),
    )
    runs = []
    for p in files:
        arr = np.loadtxt(p, skiprows=1)
        runs.append(arr)
    return np.stack(runs, axis=0)


def compute_summary(data: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for j, name in enumerate(COLUMNS):
        col = data[:, :, j]
        per_run_range = col.max(axis=1) - col.min(axis=1)
        rows.append({
            "column": name,
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "median_run_range": float(np.median(per_run_range)),
            "p90_run_range": float(np.percentile(per_run_range, 90)),
        })
    return rows


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({
                k: (f"{v:.6g}" if isinstance(v, float) else v)
                for k, v in r.items()
            })


def plot_trajectories(data: np.ndarray, path: Path, zoom_post_t0: bool = False) -> None:
    t = data[0, :, 0]
    cols_to_plot = [j for j, name in enumerate(COLUMNS) if name != "time_d"]
    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True)
    axes = axes.ravel()
    for k, j in enumerate(cols_to_plot):
        ax = axes[k]
        col = data[:, :, j]
        for r in range(col.shape[0]):
            ax.plot(t, col[r], color="C0", alpha=0.03, linewidth=0.5)
        ax.plot(t, np.median(col, axis=0), color="C3", linewidth=1.5)
        ax.set_title(COLUMNS[j], fontsize=10)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        if zoom_post_t0:
            post = col[:, 1:]
            ymin, ymax = float(post.min()), float(post.max())
            span = ymax - ymin
            pad = 0.05 * span if span > 0 else max(abs(ymax), 1e-12) * 0.05
            ax.set_ylim(ymin - pad, ymax + pad)
    for ax in axes[-3:]:
        ax.set_xlabel("time (d)", fontsize=9)
    title = "All 1000 trajectories per column (red = median across runs)"
    if zoom_post_t0:
        title += "\n[y-axis zoomed to post-t=0 range; t=0 spikes go off-axis]"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    EDA_DIR.mkdir(exist_ok=True)
    data = load_all()
    print(f"Loaded data shape: {data.shape}")
    rows = compute_summary(data)
    write_summary(rows, EDA_DIR / "summary.csv")
    plot_trajectories(data, EDA_DIR / "trajectories.png")
    plot_trajectories(data, EDA_DIR / "trajectories_zoomed.png", zoom_post_t0=True)
    print(f"Wrote {(EDA_DIR / 'summary.csv').relative_to(ROOT)}")
    print(f"Wrote {(EDA_DIR / 'trajectories.png').relative_to(ROOT)}")
    print(f"Wrote {(EDA_DIR / 'trajectories_zoomed.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
