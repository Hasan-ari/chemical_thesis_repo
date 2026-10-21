"""Plot per-rock LORO results for weight decay 0, 1e-4 and 1e-3.

Data: wd_rmse_by_rock.csv, copied from the cell 20 output of
colab_eight_rocks_wd_loro.ipynb (rounded to 3 decimals).
Output: wd_rmse_by_rock.png (same folder).

Run from the repo root:
    env312/bin/python conditional_model_v3_8_rocks_wd/plot_wd_rmse.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "wd_rmse_by_rock.csv"
PNG_PATH = HERE / "wd_rmse_by_rock.png"

# Ordinal blue ramp: light = no weight decay, dark = strongest weight decay.
WD_ORDER = ["0", "1e-4", "1e-3"]
WD_LABELS = {"0": "wd = 0 (v2)", "1e-4": "wd = 1e-4", "1e-3": "wd = 1e-3"}
WD_COLORS = {"0": "#86b6ef", "1e-4": "#2a78d6", "1e-3": "#104281"}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def load() -> pd.DataFrame:
    frame = pd.read_csv(CSV_PATH, dtype={"weight_decay": str})
    rocks = list(dict.fromkeys(frame["held_out_rock"]))  # keep the CSV order
    frame["held_out_rock"] = pd.Categorical(frame["held_out_rock"], rocks, ordered=True)
    return frame


def grouped_bars(ax, frame: pd.DataFrame, column: str) -> None:
    table = frame.pivot(index="held_out_rock", columns="weight_decay", values=column)
    x = np.arange(len(table.index))
    width = 0.26
    for i, wd in enumerate(WD_ORDER):
        ax.bar(
            x + (i - 1) * width,
            table[wd].to_numpy(),
            width=width,
            color=WD_COLORS[wd],
            edgecolor=SURFACE,  # thin gap between adjacent bars
            linewidth=1.5,
            label=WD_LABELS[wd],
            zorder=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in table.index], color=TEXT_SECONDARY)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)


def reference_line(ax, y: float, text: str) -> None:
    ax.axhline(y, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=3)
    ax.annotate(
        text, xy=(1.0, y), xycoords=("axes fraction", "data"),
        xytext=(0, 4), textcoords="offset points",
        ha="right", va="bottom", fontsize=9, color=TEXT_SECONDARY,
    )


def main() -> None:
    frame = load()
    fig, (ax_unseen, ax_seen) = plt.subplots(2, 1, figsize=(11, 9), facecolor=SURFACE)

    # Top panel: unseen rock. Values span 0.09 to 5.5, so use a log scale.
    grouped_bars(ax_unseen, frame, "unseen_rmse_mean_normalized")
    style(ax_unseen)
    ax_unseen.set_yscale("log")
    ax_unseen.set_ylim(0.05, 8)
    ax_unseen.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    reference_line(ax_unseen, 1.0, "1.0 = model that always predicts the mean")
    ax_unseen.set_title(
        "Unseen rock — normalized RMSE, log scale",
        loc="left", color=TEXT_PRIMARY, fontsize=12,
    )
    ax_unseen.set_ylabel("normalized RMSE", color=TEXT_SECONDARY)
    ax_unseen.legend(
        frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3,
        labelcolor=TEXT_PRIMARY,
    )

    # Bottom panel: rocks the same model was trained on. Linear scale.
    grouped_bars(ax_seen, frame, "seen_rmse_mean_normalized")
    style(ax_seen)
    ax_seen.set_ylim(0, 0.22)
    # The line label would collide with the bars, so the title explains the line.
    reference_line(ax_seen, 0.10, "")
    ax_seen.set_title(
        "Seen rocks — normalized RMSE, linear scale  ·  dashed line: 0.10 threshold",
        loc="left", color=TEXT_PRIMARY, fontsize=12,
    )
    ax_seen.set_ylabel("normalized RMSE", color=TEXT_SECONDARY)
    ax_seen.set_xlabel("held-out rock (LORO fold)", color=TEXT_SECONDARY)

    fig.suptitle(
        "Eight-rock LORO: weight decay 0 / 1e-4 / 1e-3",
        x=0.01, ha="left", fontsize=14, color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=200, facecolor=SURFACE)
    print(f"saved: {PNG_PATH}")


if __name__ == "__main__":
    main()
