"""
Utility script to visualize MATLAB vs. PyTorch parameter optimization results.

Reads the `matlab_parameter_summary.csv` and `pytorch_parameter_summary.csv`
files generated during the fitting stage, aligns parameters, computes deltas,
and produces a side-by-side comparison plot plus a merged CSV.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


DATA_DIR = Path(__file__).parent
MATLAB_CSV = DATA_DIR / "matlab_parameter_summary.csv"
PYTORCH_CSV = DATA_DIR / "pytorch_parameter_summary.csv"
MERGED_CSV = DATA_DIR / "parameter_comparison.csv"
MATLAB_PLOT = DATA_DIR / "matlab_parameters.png"
PYTORCH_PLOT = DATA_DIR / "pytorch_parameters.png"
COMPARISON_PLOT = DATA_DIR / "parameter_comparison.png"
DELTA_PLOT = DATA_DIR / "parameter_differences.png"
AXIS_FORMATTER = FormatStrFormatter('%.5f')
LABEL_FORMAT = "{:.5f}"
PARAMETER_UNITS = {
    "k_meth": "1/day",
    "k_sulf": "1/day",
    "k_aceto": "1/day",
    "Y_m": "mmol biomass/mmol substrate",
    "Y_s": "mmol biomass/mmol substrate",
    "Y_a": "mmol biomass/mmol substrate",
    "KI_meth": "mmol/L",
    "KI_sulf": "mmol/L",
    "KI_aceto": "mmol/L",
    "k_precip": "1/day",
    "H2S_sat": "mmol/L",
    "H2_thresh": "mmol/L",
    "DG_thresh": "kJ/mol",
}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names so MATLAB/PyTorch summaries align."""
    df = df.rename(columns=str.lower)
    canonical = {
        "lowerbound": "lower_bound",
        "initialguess": "initial_guess",
        "fittedvalue": "fitted_value",
        "upperbound": "upper_bound",
        "deltafrominitial": "delta_from_initial",
        "fractionalchange": "fractional_change",
    }
    df = df.rename(columns=canonical)
    return df


def load_and_align() -> pd.DataFrame:
    """Load both CSV files and align them on parameter names."""
    df_mat = _canonicalize_columns(pd.read_csv(MATLAB_CSV))
    df_py = _canonicalize_columns(pd.read_csv(PYTORCH_CSV))
    df_mat["source"] = "MATLAB"
    df_py["source"] = "PyTorch"

    # In MATLAB CSV the column is "parameter"; in PyTorch it's also "parameter".
    # Use a consistent key for merging.
    merged = df_mat.merge(
        df_py,
        on="parameter",
        how="inner",
        suffixes=("_matlab", "_pytorch"),
    )
    merged = merged.sort_values("parameter").reset_index(drop=True)

    # Helpful derived metrics
    merged["fitted_difference"] = merged["fitted_value_pytorch"] - merged["fitted_value_matlab"]
    merged["fractional_change_difference"] = (
        merged["fractional_change_pytorch"] - merged["fractional_change_matlab"]
    )
    return merged


def _format_xaxis(axis, linthresh: float = 1e-3):
    axis.xaxis.set_major_formatter(AXIS_FORMATTER)
    axis.set_xscale('symlog', linthresh=linthresh)
    axis.grid(axis="x", alpha=0.35, linestyle="--")
    axis.axvline(0, color="black", linewidth=0.5, linestyle="--")
    axis.margins(x=0.15)


def _annotate_barh(ax, bars, offset_pts: float = 15.0):
    """Attach formatted value labels to each horizontal bar."""
    for rect in bars:
        width = rect.get_width()
        y = rect.get_y() + rect.get_height() / 2
        text = LABEL_FORMAT.format(width)
        if width >= 0:
            ax.annotate(
                text,
                xy=(width, y),
                xytext=(offset_pts, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
            )
        else:
            ax.annotate(
                text,
                xy=(width, y),
                xytext=(-offset_pts, 0),
                textcoords="offset points",
                va="center",
                ha="right",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
            )


def plot_single_source(df: pd.DataFrame, source: str, path: Path) -> None:
    """Plot initial vs fitted values for a single implementation."""
    params = df["parameter"]
    y = range(len(params))
    bar_height = 0.45

    fig, ax = plt.subplots(figsize=(11, 7))

    offset = bar_height / 2
    bars_init = ax.barh(
        [yi - offset for yi in y],
        df["initial_guess"],
        height=bar_height * 0.8,
        color="#90c2e7",
        edgecolor="black",
        linewidth=0.4,
        label="Initial guess",
    )
    bars_fit = ax.barh(
        [yi + offset for yi in y],
        df["fitted_value"],
        height=bar_height * 0.8,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.4,
        label="Fitted value",
    )

    ax.set_xlabel("Parameter value (model units, symlog)")
    ax.set_title(f"{source} parameter initialization vs fitted result")
    ax.set_yticks(list(y))
    labels_with_units = [f"{p} [{PARAMETER_UNITS.get(p, 'unitless')}]" for p in params]
    ax.set_yticklabels(labels_with_units)
    _format_xaxis(ax, linthresh=1e-3)
    _annotate_barh(ax, bars_init)
    _annotate_barh(ax, bars_fit)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0, frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved {source} plot to {path}")


def plot_comparison(df: pd.DataFrame) -> None:
    """Create grouped horizontal bar chart for fitted values plus difference plot."""
    params = df["parameter"]
    n = len(params)
    bar_height = 0.35

    y_pos = [i - bar_height / 2 for i in range(n)]
    y_pos_offset = [i + bar_height / 2 for i in range(n)]

    fig, ax = plt.subplots(figsize=(13, 8))

    bars_mat = ax.barh(
        y_pos,
        df["fitted_value_matlab"],
        height=bar_height,
        label="MATLAB",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.4,
    )
    bars_py = ax.barh(
        y_pos_offset,
        df["fitted_value_pytorch"],
        height=bar_height,
        label="PyTorch",
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Fitted value (model units, symlog)")
    ax.set_title("Parameter fits: MATLAB vs. PyTorch")
    ax.set_yticks(range(n))
    labels_with_units = [f"{p} [{PARAMETER_UNITS.get(p, 'unitless')}]" for p in params]
    ax.set_yticklabels(labels_with_units)
    _format_xaxis(ax, linthresh=1e-3)
    _annotate_barh(ax, bars_mat)
    _annotate_barh(ax, bars_py)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0, frameon=False)

    fig.tight_layout()
    fig.savefig(COMPARISON_PLOT, dpi=200, bbox_inches="tight")
    print(f"✓ Saved comparison plot to {COMPARISON_PLOT}")

    # Difference-only horizontal bar plot
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    bars_diff = ax2.barh(range(n), df["fitted_difference"], height=0.4, color="#2ca02c", edgecolor="black", linewidth=0.4)
    ax2.set_xlabel("Difference (PyTorch − MATLAB, symlog)")
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels_with_units)
    _format_xaxis(ax2, linthresh=0.1)
    _annotate_barh(ax2, bars_diff)

    fig2.tight_layout()
    fig2.savefig(DELTA_PLOT, dpi=200, bbox_inches="tight")
    print(f"✓ Saved difference plot to {DELTA_PLOT}")


def main() -> None:
    df = load_and_align()
    df.to_csv(MERGED_CSV, index=False)
    print(f"✓ Exported merged table to {MERGED_CSV}")

    df_mat = df[
        [
            "parameter",
            "lower_bound_matlab",
            "initial_guess_matlab",
            "fitted_value_matlab",
            "fractional_change_matlab",
        ]
    ].rename(
        columns={
            "lower_bound_matlab": "lower_bound",
            "initial_guess_matlab": "initial_guess",
            "fitted_value_matlab": "fitted_value",
            "fractional_change_matlab": "fractional_change",
        }
    )
    df_py = df[
        [
            "parameter",
            "lower_bound_pytorch",
            "initial_guess_pytorch",
            "fitted_value_pytorch",
            "fractional_change_pytorch",
        ]
    ].rename(
        columns={
            "lower_bound_pytorch": "lower_bound",
            "initial_guess_pytorch": "initial_guess",
            "fitted_value_pytorch": "fitted_value",
            "fractional_change_pytorch": "fractional_change",
        }
    )

    plot_single_source(df_mat, "MATLAB", MATLAB_PLOT)
    plot_single_source(df_py, "PyTorch", PYTORCH_PLOT)
    plot_comparison(df)


if __name__ == "__main__":
    main()

