"""
LSTM Synthetic Data Generator - Main Entry Point
=================================================
Generates synthetic data for LSTM training using the Basalt @ 25C
two-phase anaerobic model with best-fit parameters.

Usage:
    uv run main.py                    # Generate 500 points (default)
    uv run main.py --n_points 1000    # Generate 1000 points
    uv run main.py --plot             # Generate and show plots
"""

import sys
from pathlib import Path

# Add src to path for module imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lstm_synth_data.generate import generate_synthetic_data, save_data, plot_results, main as cli_main
from lstm_synth_data.params import get_experimental_data


def main():
    """Run data generation with default settings or CLI args."""
    # If no arguments, run with defaults
    if len(sys.argv) == 1:
        print("=" * 60)
        print("LSTM Synthetic Data Generator - Basalt @ 25C")
        print("=" * 60)

        # Generate 500 points
        data = generate_synthetic_data(
            n_points=500,
            t_start=0.0,
            t_end=19.0,
            verbose=True
        )

        # Save to default output directory
        output_dir = Path(__file__).parent / "data" / "output"
        save_data(data, output_dir)

        print("\n" + "=" * 60)
        print("Data generation complete!")
        print(f"Output saved to: {output_dir}")
        print("=" * 60)

    else:
        # Use CLI parser
        cli_main()


if __name__ == "__main__":
    main()
