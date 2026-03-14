"""
Publication-Quality Visualization Module
=========================================
Creates multi-panel figures for visualizing anaerobic model results
with experimental data overlay.

Best Practices:
- Uses object-oriented matplotlib interface (fig, ax = plt.subplots())
- Uses constrained_layout=True for proper spacing
- Colorblind-friendly palettes (tab10, viridis-based)
- 300 dpi PNG output for publication quality
- Proper axis labels, legends, and grids

Usage:
    from lstm_synth_data.visualize import plot_publication_figure
    from lstm_synth_data.generate import generate_synthetic_data
    from lstm_synth_data.params import get_experimental_data

    data = generate_synthetic_data(n_points=500)
    exp_data = get_experimental_data()
    fig = plot_publication_figure(data, exp_data, save_path="figures/model_results.png")
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from pathlib import Path
from typing import Optional, Dict, Tuple, Union

# -----------------------------------------------------------------------------
# Publication-quality style settings
# -----------------------------------------------------------------------------

# Colorblind-friendly palette (tab10 subset + custom)
COLORS = {
    'model': '#1f77b4',      # Blue
    'exp': '#000000',        # Black
    'methane': '#2ca02c',    # Green
    'sulfate': '#d62728',    # Red
    'acetate': '#9467bd',    # Purple
    'h2s': '#ff7f0e',        # Orange
    'biomass': '#17becf',    # Cyan
    'fe': '#8c564b',         # Brown
    'ph': '#7f7f7f',         # Gray
}

# Figure defaults
FIGURE_DEFAULTS = {
    'figsize': (14, 10),
    'dpi': 300,
    'facecolor': 'white',
    'edgecolor': 'none',
}

# Line and marker styles
LINE_DEFAULTS = {
    'linewidth': 1.5,
    'alpha': 0.9,
}

MARKER_DEFAULTS = {
    'marker': 'o',
    'markersize': 6,
    'markerfacecolor': 'white',
    'markeredgecolor': 'black',
    'markeredgewidth': 1.2,
    'linestyle': 'none',
    'zorder': 10,  # Ensure markers are on top
}


def configure_publication_style() -> None:
    """
    Configure matplotlib rcParams for publication-quality figures.

    Sets font sizes, line widths, and other defaults suitable for
    journal publication.
    """
    plt.rcParams.update({
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,

        # Line settings
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,

        # Grid settings
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,

        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'gray',
        'legend.fancybox': False,

        # Figure settings
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def format_axis(ax: plt.Axes,
                xlabel: str = None,
                ylabel: str = None,
                title: str = None,
                grid: bool = True,
                minor_ticks: bool = True) -> None:
    """
    Apply consistent formatting to an axis.

    Args:
        ax: Matplotlib axes object
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Subplot title
        grid: Whether to show grid
        minor_ticks: Whether to show minor ticks
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight='medium')

    if grid:
        ax.grid(True, which='major', linestyle='-', alpha=0.3)
        if minor_ticks:
            ax.grid(True, which='minor', linestyle=':', alpha=0.15)

    if minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Ensure tick marks point inward
    ax.tick_params(direction='in', which='both', top=True, right=True)


def plot_publication_figure(
    data: Dict,
    exp_data: Dict = None,
    save_path: Union[str, Path] = None,
    show: bool = True,
    title: str = None,
    figsize: Tuple[float, float] = None
) -> plt.Figure:
    """
    Create a publication-quality 3x4 multi-panel figure.

    Layout:
        Row 1: Gas phase species (nH2_g, nCO2_g, nCH4_g, nH2S_g)
               with experimental data overlaid as black circles
        Row 2: Aqueous concentrations (H2_aq, CO2_aq, SO4, S_tot)
        Row 3: Biomass, reaction rates, pH, Fe_pool

    Args:
        data: Dictionary from generate_synthetic_data() containing:
              - time: Time array (days)
              - states: State array (n_points x 14)
              - rates: Reaction rates (n_points x 4)
              - pH: pH values
        exp_data: Dictionary from get_experimental_data() (optional)
        save_path: Path to save figure (PNG at 300 dpi)
        show: Whether to display the figure
        title: Optional figure title (suptitle)
        figsize: Figure size in inches (width, height)

    Returns:
        fig: Matplotlib Figure object
    """
    # Apply publication style
    configure_publication_style()

    # Extract data
    t_sim = data['time']
    y_sim = data['states']
    rates = data['rates']
    pH_sim = data['pH']
    n_points = data.get('n_points', len(t_sim))

    # Create figure with constrained_layout for proper spacing
    if figsize is None:
        figsize = FIGURE_DEFAULTS['figsize']

    fig, axes = plt.subplots(
        3, 4,
        figsize=figsize,
        constrained_layout=True,
        facecolor=FIGURE_DEFAULTS['facecolor']
    )

    # Optional figure title
    if title is None:
        title = f'Basalt @ 25C Model Results ({n_points} points)'
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)

    # =========================================================================
    # Row 1: Gas Phase Species
    # =========================================================================
    gas_config = [
        {'idx': 0, 'label': r'$n_{\mathrm{H_2}}$ (mmol)', 'exp_key': 'nH2_g_exp', 'color': COLORS['model']},
        {'idx': 1, 'label': r'$n_{\mathrm{CO_2}}$ (mmol)', 'exp_key': 'nCO2_g_exp', 'color': COLORS['model']},
        {'idx': 2, 'label': r'$n_{\mathrm{CH_4}}$ (mmol)', 'exp_key': 'nCH4_g_exp', 'color': COLORS['methane']},
        {'idx': 3, 'label': r'$n_{\mathrm{H_2S}}$ (mmol)', 'exp_key': 'nH2S_g_exp', 'color': COLORS['h2s']},
    ]

    for i, cfg in enumerate(gas_config):
        ax = axes[0, i]

        # Plot model line
        ax.plot(t_sim, y_sim[:, cfg['idx']],
                color=cfg['color'],
                label='Model',
                **LINE_DEFAULTS)

        # Overlay experimental data if available
        if exp_data is not None and cfg['exp_key'] in exp_data:
            ax.plot(exp_data['t_exp'], exp_data[cfg['exp_key']],
                    label='Exp. Data',
                    **MARKER_DEFAULTS)

        format_axis(ax, xlabel='Time (days)', ylabel=cfg['label'])

        # Add legend to first panel only (for gas row)
        if i == 0:
            ax.legend(loc='upper right')

    # =========================================================================
    # Row 2: Aqueous Concentrations
    # =========================================================================
    aq_config = [
        {'idx': 4, 'label': r'$[\mathrm{H_2}]_{aq}$ (mM)', 'exp_key': None, 'color': COLORS['model']},
        {'idx': 5, 'label': r'$[\mathrm{CO_2}]_{aq}$ (mM)', 'exp_key': None, 'color': COLORS['model']},
        {'idx': 6, 'label': r'$[\mathrm{SO_4^{2-}}]$ (mM)', 'exp_key': 'SO4_exp', 'color': COLORS['sulfate']},
        {'idx': 11, 'label': r'$S_{tot}$ (mM)', 'exp_key': None, 'color': COLORS['h2s']},
    ]

    for i, cfg in enumerate(aq_config):
        ax = axes[1, i]

        # Plot model line
        ax.plot(t_sim, y_sim[:, cfg['idx']],
                color=cfg['color'],
                label='Model',
                **LINE_DEFAULTS)

        # Overlay experimental data if available
        if exp_data is not None and cfg['exp_key'] is not None and cfg['exp_key'] in exp_data:
            ax.plot(exp_data['t_exp'], exp_data[cfg['exp_key']],
                    label='Exp. Data',
                    **MARKER_DEFAULTS)
            ax.legend(loc='best')

        format_axis(ax, xlabel='Time (days)', ylabel=cfg['label'])

    # =========================================================================
    # Row 3: Biomass, Rates, pH, Fe_pool
    # =========================================================================

    # Panel 3,0: Biomass
    ax = axes[2, 0]
    ax.plot(t_sim, y_sim[:, 8], color=COLORS['biomass'], **LINE_DEFAULTS)
    format_axis(ax, xlabel='Time (days)', ylabel=r'Biomass $X$ (mM)')

    # Panel 3,1: Reaction Rates
    ax = axes[2, 1]
    rate_labels = [
        (0, r'$r_{meth}$', COLORS['methane']),
        (1, r'$r_{sulf}$', COLORS['sulfate']),
        (3, r'$r_{aceto}$', COLORS['acetate']),
    ]
    for idx, label, color in rate_labels:
        ax.plot(t_sim, rates[:, idx], color=color, label=label, **LINE_DEFAULTS)
    ax.legend(loc='upper right', ncol=1)
    format_axis(ax, xlabel='Time (days)', ylabel='Rate (mM/day)')

    # Panel 3,2: pH
    ax = axes[2, 2]
    ax.plot(t_sim, pH_sim, color=COLORS['ph'], **LINE_DEFAULTS, label='Model pH')

    # Overlay experimental pH if available
    if exp_data is not None and 'pH_exp' in exp_data:
        ax.plot(exp_data['t_exp'], exp_data['pH_exp'],
                label='Exp. Data',
                **MARKER_DEFAULTS)
        ax.legend(loc='best')

    format_axis(ax, xlabel='Time (days)', ylabel='pH')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Panel 3,3: Fe_pool
    ax = axes[2, 3]
    ax.plot(t_sim, y_sim[:, 13], color=COLORS['fe'], **LINE_DEFAULTS)
    format_axis(ax, xlabel='Time (days)', ylabel=r'Fe(II) pool (mM)')

    # =========================================================================
    # Save and display
    # =========================================================================
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, facecolor='white', edgecolor='none')
        print(f"Saved publication figure: {save_path}")

    if show:
        plt.show()

    return fig


def plot_gas_phase_detail(
    data: Dict,
    exp_data: Dict = None,
    save_path: Union[str, Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a detailed 2x2 figure focusing on gas phase species.

    Useful for presentations or when gas phase dynamics are the focus.

    Args:
        data: Dictionary from generate_synthetic_data()
        exp_data: Dictionary from get_experimental_data()
        save_path: Path to save figure
        show: Whether to display the figure

    Returns:
        fig: Matplotlib Figure object
    """
    configure_publication_style()

    t_sim = data['time']
    y_sim = data['states']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fig.suptitle('Gas Phase Dynamics', fontsize=12, fontweight='bold')

    configs = [
        {'ax': axes[0, 0], 'idx': 0, 'label': r'$n_{\mathrm{H_2}}$ (mmol)', 'exp_key': 'nH2_g_exp', 'title': 'Hydrogen'},
        {'ax': axes[0, 1], 'idx': 1, 'label': r'$n_{\mathrm{CO_2}}$ (mmol)', 'exp_key': 'nCO2_g_exp', 'title': 'Carbon Dioxide'},
        {'ax': axes[1, 0], 'idx': 2, 'label': r'$n_{\mathrm{CH_4}}$ (mmol)', 'exp_key': 'nCH4_g_exp', 'title': 'Methane'},
        {'ax': axes[1, 1], 'idx': 3, 'label': r'$n_{\mathrm{H_2S}}$ (mmol)', 'exp_key': 'nH2S_g_exp', 'title': 'Hydrogen Sulfide'},
    ]

    for cfg in configs:
        ax = cfg['ax']

        ax.plot(t_sim, y_sim[:, cfg['idx']],
                color=COLORS['model'],
                label='Model',
                **LINE_DEFAULTS)

        if exp_data is not None and cfg['exp_key'] in exp_data:
            ax.plot(exp_data['t_exp'], exp_data[cfg['exp_key']],
                    label='Experimental',
                    **MARKER_DEFAULTS)

        format_axis(ax, xlabel='Time (days)', ylabel=cfg['label'], title=cfg['title'])
        ax.legend(loc='best')

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved gas phase figure: {save_path}")

    if show:
        plt.show()

    return fig


def plot_reaction_rates_detail(
    data: Dict,
    save_path: Union[str, Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a detailed figure showing all reaction rates over time.

    Includes both stacked area and individual line representations.

    Args:
        data: Dictionary from generate_synthetic_data()
        save_path: Path to save figure
        show: Whether to display the figure

    Returns:
        fig: Matplotlib Figure object
    """
    configure_publication_style()

    t_sim = data['time']
    rates = data['rates']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle('Reaction Rate Dynamics', fontsize=12, fontweight='bold')

    rate_names = [r'$r_{meth}$', r'$r_{sulf}$', r'$r_{precip}$', r'$r_{aceto}$']
    rate_colors = [COLORS['methane'], COLORS['sulfate'], COLORS['fe'], COLORS['acetate']]

    # Left panel: Individual lines
    ax = axes[0]
    for i, (name, color) in enumerate(zip(rate_names, rate_colors)):
        ax.plot(t_sim, rates[:, i], color=color, label=name, **LINE_DEFAULTS)

    format_axis(ax, xlabel='Time (days)', ylabel='Rate (mM/day)', title='Individual Rates')
    ax.legend(loc='upper right')

    # Right panel: Stacked area (excluding precipitation for clarity)
    ax = axes[1]
    bio_rates = rates[:, [0, 1, 3]]  # meth, sulf, aceto
    bio_names = [r'$r_{meth}$', r'$r_{sulf}$', r'$r_{aceto}$']
    bio_colors = [COLORS['methane'], COLORS['sulfate'], COLORS['acetate']]

    ax.stackplot(t_sim, bio_rates.T, labels=bio_names, colors=bio_colors, alpha=0.7)
    format_axis(ax, xlabel='Time (days)', ylabel='Rate (mM/day)', title='Stacked Biotic Rates')
    ax.legend(loc='upper right')

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved reaction rates figure: {save_path}")

    if show:
        plt.show()

    return fig


def plot_sulfur_cycle(
    data: Dict,
    exp_data: Dict = None,
    save_path: Union[str, Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a focused figure on sulfur cycling dynamics.

    Shows SO4, S_tot, H2S speciation, and sulfate reduction rate.

    Args:
        data: Dictionary from generate_synthetic_data()
        exp_data: Dictionary from get_experimental_data()
        save_path: Path to save figure
        show: Whether to display the figure

    Returns:
        fig: Matplotlib Figure object
    """
    configure_publication_style()

    t_sim = data['time']
    y_sim = data['states']
    rates = data['rates']
    H2S_aq = data['H2S_aq']
    HS_aq = data['HS_aq']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fig.suptitle('Sulfur Cycle Dynamics', fontsize=12, fontweight='bold')

    # SO4 with experimental data
    ax = axes[0, 0]
    ax.plot(t_sim, y_sim[:, 6], color=COLORS['sulfate'], label='Model', **LINE_DEFAULTS)
    if exp_data is not None and 'SO4_exp' in exp_data:
        ax.plot(exp_data['t_exp'], exp_data['SO4_exp'], label='Exp. Data', **MARKER_DEFAULTS)
    format_axis(ax, xlabel='Time (days)', ylabel=r'$[\mathrm{SO_4^{2-}}]$ (mM)', title='Sulfate')
    ax.legend(loc='best')

    # Total dissolved sulfide
    ax = axes[0, 1]
    ax.plot(t_sim, y_sim[:, 11], color=COLORS['h2s'], **LINE_DEFAULTS)
    format_axis(ax, xlabel='Time (days)', ylabel=r'$S_{tot}$ (mM)', title='Total Dissolved Sulfide')

    # H2S speciation
    ax = axes[1, 0]
    ax.plot(t_sim, H2S_aq, color=COLORS['h2s'], label=r'$\mathrm{H_2S}$', **LINE_DEFAULTS)
    ax.plot(t_sim, HS_aq, color=COLORS['sulfate'], label=r'$\mathrm{HS^-}$', **LINE_DEFAULTS)
    format_axis(ax, xlabel='Time (days)', ylabel='Concentration (mM)', title='Sulfide Speciation')
    ax.legend(loc='best')

    # Sulfate reduction rate
    ax = axes[1, 1]
    ax.plot(t_sim, rates[:, 1], color=COLORS['sulfate'], **LINE_DEFAULTS)
    ax.fill_between(t_sim, 0, rates[:, 1], color=COLORS['sulfate'], alpha=0.2)
    format_axis(ax, xlabel='Time (days)', ylabel=r'$r_{sulf}$ (mM/day)', title='Sulfate Reduction Rate')

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved sulfur cycle figure: {save_path}")

    if show:
        plt.show()

    return fig


def create_all_figures(
    data: Dict,
    exp_data: Dict = None,
    output_dir: Union[str, Path] = "figures",
    prefix: str = "basalt_25c",
    show: bool = False
) -> Dict[str, Path]:
    """
    Generate all publication figures and save to output directory.

    Convenience function to create a complete set of figures for a thesis
    or publication.

    Args:
        data: Dictionary from generate_synthetic_data()
        exp_data: Dictionary from get_experimental_data()
        output_dir: Directory to save figures
        prefix: Filename prefix
        show: Whether to display figures interactively

    Returns:
        Dictionary mapping figure names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_points = data.get('n_points', len(data['time']))

    saved_paths = {}

    # Main multi-panel figure
    path = output_dir / f"{prefix}_overview_{n_points}pts.png"
    plot_publication_figure(data, exp_data, save_path=path, show=show)
    saved_paths['overview'] = path

    # Gas phase detail
    path = output_dir / f"{prefix}_gas_phase_{n_points}pts.png"
    plot_gas_phase_detail(data, exp_data, save_path=path, show=show)
    saved_paths['gas_phase'] = path

    # Reaction rates
    path = output_dir / f"{prefix}_rates_{n_points}pts.png"
    plot_reaction_rates_detail(data, save_path=path, show=show)
    saved_paths['rates'] = path

    # Sulfur cycle
    path = output_dir / f"{prefix}_sulfur_{n_points}pts.png"
    plot_sulfur_cycle(data, exp_data, save_path=path, show=show)
    saved_paths['sulfur'] = path

    print(f"\nCreated {len(saved_paths)} publication figures in: {output_dir}")

    return saved_paths


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main():
    """Command-line interface for generating publication figures."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate publication-quality figures for Basalt @ 25C model'
    )
    parser.add_argument(
        '--n_points', type=int, default=500,
        help='Number of data points to generate (default: 500)'
    )
    parser.add_argument(
        '--output', type=str, default='figures',
        help='Output directory for figures (default: figures)'
    )
    parser.add_argument(
        '--prefix', type=str, default='basalt_25c',
        help='Filename prefix (default: basalt_25c)'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Display figures interactively'
    )
    parser.add_argument(
        '--overview-only', action='store_true',
        help='Generate only the main overview figure'
    )

    args = parser.parse_args()

    # Import here to avoid circular imports
    from .generate import generate_synthetic_data
    from .params import get_experimental_data

    print("Generating synthetic data...")
    data = generate_synthetic_data(n_points=args.n_points, verbose=True)
    exp_data = get_experimental_data()

    if args.overview_only:
        path = Path(args.output) / f"{args.prefix}_overview_{args.n_points}pts.png"
        plot_publication_figure(data, exp_data, save_path=path, show=args.show)
    else:
        create_all_figures(
            data,
            exp_data,
            output_dir=args.output,
            prefix=args.prefix,
            show=args.show
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
