"""
LSTM Synthetic Data Generation Module
=====================================
Generates synthetic data for LSTM training using the two-phase anaerobic model
with best-fit parameters from Muller 2024 experiments.

Basalt @ 25C configuration.

Modules:
- generate: Synthetic data generation from ODE model
- lstm_trainer: Production LSTM training pipeline
"""

__version__ = "0.1.0"

from .generate import generate_synthetic_data, save_data
from .params import (
    load_basalt_25c_params,
    get_experimental_data,
    get_default_params,
    BASALT_25C_DATA,
)
from .visualize import (
    plot_publication_figure,
    plot_gas_phase_detail,
    plot_reaction_rates_detail,
    plot_sulfur_cycle,
    create_all_figures,
    configure_publication_style,
)
from .lstm_trainer import (
    TrainingConfig,
    DataProcessor,
    StackedLSTM,
    LSTMTrainer,
    RecursiveForecaster,
    ModelEvaluator,
    run_training_pipeline,
    STATE_NAMES,
)

__all__ = [
    # Data generation
    'generate_synthetic_data',
    'save_data',
    # Parameters
    'load_basalt_25c_params',
    'get_experimental_data',
    'get_default_params',
    'BASALT_25C_DATA',
    # Visualization
    'plot_publication_figure',
    'plot_gas_phase_detail',
    'plot_reaction_rates_detail',
    'plot_sulfur_cycle',
    'create_all_figures',
    'configure_publication_style',
    # LSTM Training
    'TrainingConfig',
    'DataProcessor',
    'StackedLSTM',
    'LSTMTrainer',
    'RecursiveForecaster',
    'ModelEvaluator',
    'run_training_pipeline',
    'STATE_NAMES',
]
