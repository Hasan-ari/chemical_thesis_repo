from __future__ import annotations

from pathlib import Path

# Raw file columns (13 total, index 0 is time_d)
RAW_COLUMN_NAMES: list = [
    "time_d", "pH", "Ptot_atm", "pH2_atm", "pCO2_atm", "pCH4_atm",
    "CH4_g_mol", "H2_g_mol", "CO2_g_mol", "SO4", "Formate", "Acetate", "Ca",
]

# Feature columns (12 — everything except time_d)
FEATURE_NAMES: list = RAW_COLUMN_NAMES[1:]
N_FEATURES: int = 12

# Columns requiring log1p transform (indices in FEATURE space, 0-based)
# These start at zero or span multiple orders of magnitude:
# pH2_atm(2), pCH4_atm(4), CH4_g_mol(5), H2_g_mol(6), CO2_g_mol(7), Formate(9), Acetate(10)
LOG_TRANSFORM_COLS: tuple = (2, 4, 5, 6, 7, 9, 10)

# Default data paths
DATA_DIR: Path = Path("data/phreeqc_v23")
N_TIMESTEPS: int = 101
N_TRAJECTORIES: int = 1000
