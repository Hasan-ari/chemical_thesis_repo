from __future__ import annotations

import torch
import torch.nn as nn

from src.data.constants import N_FEATURES


class PhreeqcLSTM(nn.Module):
    """Multi-layer LSTM with linear output head for next-step prediction.

    Architecture:
        LSTM(n_features -> hidden_size, num_layers, batch_first=True)
        -> Linear(hidden_size -> n_features)
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, n_features) — next-step prediction
        """
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
