from __future__ import annotations

import torch
import torch.nn as nn


class ConditionTimeLSTM(nn.Module):
    """Many-to-many LSTM: condition+time sequence -> full output trajectory."""

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _state = self.lstm(x)
        return self.head(sequence)
