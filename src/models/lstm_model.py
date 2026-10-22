
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size=18,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        output, _ = self.lstm(x)

        # Use the final time step's hidden representation
        last_output = output[:, -1, :]

        # Predict one RUL value per sequence
        prediction = self.regressor(last_output)

        return prediction.squeeze(-1)
