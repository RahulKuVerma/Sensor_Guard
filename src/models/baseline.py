import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    Clean baseline LSTM for binary early-fault prediction.

    Input:
        X -> (batch_size, sequence_length, sensor_count)

    Default SensorGuard configuration:
        sequence_length = 30
        sensor_count = 21
        hidden_size = 128
        num_layers = 2
        dropout = 0.30

    Output:
        logits -> (batch_size,)
    """

    def __init__(
        self,
        input_size: int = 21,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.30,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError(
                "input_size must be greater than zero."
            )

        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be greater than zero."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0.0, 1.0)."
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # =====================================================
        # LSTM
        # =====================================================

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        # =====================================================
        # Classification head
        # =====================================================

        self.dropout = nn.Dropout(
            p=dropout
        )

        self.classifier = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Tensor of shape:

                (batch_size, sequence_length, input_size)

            For SensorGuard:

                (batch_size, 30, 21)

        Returns
        -------
        torch.Tensor
            Logits of shape:

                (batch_size,)
        """

        if x.ndim != 3:
            raise ValueError(
                "Expected input tensor with 3 dimensions: "
                "(batch_size, sequence_length, input_size). "
                f"Received shape: {tuple(x.shape)}"
            )

        if x.shape[-1] != self.input_size:
            raise ValueError(
                "Input sensor dimension does not match "
                f"model input_size={self.input_size}. "
                f"Received {x.shape[-1]}."
            )

        # -----------------------------------------------------
        # LSTM
        # -----------------------------------------------------

        lstm_output, _ = self.lstm(x)

        # -----------------------------------------------------
        # Take the representation from the final time step.
        #
        # Shape:
        #
        # (batch, sequence_length, hidden_size)
        #                     ↓
        # (batch, hidden_size)
        # -----------------------------------------------------

        last_hidden = lstm_output[:, -1, :]

        # -----------------------------------------------------
        # Dropout
        # -----------------------------------------------------

        features = self.dropout(
            last_hidden
        )

        # -----------------------------------------------------
        # Binary classification logit
        #
        # (batch, hidden_size)
        #          ↓
        # (batch, 1)
        #          ↓
        # (batch,)
        # -----------------------------------------------------

        logits = self.classifier(
            features
        ).squeeze(-1)

        return logits


def create_baseline_model(
    input_size: int = 21,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.30,
) -> LSTMBaseline:
    """
    Factory function for creating the SensorGuard
    clean baseline LSTM.
    """

    return LSTMBaseline(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )