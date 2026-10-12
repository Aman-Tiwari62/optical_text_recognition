from __future__ import annotations

import torch
import torch.nn as nn


class CNNBiLSTMCTC(nn.Module):
    """A simple CNN + BiLSTM model for word-level handwritten text recognition.

    Input tensor shape: (batch_size, 1, image_height, image_width)
    CNN output shape: (batch_size, channels, 1, time_steps)
    LSTM input shape: (time_steps, batch_size, channels)
    Logits shape: (time_steps, batch_size, vocabulary_size)
    """

    def __init__(
        self,
        image_height: int = 32,
        cnn_channels: tuple[int, ...] = (32, 64, 128),
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        vocab_size: int = 80,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.image_height = image_height
        self.cnn_channels = tuple(cnn_channels)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.vocab_size = vocab_size

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in self.cnn_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            input_size=self.cnn_channels[-1],
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if rnn_num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(rnn_hidden_size * 2, vocab_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw logits compatible with torch.nn.CTCLoss.

        We do not apply softmax here: CTC loss expects logit scores before the final
        activation. This keeps the implementation numerically stable and correct.
        """

        if images.dim() != 4:
            raise ValueError(f"Expected 4D images (B, C, H, W), got {tuple(images.shape)}")

        features = self.cnn(images)
        features = self.pool(features)  # (B, C, 1, T)
        features = features.squeeze(2)  # (B, C, T)
        sequence = features.permute(2, 0, 1)  # (T, B, C)
        recurrent_output, _ = self.rnn(sequence)
        logits = self.classifier(recurrent_output)  # (T, B, vocab_size)
        return logits

    def compute_output_lengths(self, input_widths: list[int] | torch.Tensor) -> torch.Tensor:
        """Return per-image time steps after the CNN's 2x pooling operations."""
        if torch.is_tensor(input_widths):
            widths = input_widths.to(dtype=torch.long)
        else:
            widths = torch.tensor(input_widths, dtype=torch.long)
        stride = 2 ** len(self.cnn_channels)
        return torch.div(widths + stride - 1, stride, rounding_mode="floor")
