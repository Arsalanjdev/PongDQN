"""
The architecture of DQN network tasked with learning to play Atari Ping-Pong.
It consists of a CNN followed by a fully connected layer of Dense layers.
"""

from typing import List
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_shape: tuple, action_size: int):
        super().__init__()

        # Verify input shape format (C, H, W)
        if len(input_shape) != 3:
            raise ValueError(f"Input shape should be (C,H,W), got {input_shape}")

        self.input_shape = input_shape
        self.action_size = action_size

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy)
            fc_input_size = conv_out.size(1)
            # print(f"Network initialized with input shape {input_shape}")
            # print(f"Flattened conv output size: {fc_input_size}")

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validation
        if x.dtype == torch.uint8:
            x = x.float() / 255.0  # Normalize byte tensor to [0,1]

        if x.shape[1:] != self.input_shape:
            raise ValueError(
                f"Expected input shape (N,{','.join(map(str, self.input_shape))}), "
                f"got {tuple(x.shape)}"
            )

        # Forward pass
        features = self.conv(x)
        return self.fc(features)
