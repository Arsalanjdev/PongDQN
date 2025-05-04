"""
The architecture of DQN network tasked with learning to play Atari Ping-Pong.
It consists of a CNN followed by a fully connected layer of Dense layers.
"""
from typing import List
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input: List, action_size:int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, input: torch.ByteTensor) -> torch.Tensor:
        input = input / 255.0 #since the inout is a byte tensor
        return self.conv(input)

