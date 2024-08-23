"""
DQN network

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchsummary import summary

class DQN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(DQN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=2)  # (17x17x24 -> 7x7x32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)              # (7x7x32 -> 5x5x64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)             # (5x5x64 -> 3x3x128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        # Apply the convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from the conv layers
        x = x.view(x.size(0), -1)  # Flatten

        # Apply the fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer for Q-values

        return x

if __name__ == "__main__":
    model = DQN(24, 6)
    summary(model, (24, 17, 17))