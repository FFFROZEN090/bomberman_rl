"""
DQN network

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchsummary import summary
import random

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

        # Initialize the Q-table
        self.experience = list()
        self.max_experience = 200000

        # Exploration probability
        self.exploration_prob = 0.1

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
    
    def store_experience(self, experience):
        if len(self.experience) < self.max_experience:
            self.experience.append(experience)
        else:
            self.experience.pop(0)
            self.experience.append(experience)


    def action(self, state):
        if np.random.rand() < self.exploration_prob:
            return np.random.randint(self.output_size)
        else:
            return self.forward(state)


    def train(self, batch_size):
        if len(self.experience) < batch_size:
            return
        batch = random.sample(self.experience, batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.tensor(state, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.int64)
            reward = torch.tensor(reward, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)

            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)

            state_action_values = self.forward(state)
            next_state_action_values = self.forward(next_state)
            
            # TODO: Implement the target
            target = reward + (1 - done)

            

    


if __name__ == "__main__":
    model = DQN(24, 6)
    summary(model, (24, 17, 17))