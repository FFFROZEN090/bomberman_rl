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
from .DQN_datatype import Experience


class DQN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(DQN, self).__init__()
        # Sequential model
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2),  # (17x17x24 -> 7x7x32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),              # (7x7x32 -> 5x5x64)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),             # (5x5x64 -> 3x3x128)
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output for fully connected layers
            # Fully connected layers
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        # Exploration probability
        self.exploration_prob = 0.9

        # Decay rate
        self.decay_rate = 0.995

        # Learning rate
        self.learning_rate = 0.001

    def forward(self, x):
        # Forward pass through the sequential model
        return self.model(x)


    def action(self, state):
        if np.random.rand() < self.exploration_prob:
            # Update the exploration probability
            self.exploration_prob *= self.decay_rate
            return np.random.randint(self.output_size)
        else:
            with torch.no_grad():
                q_values = self.forward(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            return torch.argmax(q_values).item()


    def train(self, replay_buffer, batch_size，device='cpu'):
        # Sample a batch of experiences from the replay buffer
        experiences = replay_buffer.sample(batch_size)
        
        # Separate the batch into numpy arrays
        states = np.array([exp.global_state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.global_next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])

        # Convert numpy arrays to PyTorch tensors and move to the specified device
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        # Compute Q-values for current states
        current_q_values = self(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Q-values for next states
        with torch.no_grad():
            next_q_values = self(next_states).max(1)[0]

        # Compute target Q-values
        target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update exploration probability
        self.exploration_prob = max(self.exploration_prob * self.decay_rate, 0.1)

        return loss.item()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class ExperienceDataset:
    def __init__(self, max_size=100000,load_from_npy=False, npy_file_path=None):
        self.experiences = []
        self.max_size = max_size
        if load_from_npy and npy_file_path:
            self.load_from_npy(npy_file_path)

    def add(self, experience: Experience):
        if len(self.experiences) >= self.max_size:
            self.experiences.pop(0)
        self.experiences.append(experience)

    def sample(self, batch_size):
        return random.sample(self.experiences, min(batch_size, len(self.experiences)))

    def __len__(self):
        return len(self.experiences)

    def save_to_npy(self, file_path):
        data = {
            'global_states': np.array([exp.global_state for exp in self.experiences]),
            'agent_states': np.array([exp.agent_state for exp in self.experiences]),
            'actions': np.array([exp.action for exp in self.experiences]),
            'rewards': np.array([exp.reward for exp in self.experiences]),
            'global_next_states': np.array([exp.global_next_state for exp in self.experiences]),
            'agent_next_states': np.array([exp.agent_next_state for exp in self.experiences]),
            'dones': np.array([exp.done for exp in self.experiences])
        }
        np.save(file_path, data)

    def load_from_npy(self, file_path):
        data = np.load(file_path, allow_pickle=True).item()
        self.experiences = []
        for i in range(len(data['actions'])):
            exp = Experience(
                global_state=data['global_states'][i],
                agent_state=data['agent_states'][i],
                action=data['actions'][i],
                reward=data['rewards'][i],
                global_next_state=data['global_next_states'][i],
                agent_next_state=data['agent_next_states'][i],
                done=data['dones'][i]
            )
            self.experiences.append(exp)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)    


if __name__ == "__main__":
    model = DQN(24, 6)
    summary(model, (24, 17, 17))