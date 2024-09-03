"""
DQN network

"""

from collections import deque
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
            nn.Linear(512, output_size),
            nn.ReLU()
        )

        # Exploration probability
        self.exploration_prob = 1.0

        # Decay rate
        self.decay_rate = 0.995

        # Learning rate
        self.learning_rate = 0.001

        self.output_size = output_size

        self.input_channels = input_channels

        self.gamma = 0.99

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.epoch = 0

    def forward(self, x):
        # Permute the dimensions from [batch, height, width, channels] to [batch, channels, height, width]
        x = x.permute(0, 3, 1, 2)
        # Forward pass through the sequential model
        return self.model(x)


    def action(self, state):
        if np.random.rand() < self.exploration_prob:
            # Update the exploration probability
            self.exploration_prob *= self.decay_rate
            return np.random.randint(self.output_size)
        else:
            # Print state shape
        
            with torch.no_grad():
                input = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.forward(input)
            return torch.argmax(q_values).item()


    def sample_experiences(self, replay_buffer, experience_buffer, batch_size):
        # Calculate the number of samples from each source
        replay_samples = int(0.4 * batch_size)
        experience_samples = batch_size - replay_samples

        # Sample from replay buffer
        replay_experiences = random.sample(replay_buffer.buffer, min(replay_samples, len(replay_buffer.buffer)))

        # Sample from experiences dataset
        dataset_experiences = experience_buffer.sample(min(experience_samples, len(experience_buffer)))

        # Combine and shuffle the samples
        experiences = replay_experiences + dataset_experiences
        random.shuffle(experiences)

        return experiences
    
    def convert_experiences_to_tensors(self, experiences, device):
        # Separate the batch into numpy arrays
        states = np.array([exp.global_state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.global_next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences]).astype(bool)

        # Convert numpy arrays to PyTorch tensors and move to the specified device
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        return states, actions, rewards, next_states, dones

    def train(self, replay_buffer, experience_buffer, batch_size, device='cpu'):
        # Sample experiences
        experiences = self.sample_experiences(replay_buffer, experience_buffer, batch_size)
        
        # Convert experiences to tensors using the new member function
        states, actions, rewards, next_states, dones = self.convert_experiences_to_tensors(experiences, device)

        # Compute Q-values for current states
        current_q_values = self(states).gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-9


        # Compute Q-values for next states
        with torch.no_grad():
            next_q_values = self(next_states).max(1)[0]

        target_q_values = rewards + (1 - dones.float()/batch_size) * self.gamma * next_q_values
        # For done states, the target Q-value is equal to the reward
        target_q_values[dones] = rewards[dones]

        # Normalize the target Q-values and  current Q-values
        target_q_values = (target_q_values - target_q_values.mean()) / target_q_values.std()
        current_q_values = (current_q_values - current_q_values.mean()) / current_q_values.std()

        # Print sum of target Q-values and current Q-values
        print(f"sum of rewards: {rewards.sum()}")
        print(f"sum of target Q-values: {target_q_values.sum()}")
        print(f"sum of current Q-values: {current_q_values.sum()}")

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f"loss: {loss.item()}")

        # Restrict the exploration probability
        self.exploration_prob = max(self.exploration_prob * self.decay_rate, 0.1)

        # If after 100 epochs, save the model
        if self.epoch % 100 == 0:
            if not os.path.exists(os.path.join(os.path.dirname(__file__), 'checkpoints')):
                os.mkdir(os.path.join(os.path.dirname(__file__), 'checkpoints'))
            self.save(os.path.join(os.path.dirname(__file__), 'checkpoints', 'Li_DQN_agent' + '_' + str(self.epoch) + '.' + 'pt'))

        return loss.item()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def init_parameters(self):
        for param in self.parameters():
            if len(param.size()) == 2:  # Linear layer
                torch.nn.init.xavier_uniform_(param)
            elif len(param.size()) == 4:  # Convolution layer
                torch.nn.init.kaiming_uniform_(param)

class ExperienceDataset:
    def __init__(self, max_size=100000,load_from_npy=False, npy_file_path=None):
        self.experiences = deque(maxlen=max_size)
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

    def add(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)    

# TODO: Design Proper Loss Function
class DQNLoss(nn.Module):
    def __init__(self, gamma=0.99):
        super(DQNLoss, self).__init__()
        self.gamma = gamma

    def forward(self, q_values, target_q_values, actions, rewards, dones):
        # Get Q-values for the actions taken
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute the target Q-values
        with torch.no_grad():
            # Compute max Q-value for next state
            next_q_values = target_q_values.max(1)[0]
            # Compute the expected Q-values
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute the loss using Huber loss (smooth L1 loss)
        loss = F.smooth_l1_loss(q_values, expected_q_values)

        return loss

def compute_td_loss(model, target_model, experiences, gamma=0.99, device='cpu'):
    states = torch.FloatTensor([exp.agent_state for exp in experiences]).to(device)
    actions = torch.LongTensor([exp.action for exp in experiences]).to(device)
    rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(device)
    next_states = torch.FloatTensor([exp.agent_next_state for exp in experiences]).to(device)
    dones = torch.FloatTensor([exp.done for exp in experiences]).to(device)

    # Compute Q(s, a) - the model computes Q(s), then we select the columns of actions taken
    q_values = model(states)
    
    # Compute Q(s', a') - the target network computes Q(s'), then we select the best action
    with torch.no_grad():
        next_q_values = target_model(next_states)

    # Compute the loss
    loss_fn = DQNLoss(gamma)
    loss = loss_fn(q_values, next_q_values, actions, rewards, dones)

    return loss


    


if __name__ == "__main__":
    model = DQN(14, 6)
    summary(model, (14, 17, 17))