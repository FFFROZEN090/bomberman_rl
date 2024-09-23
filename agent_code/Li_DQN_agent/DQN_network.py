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
import wandb
import logging
from typing import List

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB = True

class DQN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(DQN, self).__init__()
        # Sequential model
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=5, stride=2), 
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048*1*1, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

        # Exploration probability
        self.exploration_prob = 1.0

        # Decay rate
        self.decay_rate = 0.9995

        # Learning rate
        self.learning_rate = 0.001

        self.output_size = output_size

        self.input_channels = input_channels

        self.gamma = 0.99

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.epoch = 0

        self.wandb = WANDB

        self.scores = []

        self.survived_rounds = []
        self.rewards_list = []
        self.action_buffer = []
        self.action_buffer_size = 8
        self.best_score = 0

        # Initialize wandb
        if self.wandb:
            wandb.init(
                project="bomberman_rl",
                config={
                    "model": 'Li_DQN_agent',
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "n_layers": 3,
                    "seq_len": 3,
                    "hidden_dim": 512,
                    "loss_fn": "Huber Loss",
                }
            )

            # Log model architecture
            wandb.watch

    def forward(self, x):
        return self.model(x)


    def action(self, state, low_state, last_action_invalid=False, last_action=None, rotate=0, device=DEVICE):
        if last_action is not None:
            last_action_index = ACTIONS.index(last_action)
        else:
            last_action_index = None

        if np.random.rand() < self.exploration_prob:
            # Update the exploration probability
            self.exploration_prob *= self.decay_rate
            # If last action is invalid, choose a random action excluding the invalid one
            if last_action_invalid and last_action_index is not None:
                valid_actions = [i for i in range(len(ACTIONS)) if i != last_action_index]
                action_code = np.random.choice(valid_actions)
            else:
                action_code = np.random.choice(len(ACTIONS))
            update_action_buffer(self.action_buffer, ACTIONS[action_code], self.action_buffer_size)
            return action_code, "exploration"
        else:
            with torch.no_grad():
                input = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                self.to(device)
                q_values = self.forward(input)

                if last_action_invalid and last_action_index is not None and last_action_index in [0, 1, 2, 3, 4, 5]:
                    # Mask the invalid action
                    q_values[0, last_action_index] = float('-inf')

                    # Identify movement actions (indices 0 to 3) excluding the invalid one
                    movement_actions_indices = [0, 1, 2, 3, 4, 5]
                    if last_action_index != 5:
                        movement_actions_indices.remove(last_action_index)

                    # Calculate opposite action index
                    if last_action_index in [0, 1, 2, 3]:
                        opposite_action_index = (last_action_index + 2) % 4  # Opposite direction
                    else:
                        opposite_action_index = random.choice([0, 1, 2, 3])  # Random movement action

                    # Prepare list of valid movement actions and their indices
                    valid_movement_indices = []
                    for idx in movement_actions_indices:
                        action = ACTIONS[idx]
                        if is_action_valid(low_state, action):
                            valid_movement_indices.append(idx)
                        else:
                            q_values[0, idx] = float('-inf')

                    # Check if opposite action is valid
                    if opposite_action_index in valid_movement_indices:
                        # Assign probabilities
                        probabilities = []
                        total_prob = 0.0
                        for idx in valid_movement_indices:
                            if idx == opposite_action_index:
                                prob = 0.4
                            else:
                                prob = 0.2
                            probabilities.append(prob)
                            total_prob += prob

                        # Normalize probabilities
                        probabilities = [p / total_prob for p in probabilities]

                        # Randomly select an action based on the probabilities
                        selected_idx = np.random.choice(valid_movement_indices, p=probabilities)
                        action_code = selected_idx
                    else:
                        # If opposite action is not valid, redistribute probabilities
                        num_valid = len(valid_movement_indices)
                        if num_valid > 0:
                            probabilities = [1.0 / num_valid] * num_valid
                            selected_idx = np.random.choice(valid_movement_indices, p=probabilities)
                            action_code = selected_idx
                        else:
                            # No valid movement actions; proceed to mask invalid actions and select the best action
                            for i, action in enumerate(ACTIONS):
                                if not is_action_valid(low_state, action):
                                    q_values[0, i] = float('-inf')
                            action_code = torch.argmax(q_values).item()
                else:
                    # Mask all invalid actions
                    for i, action in enumerate(ACTIONS):
                        if not is_action_valid(low_state, action):
                            q_values[0, i] = float('-inf')
                    action_code = torch.argmax(q_values).item()

                update_action_buffer(self.action_buffer, ACTIONS[action_code], self.action_buffer_size)
                return action_code, "network"


    def sample_experiences(self, replay_buffer, experience_buffer, batch_size):
        # Calculate the number of samples from each source
        replay_samples = int(0.9 * batch_size)
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
        agent_states = np.array([exp.agent_state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.global_next_state for exp in experiences])
        next_agent_states = np.array([exp.agent_next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences]).astype(bool)
        last_actions = np.array([exp.last_action for exp in experiences])
        last_actions_invalid = np.array([exp.last_action_invalid for exp in experiences])
        # Convert last actions to valid indices
        last_actions = [ACTIONS.index(action) if action is not None else None for action in last_actions]

        # Normalize states and next_states
        states = self.normalize_states(states)
        next_states = self.normalize_states(next_states)

        # Convert numpy arrays to PyTorch tensors and move to the specified device
        states = torch.FloatTensor(states).to(device)
        
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        last_actions = torch.LongTensor(last_actions).to(device)
        last_actions_invalid = torch.BoolTensor(last_actions_invalid).to(device)

        return states, agent_states, actions, rewards, next_states, next_agent_states, dones, last_actions, last_actions_invalid

    def normalize_states(self, states):
        # Implement normalization logic here
        # Example using min-max scaling
        min_val = states.min(axis=0)
        max_val = states.max(axis=0)
        states = (states - min_val) / (max_val - min_val + 1e-8)
        return states


    def dqn_train(self, replay_buffer, experience_buffer, batch_size, target_model=None, device=DEVICE):
        self.epoch += 1
        # Sample experiences
        experiences = self.sample_experiences(replay_buffer, experience_buffer, batch_size)
        
        # Unpack and convert experiences to tensors
        states, agent_states, actions, rewards, next_states, next_agent_states, dones, last_actions, last_actions_invalid = self.convert_experiences_to_tensors(experiences, device)

        # Convert model weights to device
        self.to(device)
        # Define a large negative finite value for masking
        large_negative_value = -1e5

        # Compute Q-values for current states
        current_q_values_full = self(states)  # Get Q-values for all actions

        # Mask invalid actions in current Q-values
        for i in range(states.size(0)):
            invalid_actions = get_invalid_actions(agent_states[i])
            current_q_values_full[i, invalid_actions] = large_negative_value

        # Now select the Q-values corresponding to the actions taken
        current_q_values = current_q_values_full.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-9

        # Compute Q-values for next states
        with torch.no_grad():
            if target_model is None:
                next_q_values_full = self(next_states)
            else:
                target_model.to(device)
                next_q_values_full = target_model(next_states)
            
            # Mask invalid actions in next Q-values
            for i in range(next_states.size(0)):
                invalid_actions = get_invalid_actions(next_agent_states[i])
                next_q_values_full[i, invalid_actions] = large_negative_value
            
            # Incorporate opposite action preference
            opposite_action_bonus = 1.0  # Adjust as needed
            for i in range(next_states.size(0)):
                if last_actions_invalid[i]:
                    last_action = last_actions[i]
                    if last_action in [0, 1, 2, 3]:  # Movement actions
                        opposite_action = (last_action + 2) % 4
                        # Ensure the opposite action is valid
                        if opposite_action not in get_invalid_actions(next_agent_states[i]):
                            next_q_values_full[i, opposite_action] += opposite_action_bonus

            # Select the maximum Q-value among valid actions
            next_q_values = next_q_values_full.max(1)[0] + 1e-9

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.HuberLoss()(current_q_values, target_q_values)

        # Optionally, add penalty for invalid Q-values
        invalid_q_values = []
        
        for i in range(states.size(0)):
            invalid_actions = get_invalid_actions(agent_states[i])
            if len(invalid_actions) > 0:
                invalid_qs = current_q_values_full[i, invalid_actions]
                invalid_q_values.append(invalid_qs)
        if invalid_q_values:
            invalid_q_values = torch.cat(invalid_q_values)
            invalid_q_loss = torch.mean(invalid_q_values ** 2)
            invalid_q_loss_weight = 0.1  # Adjust as needed
            loss += invalid_q_loss_weight * invalid_q_loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.wandb:
            wandb.log({
                "sum of rewards": rewards.sum(),
                "sum of target Q-values": target_q_values.sum(),
                "sum of current Q-values": current_q_values.sum(),
                "sum of next Q-values": next_q_values.sum(),
                "loss": loss.item(),
                "exploration_prob": self.exploration_prob,
                "score": sum(self.scores) / len(self.scores),
                "survived_rounds": sum(self.survived_rounds) / len(self.survived_rounds)
            })
        self.rewards_list.append(rewards.sum().to('cpu').detach().numpy())


        if self.epoch % 50 == 0:
            if target_model is not None:
                # Copy Parameters from the model to the target model
                target_model.load_state_dict(self.state_dict())


        
        if self.epoch % 50 == 0 and self.epoch != 0:
            if is_downstream_trend(self.rewards_list):
                # Uptune exploration probability by 0.1
                self.exploration_prob = min(self.exploration_prob + 0.05, 0.15)
                # Reset rewards
                self.rewards_list = []


        # If after 100 epochs, save the model
        if self.epoch % 50 == 0:
            if not os.path.exists(os.path.join(os.path.dirname(__file__), 'checkpoints')):
                os.mkdir(os.path.join(os.path.dirname(__file__), 'checkpoints'))
            self.save(os.path.join(os.path.dirname(__file__), 'checkpoints', 'Li_DQN_agent' + '_' + str(self.epoch) + '.' + 'pt'))

            # Upload the model to wandb
            if self.wandb:
                wandb.save(os.path.join(os.path.dirname(__file__), 'checkpoints', 'Li_DQN_agent' + '_' + str(self.epoch) + '.' + 'pt'))

                # Clear scores and survived rounds
        self.scores = []
        self.survived_rounds = []
        # Restrict the exploration probability
        self.exploration_prob = max(self.exploration_prob * self.decay_rate, 0.1)

        return loss.item()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class ExperienceDataset:
    def __init__(self, max_size=1000000,load_from_npy=False, npy_file_path=None):
        self.experiences = deque(maxlen=max_size)
        self.max_size = max_size
        if load_from_npy and npy_file_path:
            self.load_from_npy(npy_file_path)

    def add(self, experience: Experience):
        if len(self.experiences) >= self.max_size:
            self.experiences.pop()
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
    
def update_action_buffer(action_buffer: List, action: str, action_buffer_size: int) -> List:
    action_buffer.append(action)
    if len(action_buffer) > action_buffer_size:
        action_buffer.pop(0)
    return action_buffer

def repeat_action(action_buffer: List, action_buffer_size: int) -> int:
    if len(action_buffer) >= action_buffer_size:
        # UP Repeat 5 times in a 10 steps window
        if action_buffer.count('UP') >= action_buffer_size // 2 - 1:
            # Pop 3 elements from the action buffer
            for _ in range(3):
                action_buffer.pop(0)
            return 0, action_buffer.copy()
        elif action_buffer.count('RIGHT') >= action_buffer_size // 2 - 1:
            for _ in range(3):
                action_buffer.pop(0)
            return 1, action_buffer.copy()
        elif action_buffer.count('DOWN') >= action_buffer_size // 2 - 1:
            for _ in range(3):
                action_buffer.pop(0)
            return 2, action_buffer.copy()
        elif action_buffer.count('LEFT') >= action_buffer_size // 2 - 1:
            for _ in range(3):
                action_buffer.pop(0)
            return 3, action_buffer.copy()
        elif action_buffer.count('WAIT') >= action_buffer_size // 2 - 3:
            for _ in range(2):
                action_buffer.pop(0)
            return 4, action_buffer.copy()
        elif action_buffer.count('BOMB') >= action_buffer_size // 2 - 3:
            for _ in range(2):
                action_buffer.pop(0)
            return 5, action_buffer.copy()
    return -1, action_buffer.copy()


def compute_td_loss(model, target_model, experiences, gamma=0.99, device=DEVICE):
    states = torch.FloatTensor([exp.global_state for exp in experiences]).to(device)
    actions = torch.LongTensor([exp.action for exp in experiences]).to(device)
    rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(device)
    next_states = torch.FloatTensor([exp.global_next_state for exp in experiences]).to(device)
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


def is_downstream_trend(data: list) -> bool:
    """
    Check if the data (list of PyTorch tensors) follows a general downstream (decreasing) trend.
    
    :param data: List of PyTorch tensors (could be on GPU)
    :return: True if the list follows a general downstream trend, False otherwise
    """
    
    # Edge case: If the data has less than 2 points, we consider it as a downstream trend
    if len(data) <= 1:
        return True

    # Create an array of indices for x (time or position)
    x = np.arange(len(data))
    
    # Perform a linear regression (fit a line to the data)
    slope, intercept = np.polyfit(x, data, 1)
    
    # Check if the slope is negative, indicating a downward trend
    return slope < 0

"""
Inverse the action from raw state to the mirrored state
"""
def inverse_rotate_action(rotated_action, angle):
    # If action is WAIT or BOMB, return the same action
    if rotated_action == 4 or rotated_action == 5:
        return rotated_action
    if angle == 90:
        return (rotated_action - 1) % 4
    elif angle == 180:
        return (rotated_action - 2) % 4
    elif angle == 270:
        return (rotated_action - 3) % 4
    else:
        return rotated_action
    
def is_action_valid(state, action):
    x, y = np.where(state[0, :, :] == 1)  # Agent's position
    x, y = x[0], y[0]
    if action == 'UP' and is_cell_free(state, x, y - 1):
        return True
    elif action == 'DOWN' and is_cell_free(state, x, y + 1):
        return True
    elif action == 'LEFT' and is_cell_free(state, x - 1, y):
        return True
    elif action == 'RIGHT' and is_cell_free(state, x + 1, y):
        return True
    elif action == 'WAIT' and state[10, x, y] == 1:
        return True
    elif action == 'BOMB' and state[10, x, y] == 1:
        return True
    return False

def get_invalid_actions(state):
    invalid_actions = []
    for i, action in enumerate(ACTIONS):
        if not is_action_valid(state, action):
            invalid_actions.append(i)
    return invalid_actions


def is_cell_free(state, x, y):
    # Check boundaries
    if x < 0 or x >= 17 or y < 0 or y >= 17:
        return False
    # Check for walls, boxes, or other obstacles
    if state[4, x, y] == 1 or state[5, x, y] == 1 or state[8, x, y] >= 1 or state[7, :, :] == 1:
        return False
    # You can add more checks if needed (e.g., other players)
    return True


if __name__ == "__main__":
    model = DQN(24, 6)
    summary(model, (24, 17, 17))