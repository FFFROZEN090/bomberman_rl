import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
from .policy_utils import BasePolicy

#TODO: Visualize the training process by wandb
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


# Simple feedforward policy
class FFPolicy(BasePolicy):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, n_layers=1, **kwargs):
        super(FFPolicy, self).__init__(feature_dim, action_dim, hidden_dim, **kwargs)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers-1)])
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
        self.init_optimizer()

        # Initialize wandb
        wandb.init(
            project="MLE_Bomberman",
            config={
                "architecture": "FFPolicy",
                "feature_dim": self.feature_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "learning_rate": self.lr,
                "gamma": self.gamma
            }
        )


    def forward(self, features):
        # Add a leaky ReLU activation function
        x = F.relu(self.fc1(features))
        # don't use softmax here, since we want to output real numbers
        for layer in self.fc:
            x = F.relu(layer(x))
        x = self.fc2(x)
        return x

    def train(self):
        loss_values = []
        
        # # check the length of the rewards
        # print('Length of rewards:', len(self.rewards))
        # print('Length of actions:', len(self.actions))
        # print('Length of game_state_history:', len(self.game_state_history))
        # print('Length of action_probs:', len(self.action_probs))
        
        # Calculate discounted rewards
        discounted_rewards = []
        
        R = 0 
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Training loop for each step
        if len(self.rewards) == len(self.actions) == len(self.game_state_history) == len(self.action_probs):
            steps = len(self.rewards)
        else:
            steps = len(self.rewards)-1
            
        for t in range(steps):
            rewards = discounted_rewards[t:]
            features = self.game_state_history[t]
            action_prob = F.softmax(self.forward(features), dim=0)
            log_prob = torch.log(action_prob + 1e-9)[self.actions[t]] # add a small epsilon to avoid log(0)
            policy_loss = -log_prob * rewards
            loss = policy_loss.sum()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_values.append(loss)
        
        self.final_rewards.append(sum(self.rewards))
        self.final_discounted_rewards.append(sum(discounted_rewards))
        self.loss_values.append(sum(loss_values)/len(loss_values))

        # Log metrics to wandb
        wandb.log({
            "episode": self.episode,
            "loss": self.loss_values[-1],
            "reward": self.final_rewards[-1],
            "discounted_reward": self.final_discounted_rewards[-1],
            "score": self.scores[-1]
        })

# LSTM policy
class LSTMPolicy(BasePolicy):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, lstm_layers=1, **kwargs):
        super(LSTMPolicy, self).__init__(feature_dim, action_dim, hidden_dim, **kwargs)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.init_optimizer()

        # Initialize wandb
        wandb.init(
            project="MLE_Bomberman",
            config={
                "architecture": "LSTMPolicy",
                "feature_dim": feature_dim,
                "action_dim": action_dim,
                "hidden_dim": hidden_dim,
                "lstm_layers": lstm_layers,
                "learning_rate": self.lr,
                "gamma": self.gamma
            }
        )


    def forward(self, features, hidden=None):
        x, hidden = self.lstm(features.unsqueeze(0), hidden)
        x = F.softmax(self.fc(x.squeeze(0)), dim=0)
        return x, hidden

    def train(self):
        loss_values = []
        hidden = None
        
        # Calculate discounted rewards
        discounted_rewards = []
        for t in range(len(self.rewards)):
            Gt = sum([r * (self.gamma ** i) for i, r in enumerate(self.rewards[t:])])
            discounted_rewards.append(Gt)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Training loop
        for t in range(len(self.game_state_history)):
            self.optimizer.zero_grad()
            
            features = self.game_state_history[t]
            action_prob, hidden = self.forward(features, hidden)
            
            log_prob = torch.log(action_prob)
            loss = -log_prob * discounted_rewards[t]
            
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            loss_values.append(loss.item())
        
        self.final_rewards.append(sum(self.rewards))
        self.final_discounted_rewards.append(sum(discounted_rewards))
        self.loss_values.append(sum(loss_values) / len(loss_values))

        # Log metrics to wandb
        wandb.log({
            "episode": self.episode,
            "loss": self.loss_values[-1],
            "reward": self.final_rewards[-1],
            "discounted_reward": self.final_discounted_rewards[-1],
            "score": self.scores[-1]
        })
        

# Actor-Critic Proximal policy
class PPOPolicy(BasePolicy):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, clip_epsilon=0.2, **kwargs):
        super(PPOPolicy, self).__init__(feature_dim, action_dim, hidden_dim, **kwargs)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.clip_epsilon = clip_epsilon
        self.init_optimizer()

        # Initialize wandb
        wandb.init(
            project="MLE_Bomberman",
            config={
                "architecture": "PPOPolicy",
                "feature_dim": feature_dim,
                "action_dim": action_dim,
                "hidden_dim": hidden_dim,
                "clip_epsilon": clip_epsilon,
                "learning_rate": self.lr,
                "gamma": self.gamma
            }
        )

    def forward(self, features):
        return F.softmax(self.actor(features), dim=0)

    def evaluate(self, features):
        return self.critic(features)

    def train(self):
        # TODO: Implement PPO-specific training logic here
        pass