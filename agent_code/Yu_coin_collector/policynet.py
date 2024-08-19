import torch
import torch.nn as nn
import torch.nn.functional as F
import os

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# General policy class
class BasePolicy(nn.Module):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, episode=0,
                 gamma=0.99, epsilon=0.1, model_name=''):
        super(BasePolicy, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Episode information
        self.game_state_history = []
        self.action_probs = []
        self.rewards = []
        
        # parameters for saving and loading the model
        self.checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
        self.name = model_name
        self.episode = episode
        self.gamma = gamma
        self.epsilon = epsilon
        self.loss_values = []
        self.final_rewards = []
        self.final_discounted_rewards = []
        self.scores = []
        
    def init_optimizer(self, lr=0.01):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, features):
        raise NotImplementedError("Subclasses should implement this!")

    def save(self):
        checkpoint = {
            'episode': self.episode,
            'model_name': self.name,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'loss_values': self.loss_values,
            'final_rewards': self.final_rewards,
            'final_discounted_rewards': self.final_discounted_rewards,
            'scores': self.scores,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        torch.save(checkpoint, os.path.join("Saves/", self.checkpoint_dir, f"{self.name}_{self.episode}.pt"))
    
    def load(self, path = None):
        if path is None:
            print('No checkpoint path is given.')
        elif os.path.isfile(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episode = checkpoint['episode']
            self.name = checkpoint['model_name']
            self.gamma = checkpoint['gamma']
            self.epsilon = checkpoint['epsilon']
            self.loss_values = checkpoint['loss_values']
            self.final_rewards = checkpoint['final_rewards']
            self.final_discounted_rewards = checkpoint['final_discounted_rewards']
            self.scores = checkpoint['scores']
            print('Model loaded from', path)
        else:
            print('No model found at', path)

    def train(self):
        raise NotImplementedError("Subclasses should implement this!")

# Simple feedforward policy
class FFPolicy(BasePolicy):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, **kwargs):
        super(FFPolicy, self).__init__(feature_dim, action_dim, hidden_dim, **kwargs)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.init_optimizer()


    def forward(self, features):
        x = F.relu(self.fc1(features))
        # don't use softmax here, since we want to output real numbers
        x = self.fc2(x)
        return x

    def train(self):
        loss_values = []
        
        # Calculate discounted rewards
        discounted_rewards = []
        for t in range(1, len(self.rewards)):
            loss = 0
            Gt = 0
            pw = 0
            for r in self.rewards[t:]:
                Gt += self.gamma**pw * r
                pw += 1
            
            discounted_rewards.append(Gt)

            for i in range(t):
                self.action_probs[i] = self.forward(self.game_state_history[i])
                log_prob = torch.log(self.action_probs[i])
                loss += -log_prob * Gt
            loss = sum(loss)/len(loss)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_values.append(loss)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        self.final_rewards.append(sum(self.rewards))
        self.final_discounted_rewards.append(sum(discounted_rewards))
        self.loss_values.append(sum(loss_values)/len(loss_values))

# LSTM policy
class LSTMPolicy(BasePolicy):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, lstm_layers=1, **kwargs):
        super(LSTMPolicy, self).__init__(feature_dim, action_dim, hidden_dim, **kwargs)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.init_optimizer()


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


    def forward(self, features):
        return F.softmax(self.actor(features), dim=0)

    def evaluate(self, features):
        return self.critic(features)

    def train(self):
        # Implement PPO-specific training logic here
        pass