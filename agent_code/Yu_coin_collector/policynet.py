import torch.nn as nn
import torch.nn.functional as F

import torch

import os

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
class Policy(nn.Module):
    def __init__(self, feature_dim, action_dim = len(ACTIONS), hidden_dim=128, episode=0, 
                 gamma=0.99, epsilon = 0.1, model_name: str = ''):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        
        # Episode information
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
        
        torch.save(checkpoint, os.path.join("Saves/",self.checkpoint_dir, self.name + '_', str(self.episode) + '.pt'))
    
    def load(self, path):
        if path is None:
            print('No checkpoint path is given.')
        elif os.path.isfile(path) == True:
            self.load_state_dict(torch.load(path))
            print('Model loaded from', path)
        else:
            print('No model found at', path)
        
    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = F.softmax(self.fc2(x), dim=1)
        self.action_probs.append(x)
        return x
      
    
    def loss(self):
        discounted_rewards = []
        for t in range(len(self.rewards)):
            Gt = 0
            pw = 0
            for r in self.rewards[t:]:
                Gt += self.gamma**pw * r
                pw += 1
            discounted_rewards.append(Gt)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        loss = 0
        for log_prob, Gt in zip(self.action_probs, discounted_rewards):
            loss += -log_prob * Gt
        self.loss_values.append(loss)
        return loss
        
    def train(self):
        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        self.optimizer.step()
        self.rewards = []
        self.action_probs = []
        return loss.item()
        
        
      
      