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
        x = F.softmax(self.fc2(x), dim=0)
        if torch.isnan(x).any():
            print('Action probabilities contain NaN values')
            print('Features:', features)
            print('Network output:', x)
            x = torch.tensor([0.23, 0.23, 0.23, 0.23, 0.08, 0])
        return x
      
        
    def train(self):
        loss_values = []
        print('The length of game state history is:', len(self.game_state_history))
        print('The length of action probs is:', len(self.action_probs))
        print('The length of rewards is:', len(self.rewards))
        discounted_rewards = []
        for t in range(1, len(self.rewards)):
            loss = 0
            Gt = 0
            pw = 0
            for r in self.rewards[t:]:
                Gt += self.gamma**pw * r
                pw += 1
            
            discounted_rewards.append(Gt)

            #loss += -torch.log(self.action_probs[t]) * Gt
            # update the action probabilities for each step
            for i in range(t):
                self.action_probs[i] = self.forward(self.game_state_history[i])
                # calculate the loss
                log_prob = torch.log(self.action_probs[i])
                loss += -log_prob * Gt
            loss = sum(loss)/len(loss)
            
            # update the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_values.append(loss)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # save the final rewards and discounted rewards
        self.final_rewards.append(sum(self.rewards))
        self.final_discounted_rewards.append(sum(discounted_rewards))
        self.loss_values.append(sum(loss_values)/len(loss_values))
            
            
        
        
      
      