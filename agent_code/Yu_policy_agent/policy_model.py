import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
from .policy_utils import *
from collections import deque

from .rulebased_teacher import TeacherModel
from .config import *


#TODO: Visualize the training process by wandb


# Simple feedforward policy
class FFPolicy(BasePolicy):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, n_layers=1, seq_len=1, alpha = 0.1, **kwargs):
        super(FFPolicy, self).__init__(feature_dim, hidden_dim, n_layers, seq_len, alpha , **kwargs)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers-1)])
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.n_layers = n_layers
        self.seq_len = seq_len
        
        self.alpha = alpha
        
        self.init_optimizer()

    def forward(self, game_state = None, index = None, print_info = False, teacher_acting = False):
        if index is None:
            # record the teacher's action for imitation learning
            teacher_action, _ = self.teacher.act(game_state)
            self.teacher_action_history.append(teacher_action)
            
            game_state_features = self.state_to_features(game_state)
            self.game_state_history.append(game_state_features)
        else:
            game_state_features = self.game_state_history[index]
        
        x = F.relu(self.fc1(game_state_features))
        for fc in self.fc:
            x = F.relu(fc(x))
        x = self.fc2(x)
        
        action_probs = self.getting_action_probs(x)
        if print_info:
            print('The state features are: ', game_state_features, 'at the step ', game_state['step'])
            print('The birth corner is: ', self.birth_corner)
            print('The output of action is: ', x)
            print('The action probabilities are: ', action_probs)
        if teacher_acting:
            teacher_action_idx = ACTIONS.index(teacher_action)
            action_probs = torch.zeros(self.action_dim)
            action_probs[teacher_action_idx] = 1
        return action_probs

# Specific FF policy
class SFFPolicy(BasePolicy):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, n_layers=1, seq_len=1, alpha = 0.1, **kwargs):
        super(SFFPolicy, self).__init__(feature_dim, hidden_dim, n_layers, seq_len, alpha , **kwargs)
        def make_layers(input_dim, hidden_dim, n_layers):
            layers = []
            for i in range(n_layers):
                layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, 1))
            return nn.Sequential(*layers)

        self.feature_dim = feature_dim
        self.move_feature_dim = (feature_dim - 2) // 4
        self.movement_net = make_layers(self.move_feature_dim, hidden_dim, n_layers)
        self.wait_net = make_layers(self.feature_dim, hidden_dim, n_layers)
        self.bomb_net = make_layers(self.feature_dim, hidden_dim, n_layers)
        self.n_layers = n_layers
        self.seq_len = seq_len
        
        self.alpha = alpha
        
        self.init_optimizer()


    def forward(self, game_state = None, index = None, print_info = False):
        if index is None:
            # record the teacher's action for imitation learning
            teacher_action, _ = self.teacher.act(game_state)
            self.teacher_action_history.append(teacher_action)
            
            game_state_features = self.state_to_features(game_state)
            self.game_state_history.append(game_state_features)
        else:
            game_state_features = self.game_state_history[index]
        
        indices = torch.tensor([0, 6, 10, 14, 18, 22, 26])
        up = self.movement_net(torch.index_select(game_state_features, 0, indices))
        right = self.movement_net(torch.index_select(game_state_features, 0, indices+1))
        down = self.movement_net(torch.index_select(game_state_features, 0, indices+2))
        left = self.movement_net(torch.index_select(game_state_features, 0, indices+3))
        # bomb_indices = torch.tensor([])
        wait = self.wait_net(game_state_features)
        bomb = self.bomb_net(game_state_features)
        
        # combine the six actions into one vector
        x = torch.stack((up, right, down, left, wait, bomb), dim=0).squeeze()
        
        action_probs = self.getting_action_probs(x)
        if print_info:
            print('The state features are: ', game_state_features, 'at the step ', game_state['step'])
            print('The birth corner is: ', self.birth_corner)
            print('The output of action is: ', x)
            print('The action probabilities are: ', action_probs)
        return action_probs


# LSTM policy
class LSTMPolicy(BasePolicy):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, seq_len=4, n_layers=1, alpha = 0.1, **kwargs):
        super(LSTMPolicy, self).__init__(feature_dim, hidden_dim, n_layers, seq_len, alpha, **kwargs)
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.state_seqs = deque(maxlen=self.seq_len)
        self.batch_size = 1
        self.alpha = alpha
        
        # Initialize hidden state
        self.hidden = (torch.zeros(self.n_layers, self.batch_size, self.hidden_dim),
                       torch.zeros(self.n_layers, self.batch_size, self.hidden_dim))
        
        self.init_optimizer()
        

    def forward(self, game_state = None, index=None, print_info = False):
        if index is None:
            # record the teacher's action for imitation learning
            teacher_action, _ = self.teacher.act(game_state)
            self.teacher_action_history.append(teacher_action)
            
            game_state_features = self.state_to_features(game_state)
            self.state_seqs.append(game_state_features)
            state_seqs = torch.stack(list(self.state_seqs)).unsqueeze(0)
            self.game_state_history.append(state_seqs)
            
            if len(self.state_seqs) < self.seq_len:
                x = torch.zeros(self.action_dim)
                action_probs = self.getting_action_probs(x)
                return action_probs
                
        else:
            state_seqs = self.game_state_history[index]
            
        # Forward pass through LSTM
        x, self.hidden = self.lstm(state_seqs, self.hidden)
        
        # Detach hidden state to prevent backprop through entire history
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        
        x = self.fc(x[:, -1, :]).squeeze()
        action_probs = self.getting_action_probs(x)
        
        if print_info:
            print('The state features are: ', game_state_features, 'at the step ', game_state['step'])
            print('The birth corner is: ', self.birth_corner)
            print('The output of action is: ', x)
            print('The action probabilities are: ', action_probs)
        return action_probs
        

# TODO: Fix the Actor-Critic Proximal policy
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
        # TODO: Implement PPO-specific training logic here
        pass