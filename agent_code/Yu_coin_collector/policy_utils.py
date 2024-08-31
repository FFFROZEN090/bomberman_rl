import numpy as np
import settings as s


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
from .rulebased_teacher import TeacherModel
#TODO: Visualize the training process by wandb
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# General policy class
class BasePolicy(nn.Module):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, n_layers=1, alpha = 0.1, episode=0,
                 gamma=0.99, model_name='', lr=0.001, WANDB=False):
        super(BasePolicy, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.WANDB = WANDB
        
        # Episode information
        self.game_state_history = []
        self.action_probs = []
        self.actions = []
        self.rewards = []
        self.state_seqs = []
        self.hidden = None
        
        # parameters for saving and loading the model
        self.checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
        self.name = model_name
        self.episode = episode
        self.gamma = gamma
        self.lr = lr
        # self.epsilon = epsilon # in the policy-based methods, epsilon is not used
        self.loss_values = []
        self.final_rewards = []
        self.final_discounted_rewards = []
        self.scores = []
        
        # teacher model for imitation learning
        self.teacher = TeacherModel()
        self.teacher_action = []
        self.alpha = alpha # weight for the imitation loss
        
        
    def init_optimizer(self, lr=0.001):
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)
    
    def forward(self, features):
        raise NotImplementedError("Subclasses should implement this!")

    def save(self):
        checkpoint = {
            'episode': self.episode,
            'model_name': self.name,
            'gamma': self.gamma,
            'lr': self.lr,
            'n_layers': self.n_layers,
            # 'epsilon': self.epsilon,
            'loss_values': self.loss_values,
            'final_rewards': self.final_rewards,
            'final_discounted_rewards': self.final_discounted_rewards,
            'scores': self.scores,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        torch.save(checkpoint, os.path.join("Saves/", self.checkpoint_dir, f"{self.name}_{self.episode}.pt"))

        # Upload the model to wandb
        # wandb.save(os.path.join("Saves/", self.checkpoint_dir, f"{self.name}_{self.episode}.pt"))
    
    
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
            self.n_layers = checkpoint['n_layers']
            self.lr = checkpoint['lr']
            self.loss_values = checkpoint['loss_values']
            self.final_rewards = checkpoint['final_rewards']
            self.final_discounted_rewards = checkpoint['final_discounted_rewards']
            self.scores = checkpoint['scores']
            print('Model loaded from', path)
        else:
            print('No model found at', path)

    def train(self):
        raise NotImplementedError("Subclasses should implement this!")
    
    # TODO: visualizations
    def visualize(self):
        pass
    
    def reset(self):
        self.hidden = None
        self.state_seqs.clear()
        self.game_state_history.clear()
        self.teacher_action.clear()
        self.actions.clear()
        self.rewards.clear()
        self.action_probs.clear()
      

def state_to_features(game_state: dict) -> torch.Tensor:
    # This is a function that converts the game state to the input of your model
    
    # General information
    self_pos = game_state['self'][3]
    arena = game_state['field']
    free = arena == 0
    bombs = game_state['bombs']
    others = game_state['others']
    bombs_time = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bombs_time.shape[0]) and (0 < j < bombs_time.shape[1]):
                bombs_time[i, j] = min(bombs_time[i, j], t)
    
    # 1. Situational awareness for the agent: Walls, crates, bombs, other agents, bomb_left
    # Features 1.1. determine which corner the agent is in: left top, right top, left bottom, right bottom
    
    # TODO: How to make use of the corner information 
    # if game_state['step'] == 0:
    #     if self_pos == (1,1): # left top
    #         self_corner = 0
    #     elif self_pos == (1,s.ROWS-2): # right top
    #         self_corner = 1
    #     elif self_pos == (s.COLS-2,1): # left bottom
    #         self_corner = 2
    #     elif self_pos == (s.COLS-2,s.ROWS-2): # right bottom
    #         self_corner = 3
    
    # Features 1.2. determine the direction feasibility
    candidate_position = [(self_pos[0], self_pos[1]-1), (self_pos[0], self_pos[1]+1), 
                          (self_pos[0]-1, self_pos[1]), (self_pos[0]+1, self_pos[1]), 
                          self_pos]
    valid_position = []
    for pos in candidate_position:
        if ((free[pos] == 0) and 
                (game_state['explosion_map'][pos] < 1) and 
                (bombs_time[pos] > 0) and
                (not any(pos == other_pos for other_pos in others)) and 
                (not any(pos == bomb for bomb, t in bombs))):
            valid_position.append(pos)
    up_feasible = [self_pos[0], self_pos[1]-1] in valid_position
    down_feasible = [self_pos[0], self_pos[1]+1] in valid_position
    left_feasible = [self_pos[0]-1, self_pos[1]] in valid_position
    right_feasible = [self_pos[0]+1, self_pos[1]] in valid_position
    wait_feasible = self_pos in valid_position
    
    # Features 1.3. bomb left
    bomb_left = game_state['self'][2] > 0 
    
    
    # 2. Pathfinding features: coins nearby, crates nearby
    # Features 2.1. determine the distance to the coins
    coins = game_state['coins']
    up_coins_score = 0
    down_coins_score = 0
    left_coins_score = 0
    right_coins_score = 0
    for (xc, yc) in coins:
        if yc > self_pos[1]:
            up_coins_score += 1/(yc-self_pos[1])
        elif yc < self_pos[1]:
            down_coins_score += 1/(self_pos[1]-yc)
        if xc > self_pos[0]:
            right_coins_score += 1/(xc-self_pos[0])
        elif xc < self_pos[0]:
            left_coins_score += 1/(self_pos[0]-xc)
    
    # Features 2.2. determine the distance to the crates
    crates_list = [(i, j) for i in range(s.COLS) for j in range(s.ROWS) if arena[i, j] == 1]
    up_crates_score = 0
    down_crates_score = 0
    left_crates_score = 0
    right_crates_score = 0
    # TODO: How to encode the crates information?
    for (xc, yc) in crates_list:
        if yc > self_pos[1]:
            up_crates_score += 1/(yc-self_pos[1])
        elif yc < self_pos[1]:
            down_crates_score += 1/(self_pos[1]-yc)
        if xc > self_pos[0]:
            right_crates_score += 1/(xc-self_pos[0])
        elif xc < self_pos[0]:
            left_crates_score += 1/(self_pos[0]-xc)
    
    # Features 2.3. determine the distance to the deadend
    dead_ends = [(i, j) for i in range(s.COLS) for j in range(s.ROWS) if (arena[i, j] == 0)
                 and ([arena[i + 1, j], arena[i - 1, j], arena[i, j + 1], arena[i, j - 1]].count(0) == 1)]
    up_dead_ends_score = 0
    down_dead_ends_score = 0
    left_dead_ends_score = 0
    right_dead_ends_score = 0
    for (xd, yd) in dead_ends:
        if yd > self_pos[1]:
            up_dead_ends_score += 1/(yd-self_pos[1])
        elif yd < self_pos[1]:
            down_dead_ends_score += 1/(self_pos[1]-yd)
        if xd > self_pos[0]:
            right_dead_ends_score += 1/(xd-self_pos[0])
        elif xd < self_pos[0]:
            left_dead_ends_score += 1/(self_pos[0]-xd)
    
    # 3. Life-saving features: bomb nearby, explosion nearby
    # Features 3.1. determine the distance to the nearest bomb
    up_bomb_nearby = 0
    down_bomb_nearby = 0
    left_bomb_nearby = 0
    right_bomb_nearby = 0
    for (xb, yb), t in bombs:
        if xb == self_pos[0] and abs(yb-self_pos[1]) < 4:
            # if the bomb is in the same column and no wall in between the bomb and the agent
            up_bomb_nearby = 1 if yb >= self_pos[1] and arena[xb, yb-1] != -1 else 0
            down_bomb_nearby = 1 if yb <= self_pos[1] and arena[xb, yb+1] != -1 else 0
        if yb == self_pos[1] and abs(xb-self_pos[0]) < 4:
            left_bomb_nearby = 1 if xb >= self_pos[0] and arena[xb-1, yb] != -1 else 0
            right_bomb_nearby = 1 if xb <= self_pos[0] and arena[xb+1, yb] != -1 else 0
    
    # merge all features
    features = np.array([up_feasible, down_feasible, left_feasible, right_feasible, wait_feasible, bomb_left,
                         up_coins_score, down_coins_score, left_coins_score, right_coins_score,
                         up_crates_score, down_crates_score, left_crates_score, right_crates_score,
                         up_dead_ends_score, down_dead_ends_score, left_dead_ends_score, right_dead_ends_score,
                         up_bomb_nearby, down_bomb_nearby, left_bomb_nearby, right_bomb_nearby])
    features = torch.tensor(features, dtype=torch.float32)
    return features