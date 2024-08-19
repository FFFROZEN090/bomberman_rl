import numpy as np
import settings as s
from collections import deque

import os

import torch
import torch.nn as nn

from .policynet import *


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_NAME = 'coin1'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', MODEL_NAME + '.pt')

def setup(self):
    np.random.seed()
    self.logger.info('Successfully entered setup code')
    self.model = FFPolicy(feature_dim=22, action_dim=6, hidden_dim=128, episode=0, gamma=0.99, epsilon=0.1, model_name='coin1')
    
    # Create a game state history for the agent
    # self.opponent_history = deque([], 5) # save the last 5 actions of the opponents
    # self.bomb_history = deque([], 5) # save the last 5 bomb positions
    
    
    if self.train:
        self.logger.info('Loading model')
        self.model.load(MODEL_PATH)
        self.logger.info('Model for training loaded')
    else:
        self.model.load(MODEL_PATH)
        self.model.eval()
        self.model.requires_grad_(False)
        self.logger.info('Model for evaluation loaded')
    


def act(self, game_state) -> str:
    # This is the main function that the agent calls to get an action
    game_state_features = state_to_features(game_state)
    self.model.game_state_history.append(game_state_features)
    
    action_probs = nn.functional.softmax(self.model.forward(game_state_features), dim=0)
    self.model.action_probs.append(action_probs)
    
    if self.train and np.random.rand() < self.model.epsilon:
        # explore step: act like a random agent
        action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
    else:
        # exploit step: act like a trained agent
        action_probs = action_probs.detach().numpy()
        action = np.random.choice(ACTIONS, p=action_probs)
        
    return action


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
