import numpy as np
import settings as s
from collections import deque

import os

import torch
import torch.nn as nn

from .policynet import Policy


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODELNAME = 'my-saved-model.pt'

def setup(self):
    np.random.seed()
    self.logger.info('Successfully entered setup code')
    self.model = Policy(len(ACTIONS))
    
    self.opponent_history = deque([], 5) # save the last 5 actions of the opponents
    self.bomb_history = deque([], 5) # save the last 5 bomb positions
    
    
    if self.train:
        self.logger.info('Loading model')
        self.model.load(MODELNAME)
        self.logger.info('Model for training loaded')
    else:
        self.model.load(MODELNAME)
        self.model.eval()
        self.model.requires_grad_(False)
        self.logger.info('Model for evaluation loaded')
    


def act(self, game_state):
    if self.train and np.random.rand() < self.model.epsilon:
        # explore step: act like a random agent
        action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
    else:
        # exploit step: act like a trained agent
        game_state_features = state_to_features(game_state)
        action_probs = self.model.forward(game_state_features)
        action_probs = action_probs.detach().numpy()
        action = np.random.choice(ACTIONS, p=action_probs)
        
    return action


def state_to_features(game_state) -> np.array:
    # This is a function that converts the game state to the input of your model
    
    # General information
    self_pos = game_state['self'][3]
    free = game_state['field'] == 0
    crates = game_state['field'] == 1
    bombs = game_state['bombs']
    bombs_time = np.ones(game_state['field'].shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bombs_time.shape[0]) and (0 < j < bombs_time.shape[1]):
                bombs_time[i, j] = min(bombs_time[i, j], t)
    
    # 1. Situational awareness for the agent: Walls, crates, bombs, other agents, bomb_left
    # Features 1.1. determine which corner the agent is in: left top, right top, left bottom, right bottom
    
    if game_state['step'] == 0:
        if self_pos == (1,1): # left top
            self_corner = 0
        elif self_pos == (1,s.ROWS-2): # right top
            self_corner = 1
        elif self_pos == (s.COLS-2,1): # left bottom
            self_corner = 2
        elif self_pos == (s.COLS-2,s.ROWS-2): # right bottom
            self_corner = 3
    
    # Features 1.2. determine the direction feasibility
    candidate_position = [(self_pos[0], self_pos[1]-1), (self_pos[0], self_pos[1]+1), 
                          (self_pos[0]-1, self_pos[1]), (self_pos[0]+1, self_pos[1]), 
                          self_pos]
    valid_position = []
    for pos in candidate_position:
        if ((free[pos] and crates[pos] == 0) and 
                (game_state['explosion_map'][pos] < 1) and 
                (bombs_time[pos] > 0) and
                (pos not in game_state['others']) and 
                (pos not in game_state['bombs'])):
            valid_position.append(pos)
    up_feasible = [self_pos[0], self_pos[1]-1] in valid_position
    down_feasible = [self_pos[0], self_pos[1]+1] in valid_position
    left_feasible = [self_pos[0]-1, self_pos[1]] in valid_position
    right_feasible = [self_pos[0]+1, self_pos[1]] in valid_position
    wait_feasible = self_pos in valid_position
    
    # Features 1.3. bomb left
    bomb_left = game_state['self'][2] > 0 and self_pos not in self.bomb_history
    
    
    # 2. Pathfinding features: coins nearby, crates nearby
    # Features 2.1. determine the distance to the nearest coin
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
    
    # Features 2.2. determine the distance to the nearest crate
    up_crates_score = 0
    down_crates_score = 0
    left_crates_score = 0
    right_crates_score = 0
    for (xc, yc) in crates:
        if yc > self_pos[1]:
            up_crates_score += 1/(yc-self_pos[1])
        elif yc < self_pos[1]:
            down_crates_score += 1/(self_pos[1]-yc)
        if xc > self_pos[0]:
            right_crates_score += 1/(xc-self_pos[0])
        elif xc < self_pos[0]:
            left_crates_score += 1/(self_pos[0]-xc)
        
    
    # 3. Life-saving features: bomb nearby, explosion nearby
    # Features 3.1. determine the distance to the nearest bomb
    
    for (xb, yb), t in bombs:
        if xb == self_pos[0] and abs(yb-self_pos[1]) < 4:
            up_bomb_nearby = yb >= self_pos[1]
            down_bomb_nearby = yb <= self_pos[1]
        if yb == self_pos[1] and abs(xb-self_pos[0]) < 4:
            left_bomb_nearby = xb >= self_pos[0]
            right_bomb_nearby = xb <= self_pos[0]
    
    # merge all features
    features = np.array([up_feasible, down_feasible, left_feasible, right_feasible, wait_feasible, bomb_left,
                         up_coins_score, down_coins_score, left_coins_score, right_coins_score,
                         up_crates_score, down_crates_score, left_crates_score, right_crates_score,
                         up_bomb_nearby, down_bomb_nearby, left_bomb_nearby, right_bomb_nearby])
    return features
