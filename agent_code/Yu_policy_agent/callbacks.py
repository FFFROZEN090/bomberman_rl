import numpy as np
import settings as s
from collections import deque

import os

import torch
import torch.nn as nn

from .policy_utils import *
from .policy_model import *
from .config import *

def setup(self):
    np.random.seed()
    self.logger.info('Successfully entered setup code')
    
    # Choose a model architecture and hyperparameters according to the arugments passed to the agent
    if MODEL_TYPE == 'FF':
        self.model = FFPolicy(feature_dim=FEATURE_DIM, action_dim=6, hidden_dim=128, seq_len=1, 
                              n_layers=N_LAYERS, alpha = ALPHA, episode=0, gamma=0.99, model_name=MODEL_NAME, WANDB=WANDB)
    elif MODEL_TYPE == 'SFF':
        self.model = SFFPolicy(feature_dim=FEATURE_DIM, action_dim=6, hidden_dim=128, seq_len=1, 
                               n_layers=N_LAYERS, alpha = ALPHA, episode=0, gamma=0.99, model_name=MODEL_NAME, WANDB=WANDB)
    elif MODEL_TYPE == 'LSTM':
        self.model = LSTMPolicy(feature_dim=FEATURE_DIM, action_dim=6, hidden_dim=128, seq_len=SEQ_LEN, 
                                n_layers=N_LAYERS, alpha = ALPHA, episode=0, gamma=0.99, model_name=MODEL_NAME, WANDB=WANDB)
    elif MODEL_TYPE == 'PPO':
        self.model = PPOPolicy(feature_dim=FEATURE_DIM, action_dim=6, hidden_dim=128, episode=0, gamma=0.99, model_name=MODEL_NAME, WANDB=WANDB)
    
    # Create a game state history for the agent
    # self.opponent_history = deque([], 5) # save the last 5 actions of the opponents
    # self.bomb_history = deque([], 5) # save the last 5 bomb positions
    self.model.state_seqs = deque([], SEQ_LEN)
    
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
    action_probs = self.model.forward(game_state=game_state)
    action_probs = action_probs.detach().numpy()
    # print(action_probs.shape)

    # Behavioral cloning if training and episode < TEACH_EPISODE
    # if self.train and self.model.episode < TEACH_EPISODE:
    #     action, _ = self.teacher.act(game_state)
    #     self.logger.info(f'Behavior cloning action: {action}')
    #     self.logger.info(f'Predicted cloning action probabilities: {action_probs[ACTIONS.index(action)]}')
    # else:
    if np.isnan(action_probs).any():
        self.logger.info('Action probabilities contain NaN values')
        action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[0.23, 0.23, 0.23, 0.23, 0.08])
    else:
        action = np.random.choice(ACTIONS, p=action_probs)
    
    if self.train:
        # record the teacher's action for imitation learning
        teacher_action, _ = self.model.teacher.act(game_state)
        self.model.teacher_action_history.append(teacher_action)
        self.logger.info(f'Teacher action {teacher_action}')
    
    # record the action index
    self.model.action_history.append(ACTIONS.index(action))
        
    return action



