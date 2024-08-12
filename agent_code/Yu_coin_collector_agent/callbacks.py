import numpy as np
import settings as s

import os

import torch
import torch.nn as nn

from policynet import PolicyNet


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODELNAME = 'my-saved-model.pt'

def setup(self):
    np.random.seed()
    self.logger.info('Successfully entered setup code')
    self.model = PolicyNet(len(ACTIONS))
    if self.train:
        self.logger.info('Loading model')
        if not os.path.exists(MODELNAME):
            self.logger.info('No model found')
        else:
            self.model = torch.load(MODELNAME)
            self.logger.info('Model for training loaded')
    else:
        self.model = torch.load(MODELNAME)
        self.model.eval()
        self.model.requires_grad_(False)
        self.logger.info('Model for evaluation loaded')
    


def act(self, game_state):...





def state_to_features(self, game_state) -> np.array:
    # This is a function that converts the game state to the input of your model
    # 1. Situational awareness for the agent: Walls, crates, bombs, other agents
    
    # 2. Pathfinding features
    
    # 3. Life-saving features
    pass
