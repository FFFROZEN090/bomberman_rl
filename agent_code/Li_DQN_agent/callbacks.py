"""
This file contains the callback functions for the DQN agent.
"""
import os
import numpy as np
from .DQN_utils import get_state



def setup(self):
    np.random.seed()


def act(agent, game_state: dict):
    agent.logger.info('Pick action at random, but no bombs.')
    # Get the state representation
    state = get_state(game_state)

    np.save('state.npy', state)
    
    # Convert state to tensor if needed for your model
    # Note: You might need to import torch at the top of the file
    # state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    
    # TODO: Use the state with your DQN model to choose an action
    # For now, we'll continue with the random choice
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'])
