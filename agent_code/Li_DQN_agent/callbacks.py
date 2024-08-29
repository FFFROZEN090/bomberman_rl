"""
This file contains the callback functions for the DQN agent.
"""
import os
import numpy as np
from .DQN_utils import get_state
from .DQN_network import DQN, ExperienceDataset, ReplayBuffer


EXPERIENCE_BUFFER_SIZE = 1000000
REPLAY_BUFFER_SIZE = 800

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', MODEL_NAME + '_' + str(LAST_EPISODE) + '.pt')

def setup(self):
    np.random.seed()
    self.logger.info('Successfully entered setup code')

    # Setup the model
    self.model = DQN(input_channels=14, output_size=6)

    # Store the last state and action
    self.last_state = None
    self.last_action = None
    self.last_reward = None

    # Store last game state
    self.last_game_state = None

    # Initialize the experience buffer
    self.experience_buffer = ExperienceDataset(max_size=EXPERIENCE_BUFFER_SIZE)

    # Initialize the replay buffer
    self.replay_buffer = ReplayBuffer(max_size=REPLAY_BUFFER_SIZE)

    if self.train:
        self.model.load(MODEL_PATH)
        self.logger.info('Model for training loaded')
    else:
        self.model.load(MODEL_PATH)
        self.model.eval()
        self.model.requires_grad_(False)
        self.logger.info('Model for evaluation loaded')


def act(agent, game_state: dict):
    agent.logger.info('Choosing action based on current state.')
    # Get the current state representation
    current_state = get_state(game_state)

    # Use the model to choose an action
    action = agent.model.action(current_state)

    # Convert the action index to the corresponding action string
    action_string = ACTIONS[action]

    return action_string
