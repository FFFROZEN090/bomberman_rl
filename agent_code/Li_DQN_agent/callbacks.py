"""
This file contains the callback functions for the DQN agent.
"""
import os
import numpy as np
from .DQN_utils import get_state, get_low_level_state, get_high_level_state
from .DQN_network import DQN, ExperienceDataset, ReplayBuffer
import logging


EXPERIENCE_BUFFER_SIZE = 5000000
REPLAY_BUFFER_SIZE = 5000

MODEL_NAME = 'Li_DQN_agent'
LAST_EPISODE = 10750

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', MODEL_NAME + '_' + str(LAST_EPISODE) + '.pt')

def setup(self):
    np.random.seed()
    self.logger.info('Successfully entered setup code')

    # Setup the model
    self.model = DQN(input_channels=14, output_size=6)
    self.model.epoch = LAST_EPISODE
    self.target_model = DQN(input_channels=14, output_size=6)

    # Store the last state and action
    self.last_state = None
    self.last_action = None
    self.last_reward = None

    # Store last game state
    self.last_game_state = None

    # Initialize the experience buffer
    self.experience_buffer = ExperienceDataset(EXPERIENCE_BUFFER_SIZE)

    # Initialize the replay buffer
    self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    self.batch_size = REPLAY_BUFFER_SIZE

    self.max_rounds = 20000

    self.surving_rounds = 0

    self.spwan_position = None
    self.rotate = 0

    # Set a action buffer with size 10
    self.action_buffer = []
    self.action_buffer_size = 10

    if self.train:
        if not os.path.exists(MODEL_PATH):
            self.model.init_parameters()
            # Copy the model parameters to the target model
            self.target_model.load_state_dict(self.model.state_dict())
            self.logger.info('Model parameters initialized for training')
        elif os.path.exists(MODEL_PATH):
            self.model.load(MODEL_PATH)
            self.target_model.load(MODEL_PATH)
            self.model.exploration_prob = 0.3
            self.logger.info('Model for training loaded')
    else:
        self.model.load(MODEL_PATH)
        self.model.eval()
        self.model.requires_grad_(False)
        self.logger.info('Model for evaluation loaded')


def act(agent, game_state: dict):
    # For the first step of each round, we record the spawn position
    if game_state['step'] == 1:
        agent.spwan_position = game_state['self'][3]
        if agent.spwan_position[0] == 1 and agent.spwan_position[1] == 1:
            agent.rotate = 0
        elif agent.spwan_position[0] == 15 and agent.spwan_position[1] == 1:
            agent.rotate = 90
        elif agent.spwan_position[0] == 15 and agent.spwan_position[1] == 15:
            agent.rotate = 180
        elif agent.spwan_position[0] == 1 and agent.spwan_position[1] == 15:
            agent.rotate = 270
    agent.logger.info('Choosing action based on current state.')
    # Get the current state representation
    current_state = get_low_level_state(game_state)

    # Use the model to choose an action
    action = agent.model.action(current_state)

    # Rotate the action by 90, 180, 270 degrees
    action = rotate_action(action, agent.rotate)

    # Convert the action index to the corresponding action string
    action_string = ACTIONS[action]

    agent.logger.info(f'Agent Action: {action_string}')

    # print location
    agent.logger.info(f'Agent Location: {game_state["self"][3]}')


    return action_string

"""
Rotate the action by 90, 180, 270 degrees.
"""
def rotate_action(action, angle):
    # If action is WAIT or BOMB, return the same action
    if action == 4 or action == 5:
        return action
    if angle == 90:
        return (action + 1) % 4
    elif angle == 180:
        return (action + 2) % 4
    elif angle == 270:
        return (action + 3) % 4
    else:
        return action