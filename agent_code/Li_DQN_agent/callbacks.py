"""
This file contains the callback functions for the DQN agent.
"""
import os
import numpy as np
from .DQN_utils import get_state, get_low_level_state, get_high_level_state
from .DQN_network import DQN, ExperienceDataset, ReplayBuffer
import logging
from collections import deque


EXPERIENCE_BUFFER_SIZE = 100000
REPLAY_BUFFER_SIZE = 600

MODEL_NAME = 'Li_DQN_agent'
LAST_EPISODE = 0
INPUT_CHANNELS = 7

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', MODEL_NAME + '_' + str(LAST_EPISODE) + '.pt')

def setup(self):
    np.random.seed()
    self.logger.info('Successfully entered setup code')

    # Setup the model
    self.model = DQN(input_channels=INPUT_CHANNELS, output_size=6)
    self.model.epoch = LAST_EPISODE
    self.target_model = DQN(input_channels=INPUT_CHANNELS, output_size=6)

    # Store the last state and action
    self.last_state = None
    self.last_action = None
    self.last_action_type = None
    self.last_reward = None
    self.last_events = None

    # Store last game state
    self.last_game_state = None

    self.bomb_cooldown = 0
    self.last_action_invalid = False

    # Initialize the experience buffer
    self.experience_buffer = ExperienceDataset(EXPERIENCE_BUFFER_SIZE)

    # Initialize the replay buffer
    self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    self.batch_size = REPLAY_BUFFER_SIZE

    self.surving_rounds = 0

    self.spwan_position = None
    self.rotate = 0

    # Set a action buffer with size 10
    self.action_buffer = []
    self.action_buffer_size = 8

    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0

    if self.train:
        print(f'Model path: {MODEL_PATH}')
        model_exists = os.path.exists(MODEL_PATH)
        print(f'Model exists: {model_exists}')
        if model_exists:
            self.model.load(MODEL_PATH)
            self.target_model.load(MODEL_PATH)
            self.model.exploration_prob = 0.1
            self.logger.info('Model for training loaded')
        elif not model_exists:
            self.model.init_parameters()
            # Copy the model parameters to the target model
            self.target_model.load_state_dict(self.model.state_dict())
            self.logger.info('Model parameters initialized for training')
    
    else:
        self.model.load(MODEL_PATH)
        self.target_model.load(MODEL_PATH)
        self.model.eval()
        self.model.requires_grad_(False)
        self.logger.info('Model for evaluation loaded')


def act(agent, game_state: dict):
    # For the first step of each round, we record the spawn position

    agent.spwan_position = game_state['self'][3]
    if 8 >= agent.spwan_position[0] >= 1 and 8 >= agent.spwan_position[1] >= 1:
        agent.rotate = 0
    elif agent.spwan_position[0] >= 8 and 8 >= agent.spwan_position[1] >= 1:
        agent.rotate = 90
    elif agent.spwan_position[0] >= 8 and agent.spwan_position[1] >= 8:
        agent.rotate = 180
    elif 8 >= agent.spwan_position[0] >= 1 and agent.spwan_position[1] >= 8:
        agent.rotate = 270
    # Get the current state representation
    current_state, low_state = get_state(game_state, rotate=agent.rotate, bomb_valid=game_state['self'][2])

    # Get Agent's position from current state, first dimension where 1 is present
    agent_position = np.where(current_state[0] == 1)

    if game_state['step'] == 1:
        agent.logger.debug(f"New round started")

    # Use the model to choose an action
    action, action_type = agent.model.action(current_state, low_state, agent.last_action_invalid, agent.last_action, agent.rotate)
    agent.logger.debug(f'Raw Action: {ACTIONS[action]}, Mirror: {agent.rotate}')

    # Convert the action index to the corresponding action string
    action_string = ACTIONS[action]

    if len(agent.action_buffer) < agent.action_buffer_size:
        agent.action_buffer.append(action_string)
    else:
        agent.action_buffer.pop(0)
        agent.action_buffer.append(action_string)

    if len(agent.model.action_buffer) < agent.action_buffer_size:
        agent.model.action_buffer.append(action_string)
    else:
        agent.model.action_buffer.pop(0)
        agent.model.action_buffer.append(action_string)

    # Rotate the action by 90, 180, 270 degrees
    action = rotate_action(action, agent.rotate)
    action_string = ACTIONS[action]

    agent.logger.info(f'Actual Action: {action_string}')

    agent.last_action_type = action_type


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