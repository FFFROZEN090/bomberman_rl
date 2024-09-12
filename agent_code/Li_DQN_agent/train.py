import numpy as np
import random
import events as e
import settings as s
import logging
# from policynet import PolicyNet
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

from typing import List

import json

from .DQN_network import ExperienceDataset, ReplayBuffer
from .DQN_datatype import Experience
from .DQN_utils import get_state, get_low_level_state, get_high_level_state


# events
COIN_CLOSE = 'COIN_CLOSE'
COIN_CLOSER = 'COIN_CLOSER'
BOMB_TIME1 = 'BOMB_TIME1' # 1 step to explode
BOMB_TIME2 = 'BOMB_TIME2' # 2 steps to explode
BOMB_TIME3 = 'BOMB_TIME3' # 3 steps to explode
BOMB_TIME4 = 'BOMB_TIME4' # 4 steps to explode
BOMB_DROPPED_FOR_CRATE = 'BOMB_DROPPED_FOR_CRATE' # Crates will be destroyed by the dropped bomb
EXCAPE_FROM_BOMB = 'EXCAPE_FROM_BOMB'
LOOP_DETECTED = 'LOOP_DETECTED'
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
UP_REPEAT = 'UP_REPEAT'
RIGHT_REPEAT = 'RIGHT_REPEAT'
DOWN_REPEAT = 'DOWN_REPEAT'
LEFT_REPEAT = 'LEFT_REPEAT'
WAIT_REPEAT = 'WAIT_REPEAT'
BOMB_REPEAT = 'BOMB_REPEAT'
WAIT = 'WAIT'
RIGHT = 'RIGHT'
UP = 'UP'
DOWN = 'DOWN'
LEFT = 'LEFT'
BOMB = 'BOMB'


# TODO: Setup model and experience structure for DQN
def setup_training(self):
    self.visited_history = deque([], 20)
    self.epoch = 0
    self.episode = 0


# TODO: Verify the game event, update the model, reward, experience
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[dict]) -> None:
    self.logger.debug(f'Encountered game event(s) {", ".join([event for event in events])}')

    # print agent position
    self.logger.info(f'Agent position: {old_game_state["self"][3]}')
    
    events = calculate_events(self, old_game_state, self_action, new_game_state, events)

    # Get DQN state
    state = get_state(new_game_state, rotate=self.rotate)

    # Get old state
    old_state = get_state(old_game_state, self.rotate)

    # Get reward
    reward = calculate_reward(events, self.last_action_type)
    
    if self.last_action is None:
        self.last_action = self_action

    # Get Last Action number
    action_number = ACTIONS.index(self_action)

    # If self.last_reward is not None, then store the experience
    if self.last_reward is not None and self.last_action is not None:
        self.experience_buffer.add(Experience(old_state, None, action_number, reward, state, None, False))
        self.replay_buffer.add(Experience(old_state, None, action_number, reward, state, None, False))

    # Update last reward
    self.last_reward = reward

    self.logger.info(f'Events: {events}')

    if len(self.replay_buffer) == self.batch_size:
        self.model.dqn_train(self.replay_buffer, self.experience_buffer, self.batch_size, self.target_model)

        # Update the epoch
        self.model.epoch += 1

        self.replay_buffer.clear()

    self.surving_rounds += 1

def calculate_events(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[dict]) -> List[dict]:
    # add position to visited history
    self.visited_history.append(new_game_state['self'][3])
    # check if the agent is in a loop, if so, add an event to events list
    if self.visited_history.count(new_game_state['self'][3]) > 4:
        events.append(LOOP_DETECTED)
    
    # distance to coins: if getting close to coins, add an event to events list
    coins_pos = old_game_state['coins']
    for coin_pos in coins_pos:
        if np.linalg.norm(np.array(coin_pos) - np.array(new_game_state['self'][3])) < 4: # falls into the coin range
            events.append(COIN_CLOSE)
            if np.linalg.norm(np.array(coin_pos) - np.array(new_game_state['self'][3])) < np.linalg.norm(np.array(coin_pos) - np.array(old_game_state['self'][3])):
                events.append(COIN_CLOSER)
    
    # distance to bombs: if still in danger zone, add an event to events list
    bombs = old_game_state['bombs']
    bombs_time = np.ones((s.COLS, s.ROWS)) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bombs_time.shape[0]) and (0 < j < bombs_time.shape[1]):
                bombs_time[i, j] = min(bombs_time[i, j], t)
        
    # If the agent was in danger zone but now safe, add an event to events list
    if bombs_time[old_game_state['self'][3]] < 5 and bombs_time[new_game_state['self'][3]] == 5:
        events.append('EXCAPE_FROM_BOMB')
        
    # If the agent dropped a bomb and crates will be destroyed, add an event to events list
    crates = old_game_state['field'] == 1
    if self_action == 'BOMB':
        for (i, j) in [(new_game_state['self'][3][0] + h, new_game_state['self'][3][1]) for h in range(-3, 4)] + [(new_game_state['self'][3][0], new_game_state['self'][3][1] + h) for h in range(-3, 4)]:
            if (0 < i < crates.shape[0]) and (0 < j < crates.shape[1]) and crates[i, j] and bombs_time[i, j] == 5:
                events.append(BOMB_DROPPED_FOR_CRATE)

    # Events for Repeat Actions, if one action repeats more than 5 times, add an event to events list, the action buffer size is 10
    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    if len(self.action_buffer) == self.action_buffer_size:
        # UP Repeat 5 times in a 10 steps window
        if self.action_buffer.count('UP') >= self.action_buffer_size // 2 - 1:
            events.append('UP_REPEAT')
            self.action_buffer.clear()
        elif self.action_buffer.count('RIGHT') >= self.action_buffer_size // 2 - 1:
            events.append('RIGHT_REPEAT')
            self.action_buffer.clear()
        elif self.action_buffer.count('DOWN') >= self.action_buffer_size // 2 - 1:
            events.append('DOWN_REPEAT')
            self.action_buffer.clear()
        elif self.action_buffer.count('LEFT') >= self.action_buffer_size // 2 - 1:
            events.append('LEFT_REPEAT')
            self.action_buffer.clear()
        elif self.action_buffer.count('WAIT') >= self.action_buffer_size // 2 - 1:
            events.append('WAIT_REPEAT')
            self.action_buffer.clear()
        elif self.action_buffer.count('BOMB') >= self.action_buffer_size // 2 - 1:
            self.action_buffer.clear()

    events.append(self_action)

    return events
                    
    

# TODO: Verify the end of the game, update the model, score, reward, and log the game info
def end_of_round(self, last_game_state, last_action, events): 
    self.logger.debug(f'Encountered game event(s) {", ".join([event for event in events])}')

    # Get DQN state
    state = get_state(last_game_state, rotate=self.rotate)

    # Get old state
    old_state = get_state(last_game_state, rotate=self.rotate)

    # Get reward
    reward = calculate_reward(events, self.last_action_type)

    # Get Action number
    action_number = ACTIONS.index(last_action)

    # If self.last_reward is not None, then store the experience
    if self.last_reward is not None and self.last_action is not None:
        self.experience_buffer.add(Experience(old_state, None, action_number, reward, state, None, True))
        self.replay_buffer.add(Experience(old_state, None, action_number, reward, state, None, True))

    self.logger.info(f'Events: {events}')

    if len(self.replay_buffer) == self.batch_size:
        self.model.dqn_train(self.replay_buffer, self.experience_buffer, self.batch_size, self.target_model)

        # Update the epoch
        self.model.epoch += 1

        self.replay_buffer.clear()

    # Reset last reward
    self.last_reward = None
    self.last_game_state = None
    self.last_state = None
    self.last_action = None


    score = last_game_state['self'][1]

    self.logger.info(f'Round Score: {score}')
    
    self.model.scores.append(score)

    survived_rounds = self.surving_rounds
    self.surving_rounds = 0
    self.model.survived_rounds.append(survived_rounds)

    # Clear action buffer
    self.action_buffer.clear()



# Function to load reward settings from a JSON file
def load_reward_settings(file_path):
    with open(file_path, 'r') as file:
        reward_settings = json.load(file)
        
    # Convert joint_event_penalties keys from string to tuple
    reward_settings['joint_event_penalties'] = {
        eval(k): v for k, v in reward_settings['joint_event_penalties'].items()
    }
    return reward_settings

def reward_from_events(events, reward_settings) -> float:
    reward = 0
    game_rewards = reward_settings['event_rewards']
    
    reward = sum([game_rewards.get(event, 0) for event in events])

    # Punish for joint event
    joint_event_penalties = reward_settings['joint_event_penalties']
    
    for joint_event, penalty in joint_event_penalties.items():
        # Check if the joint event is a subset of the events
        if set(joint_event).issubset(set(events)):
            reward += penalty

    return reward

def calculate_reward(events, action_type):

    if action_type == "network":
        reward_settings = load_reward_settings("network_rewards.json")
    elif action_type == "exploration":
        reward_settings = load_reward_settings("exploration_rewards.json")
    
    reward = reward_from_events(events, reward_settings)

    return reward

    