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

import wandb

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
WANDB = True

# TODO: Setup model and experience structure for DQN
def setup_training(self):
    self.visited_history = deque([], 20)
    self.epoch = 0
    self.episode = 0


# TODO: Verify the game event, update the model, reward, experience
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[dict]) -> None:
    self.logger.debug(f'Encountered game event(s) {", ".join([event for event in events])}')
    # print agent position
    self.logger.info(f'Agent position: {new_game_state["self"][3]}')
    
    events = calculate_events(self, old_game_state, self_action, new_game_state, events)

    # Get DQN state
    state = get_low_level_state(new_game_state)

    # Get old state
    old_state = get_low_level_state(old_game_state)

    # Get reward
    reward = reward_from_events(events)
    
    if self.last_action is None:
        self.last_action = self_action

    # Get Last Action number
    last_action_number = ACTIONS.index(self.last_action) if self.last_action is not None else None

    # If self.last_reward is not None, then store the experience
    if self.last_reward is not None and self.last_action is not None:
        self.experience_buffer.add(Experience(old_state, None, last_action_number, reward, state, None, False))
        self.replay_buffer.add(Experience(old_state, None, last_action_number, reward, state, None, False))

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
    if self.visited_history.count(new_game_state['self'][3]) > 3:
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
    
    return events
                    
    

# TODO: Verify the end of the game, update the model, score, reward, and log the game info
def end_of_round(self, last_game_state, last_action, events): 
    self.logger.debug(f'Encountered game event(s) {", ".join([event for event in events])}')

    # Get DQN state
    state = get_low_level_state(last_game_state)

    # Get old state
    old_state = get_low_level_state(last_game_state)

    # Get reward
    reward = reward_from_events(events)

    # Get Action number
    last_action = ACTIONS.index(self.last_action) if self.last_action is not None else None

    # If self.last_reward is not None, then store the experience
    if self.last_reward is not None and self.last_action is not None:
        self.experience_buffer.add(Experience(old_state, None, last_action, reward, state, None, False))
        self.replay_buffer.add(Experience(old_state, None, last_action, reward, state, None, False))

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
    
    self.model.scores.append(score)

    survived_rounds = self.surving_rounds
    self.surving_rounds = 0
    self.model.survived_rounds.append(survived_rounds)


def reward_from_events(events) -> float:
    reward = 0
    game_rewards = {
        e.INVALID_ACTION: -0.5,
        e.MOVED_LEFT: 5.0,
        e.MOVED_RIGHT: 5.0,
        e.MOVED_UP: 2.0,
        e.MOVED_DOWN: 2.0,
        e.WAITED: -0.1,
        e.BOMB_DROPPED: 5.0,
        e.OPPONENT_ELIMINATED: 10.00,
        
        LOOP_DETECTED: -0.5,
        
        e.CRATE_DESTROYED: 10.0,
        e.COIN_FOUND: 10.0,
        COIN_CLOSE: 10.0,
        COIN_CLOSER: 20.0,
        e.COIN_COLLECTED: 100.0,
        
        EXCAPE_FROM_BOMB: 10.0,
        e.BOMB_EXPLODED: 1.0,
        
        BOMB_DROPPED_FOR_CRATE: 5.0,
        
        e.KILLED_OPPONENT: 500.0,
        e.GOT_KILLED: -100,
        e.KILLED_SELF: -5,
        e.SURVIVED_ROUND: 10.0
    }
    reward = sum([game_rewards[event] for event in events])

    # Punish for joint event
    joint_event_penalties = {
        (e.WAITED, LOOP_DETECTED): -1.0,
        (e.INVALID_ACTION, LOOP_DETECTED): -1.0,
        (e.BOMB_DROPPED, e.INVALID_ACTION): -1.0,
        (e.BOMB_DROPPED, e.WAITED): -1.0,
        (e.GOT_KILLED, e.KILLED_SELF): +100.0,
        (e.SURVIVED_ROUND, LOOP_DETECTED): -100.0,
        (e.MOVED_RIGHT, EXCAPE_FROM_BOMB): 20.0,
        (e.MOVED_LEFT, EXCAPE_FROM_BOMB): 20.0,
        (e.MOVED_UP, EXCAPE_FROM_BOMB): 5.0,
        (e.MOVED_DOWN, EXCAPE_FROM_BOMB): 5.0,
        (e.MOVED_RIGHT, COIN_CLOSE): 3.0,
        (e.MOVED_LEFT, COIN_CLOSE): 3.0,
        (e.MOVED_UP, COIN_CLOSE): 3.0,
        (e.MOVED_DOWN, COIN_CLOSE): 3.0,
        (e.MOVED_RIGHT, COIN_CLOSER): 5.0,
        (e.MOVED_LEFT, COIN_CLOSER): 5.0,
        (e.MOVED_UP, COIN_CLOSER): 5.0,
        (e.MOVED_DOWN, COIN_CLOSER): 5.0,
    }

        # Check if joint events occur and apply penalties
    for joint_event, penalty in joint_event_penalties.items():
        if all(event in events for event in joint_event):
            reward += penalty

    
    return reward

    