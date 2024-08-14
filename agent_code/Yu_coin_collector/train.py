import numpy as np
import random
import events as e
import settings as s
import logging
# from .policynet import Policy
import torch
import torch.nn as nn

from collections import deque

from typing import List

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


def setup_training(self):
    self.gamma = 0.95
    self.visited_history = deque([], 20)
    self.episode = 0
    
    self.logger = logging.getLogger(__name__)
    
    
    

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[dict]) -> None:
    self.logger.debug(f'Encountered game event(s) {", ".join([event for event in events])}')
    
    # add position to visited history
    self.visited_history.append(new_game_state['self'][3])
    # check if the agent is in a loop, if so, add an event to events list
    if self.visited_history.count(new_game_state['self'][3]) > 2:
        events.append(LOOP_DETECTED)
    
    # distance to coins: if getting close to coins, add an event to events list
    coins_pos = old_game_state['coins']
    for coin_pos in coins_pos:
        if np.linalg.norm(np.array(coin_pos) - np.array(new_game_state['self'][3])) < 4: # falls into the coin range
            events.append(COIN_CLOSE)
            if np.linalg.norm(np.array(coin_pos) - np.array(new_game_state['self'][3])) < np.linalg.norm(np.array(coins_pos[0]) - np.array(old_game_state['self'][3])):
                events.append(COIN_CLOSER)
    
    # distance to bombs: if still in danger zone, add an event to events list
    bombs = old_game_state['bombs']
    bombs_time = np.ones((s.COLS, s.ROWS)) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bombs_time.shape[0]) and (0 < j < bombs_time.shape[1]):
                bombs_time[i, j] = min(bombs_time[i, j], t)
    if bombs_time[new_game_state['self'][3]] == 1:
        events.append(BOMB_TIME1)
    elif bombs_time[new_game_state['self'][3]] == 2:
        events.append(BOMB_TIME2)
    elif bombs_time[new_game_state['self'][3]] == 3:
        events.append(BOMB_TIME3)
    elif bombs_time[new_game_state['self'][3]] == 4:
        events.append(BOMB_TIME4)
        
    # If the agent was in danger zone but now safe, add an event to events list
    if bombs_time[old_game_state['self'][3]] < 5 and bombs_time[new_game_state['self'][3]] == 5:
        events.append('EXCAPE_FROM_BOMB')
        
    # If the agent dropped a bomb and crates will be destroyed, add an event to events list
    crates = old_game_state['field'] == 1
    if self_action == 'BOMB':
        for (i, j) in [(new_game_state['self'][3][0] + h, new_game_state['self'][3][1]) for h in range(-3, 4)] + [(new_game_state['self'][3][0], new_game_state['self'][3][1] + h) for h in range(-3, 4)]:
            if (0 < i < crates.shape[0]) and (0 < j < crates.shape[1]) and crates[i, j] and bombs_time[i, j] == 5:
                events.append(BOMB_DROPPED_FOR_CRATE)
                
def reset_params(self):
    self.visited_history = deque([], 20)
    self.episode += 1
    
    self.model.rewards = []
    self.model.action_probs = []
    

def end_of_round(self, last_game_state, last_action, events):
    # last step of the round, calculate the reward and update the model
    reward = self.reward_from_events(events)
    self.model.rewards.append(reward)
    self.model.scores.append(last_game_state['self'][1])
    self.model.action_probs.append(self.policy_net.get_action_probs(last_game_state))
    
    # update the model
    self.train()
    
    # reset the parameters for the next round
    self.reset_params()
    
    self.model.episode += 1
    
    # Save model for every 200 episodes
    if self.model.episode % 200 == 0:
        self.model.save()
        

def reward_from_events(self, events) -> float:
    reward = 0
    game_rewards = {
        e.INVALID_ACTION: -0.05,
        e.MOVED_LEFT: -0.01,
        e.MOVED_RIGHT: -0.01,
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.WAITED: -0.03,
        e.BOMB_DROPPED: -0.01,
        
        LOOP_DETECTED: -0.1,
        
        e.CRATE_DESTROYED: 0.05,
        e.COIN_FOUND: 0.3,
        COIN_CLOSE: 0.05,
        COIN_CLOSER: 0.15,
        e.COIN_COLLECTED: 1,
        
        BOMB_TIME4: -0.1,
        BOMB_TIME3: -0.2,
        BOMB_TIME2: -0.3,
        BOMB_TIME1: -0.5,
        EXCAPE_FROM_BOMB: 0.5,
        
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,
        e.SURVIVED_ROUND: 5
    }
    reward = sum([game_rewards[event] for event in events])
    self.logger.info(f"Awarded {reward} for events {' ,'.join(events)}")
    return reward