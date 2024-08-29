
import os
import torch
import numpy as np

""" 
Experience:
 Basic Experience structure: Every state has an Experience: (s, a, r, s', done)
"""
class Experience:
    def __init__(self, global_state=None, agent_state=None, action=None, reward=None, global_next_state=None, agent_next_state=None, done=None):
        self.global_state = global_state
        self.agent_state = agent_state
        self.action = action
        self.reward = reward
        self.global_next_state = global_next_state
        self.agent_next_state = agent_next_state
        self.done = done

    def set_global_state(self, global_state):
        self.global_state = global_state

    def set_agent_state(self, agent_state):
        self.agent_state = agent_state

    def set_action(self, action):
        self.action = action

    def set_reward(self, reward):
        self.reward = reward

    def set_global_next_state(self, global_next_state):
        self.global_next_state = global_next_state

    def set_agent_next_state(self, agent_next_state):
        self.agent_next_state = agent_next_state

    def set_done(self, done):
        self.done = done

    def is_complete(self):
        return all(attr is not None for attr in [self.global_state, self.agent_state, self.action, self.reward, self.global_next_state, self.agent_next_state, self.done])
