
import os
import torch
import numpy as np

""" 
Experience:
 Basic Experience structure: Every state has an Experience: (s, a, r, s', done)
"""
class Experience:
    def __init__(self, global_state=None, agent_state=None, action=None, reward=None, global_next_state=None, agent_next_state=None, done=None, last_action=None, last_action_invalid=None):
        self.global_state = global_state
        self.agent_state = agent_state
        self.action = action
        self.reward = reward
        self.global_next_state = global_next_state
        self.agent_next_state = agent_next_state
        self.done = done
        self.action_type = None
        self.last_action = last_action
        self.last_action_invalid = last_action_invalid

    # Add setters for the new attributes if needed
    def set_last_action(self, last_action):
        self.last_action = last_action

    def set_last_action_invalid(self, last_action_invalid):
        self.last_action_invalid = last_action_invalid

    def is_complete(self):
        return all(attr is not None for attr in [
            self.global_state, self.agent_state, self.action, self.reward,
            self.global_next_state, self.agent_next_state, self.done,
            self.last_action, self.last_action_invalid
        ])

