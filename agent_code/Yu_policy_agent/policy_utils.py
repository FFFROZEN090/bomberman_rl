import numpy as np
import settings as s

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
from .rulebased_teacher import TeacherModel

from .config import ACTIONS

# General policy class
class BasePolicy(nn.Module):
    def __init__(self, feature_dim, hidden_dim, n_layers, seq_len, alpha, action_dim=len(ACTIONS), episode=0,
                 gamma=0.99, model_name='', lr=0.001, WANDB=False):
        super(BasePolicy, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.WANDB = WANDB
        
        # Episode information
        self.game_state_history = []
        self.action_probs = []
        self.action_history = []
        self.rewards = []
        self.state_seqs = []
        self.hidden = None
        self.birth_corner = None # 0: left top, 1: right top, 2: left bottom, 3: right bottom
        self.bomb_dropped = 0
        
        # parameters for saving and loading the model
        self.checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
        self.name = model_name
        self.episode = episode
        self.gamma = gamma
        self.lr = lr
        # self.epsilon = epsilon # in the policy-based methods, epsilon is not used
        
        # performance metrics
        self.loss_values = []
        self.teacher_loss = []
        self.policy_loss = []
        self.final_rewards = []
        self.final_discounted_rewards = []
        self.scores = []
        self.survival_time = []
        self.bomb_dropped_history = []
        self.scoring_efficiency = []
        
        # teacher model for imitation learning
        self.teacher = TeacherModel()
        self.teacher_action_history = []
        self.alpha = alpha # weight for the imitation loss
        
        # Initialize wandb
        if self.WANDB:
            wandb.init(
                project="MLE_Bomberman",
                config={
                    "architecture": self.name,
                    "episode": self.episode,
                    # "feature_dim": self.feature_dim,
                    "action_dim": self.action_dim,
                    "hidden_dim": self.hidden_dim,
                    "n_layers": self.n_layers,
                    "seq_len": self.seq_len,
                    "alpha": self.alpha,
                    "learning_rate": self.lr,
                    "gamma": self.gamma
                }
            )
        
        
    def init_optimizer(self, lr=0.001):
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)
    
    def forward(self, features):
        raise NotImplementedError("Subclasses should implement this!")

    def save(self, path = None):
        checkpoint = {
            'episode': self.episode,
            'model_name': self.name,
            'gamma': self.gamma,
            'lr': self.lr,
            'n_layers': self.n_layers,
            'seq_len': self.seq_len,
            'alpha': self.alpha,
            'loss_values': self.loss_values,
            'final_rewards': self.final_rewards,
            'final_discounted_rewards': self.final_discounted_rewards,
            'scores': self.scores,
            'survival_time': self.survival_time,
            'bomb_dropped_history': self.bomb_dropped_history,
            'scoring_efficiency': self.scoring_efficiency,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        if path is None:
            path = os.path.join(self.checkpoint_dir, f"{self.name}_{self.episode}.pt")
        
        torch.save(checkpoint, path)

        # Upload the model to wandb
        # wandb.save(path)
    
    
    def load(self, path = None):
        if path is None:
            print('No checkpoint path is given.')
        elif os.path.isfile(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episode = checkpoint['episode']
            self.name = checkpoint['model_name']
            self.gamma = checkpoint['gamma']
            self.n_layers = checkpoint['n_layers']
            self.lr = checkpoint['lr']
            self.loss_values = checkpoint['loss_values']
            self.final_rewards = checkpoint['final_rewards']
            self.final_discounted_rewards = checkpoint['final_discounted_rewards']
            self.scores = checkpoint['scores']
            self.alpha = checkpoint['alpha']
            
            self.seq_len = checkpoint['seq_len']
            self.bomb_dropped_history = checkpoint['bomb_dropped_history']
            self.survival_time = checkpoint['survival_time']
            self.scoring_efficiency = checkpoint['scoring_efficiency']
            print('Model loaded from', path)
        else:
            print('No model found at', path)

    def train(self):
        raise NotImplementedError("Subclasses should implement this!")
    
    # TODO: visualizations
    def visualize(self):
        pass
    
    def reset(self):
        self.hidden = None
        self.state_seqs.clear()
        self.game_state_history.clear()
        self.teacher_action_history.clear()
        self.action_history.clear()
        self.rewards.clear()
        self.action_probs.clear()
        self.birth_corner = None
        self.bomb_dropped = 0
    
    def train(self):
        discounted_rewards = self.getting_discounted_rewards(standadised=True)
        
        # Get steps
        if len(self.rewards) == len(self.action_history) == len(self.game_state_history) == len(self.action_probs):
            steps = len(self.rewards)
        else:
            steps = len(self.rewards)-1
        
        total_loss, total_teacher_loss, total_policy_loss = self.getting_loss(steps=steps, discounted_rewards=discounted_rewards)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.final_rewards.append(sum(self.rewards))
        self.final_discounted_rewards.append(discounted_rewards[0])
        self.loss_values.append(total_loss.item()/steps)
        self.teacher_loss.append(total_teacher_loss.item()/steps)
        self.policy_loss.append(total_policy_loss.item()/steps)
        self.survival_time.append(steps)
        self.bomb_dropped_history.append(self.bomb_dropped)
        self.scoring_efficiency.append(self.scores[-1]/steps)

    def state_to_features(self, game_state: dict) -> torch.Tensor:
        # This is a function that converts the game state to the input of your model
        
        # General information
        self_pos = game_state['self'][3]
        arena = game_state['field']
        free = arena == 0
        bombs = game_state['bombs']
        others = [xy for (n, s, b, xy) in game_state['others']]
        
        bombs_time = np.ones(arena.shape) * 5
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bombs_time.shape[0]) and (0 < j < bombs_time.shape[1]):
                    bombs_time[i, j] = min(bombs_time[i, j], t)
        
        # 1. Situational awareness for the agent: Walls, crates, bombs, other agents, bomb_left
        # Features 1.1. determine which corner the agent is in: left top, right top, left bottom, right bottom
        
        # make use of the corner information 
        if game_state['step'] == 1:
            if self_pos == (1,1): # left top
                self.birth_corner = 0
            elif self_pos == (1,s.ROWS-2): # left bottom?
                self.birth_corner = 1
            elif self_pos == (s.COLS-2,1): # right top?
                self.birth_corner = 2
            elif self_pos == (s.COLS-2,s.ROWS-2): # right bottom
                self.birth_corner = 3
            else:
                raise ValueError("The birth corner is not determined at step 0.")
        # else:
            # print("The birth corner is: ", self.birth_corner, 'at step ', game_state['step'])
        
        # Features 1.2. determine the direction feasibility
        candidate_position = [(self_pos[0], self_pos[1]-1), (self_pos[0], self_pos[1]+1), 
                            (self_pos[0]-1, self_pos[1]), (self_pos[0]+1, self_pos[1]), 
                            self_pos]
        valid_position = []
        for pos in candidate_position:
            if ((arena[pos] == 0) and 
                    (game_state['explosion_map'][pos] < 1) and 
                    (bombs_time[pos] > 0) and
                    (all(pos != other_pos for other_pos in others)) and 
                    (all(pos != bomb for bomb, t in bombs))):
                valid_position.append(pos)
        up_feasible = (self_pos[0], self_pos[1]-1) in valid_position
        down_feasible = (self_pos[0], self_pos[1]+1) in valid_position
        left_feasible = (self_pos[0]-1, self_pos[1]) in valid_position
        right_feasible = (self_pos[0]+1, self_pos[1]) in valid_position
        wait_feasible = self_pos in valid_position
        
        # Features 1.3. bomb left
        bomb_left = game_state['self'][2] > 0 
        
        
        # 2. Pathfinding features: coins nearby, crates nearby
        # Features 2.1. determine the distance to the coins
        coins = game_state['coins']
        up_coins_score = 0
        down_coins_score = 0
        left_coins_score = 0
        right_coins_score = 0
        for (xc, yc) in coins:
            if yc > self_pos[1]:
                up_coins_score += 1/(yc-self_pos[1])
            elif yc < self_pos[1]:
                down_coins_score += 1/(self_pos[1]-yc)
            if xc > self_pos[0]:
                right_coins_score += 1/(xc-self_pos[0])
            elif xc < self_pos[0]:
                left_coins_score += 1/(self_pos[0]-xc)
                
        up_coin_distance = 0
        down_coin_distance = 0
        left_coin_distance = 0
        right_coin_distance = 0
        for (xc, yc) in coins:
            up_coin_distance 
            
        
        # Features 2.2. determine the distance to the crates
        crates_list = [(i, j) for i in range(s.COLS) for j in range(s.ROWS) if arena[i, j] == 1]
        up_crates_score = 0
        down_crates_score = 0
        left_crates_score = 0
        right_crates_score = 0
        # TODO: How to encode the crates information?
        for (xc, yc) in crates_list:
            if yc > self_pos[1] and up_crates_score < 1/(yc-self_pos[1]):
                up_crates_score = 1/(yc-self_pos[1])
            elif yc < self_pos[1] and down_crates_score < 1/(self_pos[1]-yc):
                down_crates_score = 1/(self_pos[1]-yc)
            if xc > self_pos[0] and right_crates_score < 1/(xc-self_pos[0]):
                right_crates_score = 1/(xc-self_pos[0])
            elif xc < self_pos[0] and left_crates_score < 1/(self_pos[0]-xc):
                left_crates_score = 1/(self_pos[0]-xc)
        
        # Features 2.3. determine the distance to the deadend
        dead_ends = [(i, j) for i in range(s.COLS) for j in range(s.ROWS) if (arena[i, j] == 0)
                    and ([arena[i + 1, j], arena[i - 1, j], arena[i, j + 1], arena[i, j - 1]].count(0) == 1)]
        up_dead_ends_score = 0
        down_dead_ends_score = 0
        left_dead_ends_score = 0
        right_dead_ends_score = 0
        for (xd, yd) in dead_ends:
            if yd > self_pos[1] and up_dead_ends_score < 1/(yd-self_pos[1]):
                up_dead_ends_score = 1/(yd-self_pos[1])
            elif yd < self_pos[1] and down_dead_ends_score < 1/(self_pos[1]-yd):
                down_dead_ends_score = 1/(self_pos[1]-yd)
            if xd > self_pos[0] and right_dead_ends_score < 1/(xd-self_pos[0]):
                right_dead_ends_score = 1/(xd-self_pos[0])
            elif xd < self_pos[0] and left_dead_ends_score < 1/(self_pos[0]-xd):
                left_dead_ends_score = 1/(self_pos[0]-xd)
                
        # Feature 2.4.determine the distance to the others
        up_opponents_score = 0
        down_opponents_score = 0
        left_opponents_score = 0
        right_opponents_score = 0
        for (xo, yo) in others:
            if yo > self_pos[1]:
                up_opponents_score += 1/(yo-self_pos[1])
            elif yo < self_pos[1]:
                down_opponents_score += 1/(self_pos[1]-yo)
            if xo > self_pos[0]:
                right_opponents_score += 1/(xo-self_pos[0])
            elif xo < self_pos[0]:
                left_opponents_score += 1/(self_pos[0]-xo)
        
        # 3. Life-saving features: bomb nearby, explosion nearby
        # Features 3.1. determine the distance to the bomb
        up_bomb1_distance = 5
        down_bomb1_distance = 5
        left_bomb1_distance = 5
        right_bomb1_distance = 5
        up_bomb1_time = 5
        down_bomb1_time = 5
        left_bomb1_time = 5
        right_bomb1_time = 5
        up_bomb1_safe = 1
        down_bomb1_safe = 1
        left_bomb1_safe = 1
        right_bomb1_safe = 1
        for (xb, yb), t in bombs:
            if abs(yb-self_pos[1]) + abs(xb-self_pos[0]) < 5:
                # if the bomb is around a wall, then the agent could hide behind
                # initialize the safety information
                up_bomb1_safe = 0
                down_bomb1_safe = 0
                left_bomb1_safe = 0
                right_bomb1_safe = 0
                # determine the safety information
                # the bomb is on the right side of the wall and the agent is on the left side of the wall 
                if arena[xb-1, yb] == -1 and self_pos[0] < xb: 
                    up_bomb1_safe = 1
                    down_bomb1_safe = 1
                if arena[xb+1, yb] == -1 and self_pos[0] > xb:
                    up_bomb1_safe = 1
                    down_bomb1_safe = 1
                if arena[xb, yb-1] == -1 and self_pos[1] < yb:
                    left_bomb1_safe = 1
                    right_bomb1_safe = 1
                if arena[xb, yb+1] == -1 and self_pos[1] > yb:
                    left_bomb1_safe = 1
                    right_bomb1_safe = 1          
                
                up_bomb1_distance = min(up_bomb1_distance, abs(yb-self_pos[1])) if yb < self_pos[1] else up_bomb1_distance
                down_bomb1_distance = min(down_bomb1_distance, abs(yb-self_pos[1])) if yb > self_pos[1] else down_bomb1_distance
                left_bomb1_distance = min(left_bomb1_distance, abs(xb-self_pos[0])) if xb < self_pos[0] else left_bomb1_distance
                right_bomb1_distance = min(right_bomb1_distance, abs(xb-self_pos[0])) if xb > self_pos[0] else right_bomb1_distance
                up_bomb1_time = min(up_bomb1_time, t) if yb < self_pos[1] else up_bomb1_time
                down_bomb1_time = min(down_bomb1_time, t) if yb > self_pos[1] else down_bomb1_time
                left_bomb1_time = min(left_bomb1_time, t) if xb < self_pos[0] else left_bomb1_time
                right_bomb1_time = min(right_bomb1_time, t) if xb > self_pos[0] else right_bomb1_time
                
                        
        
        # merge all features
        if self.birth_corner == 0: # left top
            features = np.array([up_feasible, right_feasible, down_feasible, left_feasible, wait_feasible, bomb_left,
                                up_coins_score, right_coins_score, down_coins_score, left_coins_score, 
                                up_crates_score, right_crates_score, down_crates_score, left_crates_score, 
                                up_dead_ends_score, right_dead_ends_score, down_dead_ends_score, left_dead_ends_score,
                                up_opponents_score, right_opponents_score, down_opponents_score, left_opponents_score,
                                up_bomb1_distance, right_bomb1_distance, down_bomb1_distance, left_bomb1_distance,
                                up_bomb1_time, right_bomb1_time, down_bomb1_time, left_bomb1_time,
                                up_bomb1_safe, right_bomb1_safe, down_bomb1_safe, left_bomb1_safe])
        elif self.birth_corner == 1: # left bottom
            features = np.array([down_feasible, right_feasible, up_feasible, left_feasible, wait_feasible, bomb_left,
                                down_coins_score, right_coins_score, up_coins_score, left_coins_score,
                                down_crates_score, right_crates_score, up_crates_score, left_crates_score,
                                down_dead_ends_score, right_dead_ends_score, up_dead_ends_score, left_dead_ends_score,
                                down_opponents_score, right_opponents_score, up_opponents_score, left_opponents_score,
                                down_bomb1_distance, right_bomb1_distance, up_bomb1_distance, left_bomb1_distance,
                                down_bomb1_time, right_bomb1_time, up_bomb1_time, left_bomb1_time,
                                down_bomb1_safe, right_bomb1_safe, up_bomb1_safe, left_bomb1_safe])
        elif self.birth_corner == 2: # right top
            features = np.array([up_feasible, left_feasible, down_feasible, right_feasible, wait_feasible, bomb_left,
                                up_coins_score, left_coins_score, down_coins_score, right_coins_score,
                                up_crates_score, left_crates_score, down_crates_score, right_crates_score, 
                                up_dead_ends_score, left_dead_ends_score, down_dead_ends_score, right_dead_ends_score,
                                up_opponents_score, left_opponents_score, down_opponents_score, right_opponents_score, 
                                up_bomb1_distance, left_bomb1_distance, down_bomb1_distance, right_bomb1_distance,
                                up_bomb1_time, left_bomb1_time, down_bomb1_time, right_bomb1_time,
                                up_bomb1_safe, left_bomb1_safe, down_bomb1_safe, right_bomb1_safe])
        elif self.birth_corner == 3: # right bottom
            features = np.array([down_feasible, left_feasible, up_feasible, right_feasible, wait_feasible, bomb_left,
                                 down_coins_score, left_coins_score, up_coins_score, right_coins_score,
                                 down_crates_score, left_crates_score, up_crates_score, right_crates_score,
                                 down_dead_ends_score, left_dead_ends_score, up_dead_ends_score, right_dead_ends_score,
                                 down_opponents_score, left_opponents_score, up_opponents_score, right_opponents_score,
                                 down_bomb1_distance, left_bomb1_distance, up_bomb1_distance, right_bomb1_distance,
                                 down_bomb1_time, left_bomb1_time, up_bomb1_time, right_bomb1_time,
                                 down_bomb1_safe, left_bomb1_safe, up_bomb1_safe, right_bomb1_safe])
        else:
            raise ValueError("The birth corner is not determined.")
        
        features = torch.tensor(features, dtype=torch.float32)
        # print("features: ", features, "birth_corner: ", self.birth_corner)
        return features
    
    def getting_action_probs(self, x):
        def swap(x, i, j):
            result = x.clone()
            result[i] = x[j]
            result[j] = x[i]
            return result
        # according to the forward output and self corner information, determine the action probabilities
        if self.birth_corner == 1: # left bottom
            # swap the first and the third element of the output
            x = swap(x, 0, 2)
        elif self.birth_corner == 2: # right top
            # swap the second and the fourth element of the output
            x = swap(x, 1, 3)
        elif self.birth_corner == 3: # right bottom
            # swap the output elements
            x = swap(x, 0, 2)
            x = swap(x, 1, 3)
        # normalize the output
        action_probs = F.softmax(x, dim=-1)
        
        # print("The action probabilities are: ", action_probs)
        return action_probs
    
    def getting_discounted_rewards(self, standadised = False):
        # calculate the discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        
        if standadised:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        return discounted_rewards
    
    def getting_loss(self, steps, discounted_rewards):
        total_loss = 0
        total_teacher_loss = 0
        total_policy_loss = 0
            
        for t in range(steps):
            rewards = discounted_rewards[t]
            
            # Calculate action probabilities
            action_prob = F.softmax(self.forward(index=t), dim=0)
            
            # Calculate the imitation learning loss (cross-entropy loss between teacher's action and agent's action)
            if self.teacher_action_history[t] is not None:
                teacher_action_idx = ACTIONS.index(self.teacher_action_history[t])
            else:
                teacher_action_idx = 5 # WAIT action
                print('The teacher action is None at step ', t)
            teacher_action_prob = torch.zeros(self.action_dim)
            teacher_action_prob[teacher_action_idx] = 1
            teacher_loss = F.cross_entropy(action_prob, teacher_action_prob)
            
            total_teacher_loss += teacher_loss
            
            # Calculate the RL loss
            log_prob = torch.log(action_prob)[self.action_history[t]] # add a small epsilon to avoid log(0)
            policy_loss = -log_prob * rewards
            
            total_policy_loss += policy_loss
            
            total_loss += teacher_loss*self.alpha + policy_loss*(1-self.alpha)
            
        return total_loss, total_teacher_loss, total_policy_loss
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
