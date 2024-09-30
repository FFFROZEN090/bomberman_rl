import numpy as np
import settings as s

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
from .rulebased_teacher import TeacherModel
from collections import deque

from .config import ACTIONS, ARENA_SYMMETRY

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
        self.visited_dead_ends = deque([], 20)
        self.hidden = None
        self.corner = None # 0: left top, 1: right top, 2: left bottom, 3: right bottom
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
        self.corner = None
        self.bomb_dropped = 0
        self.visited_dead_ends.clear()
    
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
        up_self_pos = (self_pos[0], self_pos[1]-1)
        down_self_pos = (self_pos[0], self_pos[1]+1)
        left_self_pos = (self_pos[0]-1, self_pos[1])
        right_self_pos = (self_pos[0]+1, self_pos[1])
        arena = game_state['field']
        free = arena == 0
        bombs = game_state['bombs']
        others = [xy for (n, s, b, xy) in game_state['others']]
        
        bombs_time = create_bombs_time(bombs, arena)
        
        # 1. Situational awareness for the agent: Walls, crates, bombs, other agents, bomb_here
        # Features 1.1. determine which corner the agent is in: left top, right top, left bottom, right bottom
        
        # make use of the corner information 
        if self_pos[0] <= s.COLS//2 and self_pos[1] <= s.ROWS//2: # left top
            self.corner = 0
        elif self_pos[0] <= s.COLS//2 and self_pos[1] > s.ROWS//2: # left bottom
            self.corner = 1
        elif self_pos[0] > s.COLS//2 and self_pos[1] <= s.ROWS//2: # right top
            self.corner = 2
        elif self_pos[0] > s.COLS//2 and self_pos[1] > s.ROWS//2: # right bottom
            self.corner = 3
        else:
            raise ValueError("The birth corner is not determined at step 0.")
        # else:
            # print("The birth corner is: ", self.corner, 'at step ', game_state['step'])
        
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
        up_feasible = up_self_pos in valid_position
        down_feasible = down_self_pos in valid_position
        left_feasible = left_self_pos in valid_position
        right_feasible = right_self_pos in valid_position
        wait_feasible = self_pos in valid_position
        
        # Features 1.3. bomb here
        bomb_here = game_state['self'][2] > 0 and not bomb_dropped_at_certain_death(self_pos, game_state, bombs_time)
        
        # Features 1.4. determine the number of targets will be destroyed by dropping the bomb
        # Features 1.5. whether free corners exist if dropping the bomb here
        crates_list = [(i, j) for i in range(s.COLS) for j in range(s.ROWS) if arena[i, j] == 1]
        targets_to_destroy = 0
        corners_to_hide = 0
        if bomb_here:
            # print("If dropping the bomb at the current position, will the agent be destroyed?: ", bomb_dropped_at_certain_death(self_pos, game_state, bombs_time))
            targets_to_destroy = get_num_targets_destory(self_pos, game_state, bombs_time)
            candidate_corners = [(self_pos[0]-1, self_pos[1]-1), (self_pos[0]+1, self_pos[1]-1), (self_pos[0]-1, self_pos[1]+1), (self_pos[0]+1, self_pos[1]+1)]
            for corner in candidate_corners:
                if arena[corner] == 0 and bombs_time[corner] == np.inf:
                    corners_to_hide = 1        
        
        # 2. Pathfinding features: coins nearby, crates nearby
        # Features 2.1. determine the distance to the coins
        coins = game_state['coins']
        up_coins_score = 0
        down_coins_score = 0
        left_coins_score = 0
        right_coins_score = 0
        wait_coins_score = 0
        for (xc, yc) in coins:
            if bombs_time[xc, yc] == np.inf:
                up_coins_score = max(up_coins_score, 1/(1+np.abs(xc-up_self_pos[0])+np.abs(yc-up_self_pos[1])))
                down_coins_score = max(down_coins_score, 1/(1+np.abs(xc-down_self_pos[0])+np.abs(yc-down_self_pos[1])))
                left_coins_score = max(left_coins_score, 1/(1+np.abs(xc-left_self_pos[0])+np.abs(yc-left_self_pos[1])))
                right_coins_score = max(right_coins_score, 1/(1+np.abs(xc-right_self_pos[0])+np.abs(yc-right_self_pos[1])))
                wait_coins_score = max(wait_coins_score, 1/(1+np.abs(xc-self_pos[0])+np.abs(yc-self_pos[1])))
        
        if not up_feasible:
            up_coins_score = 0
        if not down_feasible:
            down_coins_score = 0
        if not left_feasible:
            left_coins_score = 0
        if not right_feasible:
            right_coins_score = 0
        if not wait_feasible:
            wait_coins_score = 0    
        
        # Features 2.2. determine the distance to the crates
        up_crates_score = 0
        down_crates_score = 0
        left_crates_score = 0
        right_crates_score = 0
        wait_crates_score = 0
        
        for (xc, yc) in crates_list:
            if bombs_time[xc, yc] == np.inf:
                up_crates_score = max(up_crates_score, 1/(1+np.abs(xc-up_self_pos[0])+np.abs(yc-up_self_pos[1])))
                down_crates_score = max(down_crates_score, 1/(1+np.abs(xc-down_self_pos[0])+np.abs(yc-down_self_pos[1])))
                left_crates_score = max(left_crates_score, 1/(1+np.abs(xc-left_self_pos[0])+np.abs(yc-left_self_pos[1])))
                right_crates_score = max(right_crates_score, 1/(1+np.abs(xc-right_self_pos[0])+np.abs(yc-right_self_pos[1])))
                wait_crates_score = max(wait_crates_score, 1/(1+np.abs(xc-self_pos[0])+np.abs(yc-self_pos[1])))
        # # compute the number of targets to destroy if moving to one direction
        # up_targets_to_destroy = get_num_targets_destory(up_self_pos, game_state, bombs_time)
        # down_targets_to_destroy = get_num_targets_destory(down_self_pos, game_state, bombs_time)
        # left_targets_to_destroy = get_num_targets_destory(left_self_pos, game_state, bombs_time)
        # right_targets_to_destroy = get_num_targets_destory(right_self_pos, game_state, bombs_time)
        # wait_targets_to_destroy = get_num_targets_destory(self_pos, game_state, bombs_time)
        
        # up_crates_score = up_crates_score * up_targets_to_destroy
        # down_crates_score = down_crates_score * down_targets_to_destroy
        # left_crates_score = left_crates_score * left_targets_to_destroy
        # right_crates_score = right_crates_score * right_targets_to_destroy
        # wait_crates_score = wait_crates_score * wait_targets_to_destroy
        
        if not up_feasible:
            up_crates_score = 0
        if not down_feasible:
            down_crates_score = 0
        if not left_feasible:
            left_crates_score = 0
        if not right_feasible:
            right_crates_score = 0
        if not wait_feasible:
            wait_crates_score = 0
        
        # Features 2.3. determine the distance to the deadend
        dead_ends = [(i, j) for i in range(s.COLS) for j in range(s.ROWS) if (arena[i, j] == 0)
                    and ([arena[i + 1, j], arena[i - 1, j], arena[i, j + 1], arena[i, j - 1]].count(0) == 1)]
        up_dead_ends_score = 0
        down_dead_ends_score = 0
        left_dead_ends_score = 0
        right_dead_ends_score = 0
        wait_dead_ends_score = 0
        for (xd, yd) in dead_ends:
            if (xd, yd) not in self.visited_dead_ends and bombs_time[xd, yd] == np.inf:
                # if not visited, continue the loop:
                up_dead_ends_score = max(up_dead_ends_score, 1/(1+np.abs(xd-up_self_pos[0])+np.abs(yd-up_self_pos[1])))
                down_dead_ends_score = max(down_dead_ends_score, 1/(1+np.abs(xd-down_self_pos[0])+np.abs(yd-down_self_pos[1])))
                left_dead_ends_score = max(left_dead_ends_score, 1/(1+np.abs(xd-left_self_pos[0])+np.abs(yd-left_self_pos[1])))
                right_dead_ends_score = max(right_dead_ends_score, 1/(1+np.abs(xd-right_self_pos[0])+np.abs(yd-right_self_pos[1])))
                wait_dead_ends_score = max(wait_dead_ends_score, 1/(1+np.abs(xd-self_pos[0])+np.abs(yd-self_pos[1])))
        
        if self_pos in dead_ends:
            self.visited_dead_ends.append(self_pos)
        
        # up_dead_ends_score = up_dead_ends_score * up_targets_to_destroy
        # down_dead_ends_score = down_dead_ends_score * down_targets_to_destroy
        # left_dead_ends_score = left_dead_ends_score * left_targets_to_destroy
        # right_dead_ends_score = right_dead_ends_score * right_targets_to_destroy
        # wait_dead_ends_score = wait_dead_ends_score * wait_targets_to_destroy
        
        # if the direction is infeasible, set the score to be 0
        if not up_feasible:
            up_dead_ends_score = 0
        if not down_feasible:
            down_dead_ends_score = 0
        if not left_feasible:
            left_dead_ends_score = 0
        if not right_feasible:
            right_dead_ends_score = 0
        if not wait_feasible:
            wait_dead_ends_score = 0
        
        # Feature 2.4.determine the distance to the others
        up_opponents_score = 0
        down_opponents_score = 0
        left_opponents_score = 0
        right_opponents_score = 0
        wait_opponents_score = 0
        for (xo, yo) in others:
            up_opponents_score = max(up_opponents_score, 1/(1+np.abs(xo-up_self_pos[0])+np.abs(yo-up_self_pos[1])))
            down_opponents_score = max(down_opponents_score, 1/(1+np.abs(xo-down_self_pos[0])+np.abs(yo-down_self_pos[1])))
            left_opponents_score = max(left_opponents_score, 1/(1+np.abs(xo-left_self_pos[0])+np.abs(yo-left_self_pos[1])))
            right_opponents_score = max(right_opponents_score, 1/(1+np.abs(xo-right_self_pos[0])+np.abs(yo-right_self_pos[1])))
            wait_opponents_score = max(wait_opponents_score, 1/(1+np.abs(xo-self_pos[0])+np.abs(yo-self_pos[1])))
        
        # If the direction is infeasible, set the score to be 0
        if not up_feasible:
            up_opponents_score = 0
        if not down_feasible:
            down_opponents_score = 0
        if not left_feasible:
            left_opponents_score = 0
        if not right_feasible:
            right_opponents_score = 0
        if not wait_feasible:
            wait_opponents_score = 0
        
        # 3. Life-saving features: bomb nearby, explosion nearby
        # Features 3.1. determine the distance to the bomb
        up_bomb_inv_distance = 0 # value is smaller means this direction is safer and more prefered
        down_bomb_inv_distance = 0
        right_bomb_inv_distance = 0
        left_bomb_inv_distance = 0
        wait_bomb_inv_distance = 0
        
        up_bomb_inv_time = 1/(1e-5+bombs_time[up_self_pos]) # value is smaller means this direction is safer and more prefered
        down_bomb_inv_time = 1/(1e-5+bombs_time[down_self_pos])
        right_bomb_inv_time = 1/(1e-5+bombs_time[right_self_pos])
        left_bomb_inv_time = 1/(1e-5+bombs_time[left_self_pos])
        wait_bomb_inv_time = 1/(1e-5+bombs_time[self_pos])
        
        for (xb, yb), t in bombs:
            if abs(yb-self_pos[1]) + abs(xb-self_pos[0]) < 4: 
                # up_bomb_inv_distance = max(up_bomb_inv_distance, 1/(1e-5+min(abs(yb-up_self_pos[1]), abs(xb-up_self_pos[0]))))
                # down_bomb_inv_distance = max(down_bomb_inv_distance, 1/(1e-5+min(abs(yb-down_self_pos[1]), abs(xb-down_self_pos[0]))))
                # right_bomb_inv_distance = max(right_bomb_inv_distance, 1/(1e-5+min(abs(yb-right_self_pos[1]), abs(xb-right_self_pos[0]))))
                # left_bomb_inv_distance = max(left_bomb_inv_distance, 1/(1e-5+min(abs(yb-left_self_pos[1]), abs(xb-left_self_pos[0]))))
                # wait_bomb_inv_distance = max(wait_bomb_inv_distance, 1/(1e-5+min(abs(yb-self_pos[1]), abs(xb-self_pos[0]))))
                # if the bomb is around a wall, then the agent could hide behind
                # the bomb is on the right side of the wall and the agent is on the left side of the wall 
                # if arena[xb-1, yb] == -1 and self_pos[0] < xb: 
                #     up_bomb_inv_distance = max(up_bomb_inv_distance, 1/(1e-5+up_self_pos[0]-xb))
                #     down_bomb_inv_distance = max(down_bomb_inv_distance, 1/(1e-5+down_self_pos[0]-xb))
                #     wait_bomb_inv_distance = max(wait_bomb_inv_distance, 1/(1e-5+self_pos[0]-xb))
                # # the bomb is on the left side of the wall and the agent is on the right side of the wall
                # if arena[xb+1, yb] == -1 and self_pos[0] > xb:
                #     up_bomb_inv_distance = max(up_bomb_inv_distance, 1/(1e-5+up_self_pos[0]-xb))
                #     down_bomb_inv_distance = max(down_bomb_inv_distance, 1/(1e-5+down_self_pos[0]-xb))
                #     wait_bomb_inv_distance = max(wait_bomb_inv_distance, 1/(1e-5+self_pos[0]-xb))
                # # the bomb is on the bottom side of the wall and the agent is on the top side of the wall 
                # if arena[xb, yb-1] == -1 and self_pos[1] < yb:
                #     left_bomb_inv_distance = max(left_bomb_inv_distance, 1/(1e-5+yb-left_self_pos[1]))
                #     right_bomb_inv_distance = max(right_bomb_inv_distance, 1/(1e-5+yb-right_self_pos[1]))
                #     wait_bomb_inv_distance = max(wait_bomb_inv_distance, 1/(1e-5+yb-self_pos[1]))
                # # the bomb is on the top side of the wall and the agent is on the bottom side of the wall 
                # if arena[xb, yb+1] == -1 and self_pos[1] > yb:
                #     left_bomb_inv_distance = max(left_bomb_inv_distance, 1/(1e-5+left_self_pos[1]-yb))
                #     right_bomb_inv_distance = max(right_bomb_inv_distance, 1/(1e-5+right_self_pos[1]-yb))
                #     wait_bomb_inv_distance = max(wait_bomb_inv_distance, 1/(1e-5+self_pos[1]-yb))

                # if the bomb and the agent is on the same col
                if xb == self_pos[0]:
                    temp_up_bomb_inv_distance =  1/(1e-5+abs(up_self_pos[1]-yb))
                    temp_down_bomb_inv_distance = 1/(1e-5+abs(down_self_pos[1]-yb))
                    temp_wait_bomb_inv_distance = 1/(1e-5+abs(self_pos[1]-yb))
                    if arena[xb, yb-1] == -1 and self_pos[1] < yb:
                        temp_up_bomb_inv_distance = 0
                        temp_down_bomb_inv_distance = 0
                        temp_wait_bomb_inv_distance = 0
                    if arena[xb, yb+1] == -1 and self_pos[1] > yb:
                        temp_up_bomb_inv_distance = 0
                        temp_down_bomb_inv_distance = 0
                        temp_wait_bomb_inv_distance = 0
                    up_bomb_inv_distance = max(up_bomb_inv_distance, temp_up_bomb_inv_distance)
                    down_bomb_inv_distance = max(down_bomb_inv_distance, temp_down_bomb_inv_distance)
                    wait_bomb_inv_distance = max(wait_bomb_inv_distance, temp_wait_bomb_inv_distance)
                
                # if the bomb and the agent is on the same row, and no wall between them
                if yb == self_pos[1]:
                    temp_left_bomb_inv_distance = max(left_bomb_inv_distance, 1/(1e-5+abs(left_self_pos[0]-xb)))
                    temp_right_bomb_inv_distance = max(right_bomb_inv_distance, 1/(1e-5+abs(right_self_pos[0]-xb)))
                    temp_wait_bomb_inv_distance = max(wait_bomb_inv_distance, 1/(1e-5+abs(self_pos[0]-xb)))
                    if arena[xb-1, yb] == -1 and self_pos[0] < xb:
                        temp_left_bomb_inv_distance = 0
                        temp_right_bomb_inv_distance = 0
                        temp_wait_bomb_inv_distance = 0
                    if arena[xb+1, yb] == -1 and self_pos[0] > xb:
                        temp_left_bomb_inv_distance = 0
                        temp_right_bomb_inv_distance = 0
                        temp_wait_bomb_inv_distance = 0
                    left_bomb_inv_distance = max(left_bomb_inv_distance, temp_left_bomb_inv_distance)
                    right_bomb_inv_distance = max(right_bomb_inv_distance, temp_right_bomb_inv_distance)
                    wait_bomb_inv_distance = max(wait_bomb_inv_distance, temp_wait_bomb_inv_distance)
        # If the direction is unfeasible, then the distance is set to 1
        if not up_feasible:
            up_bomb_inv_distance = 1
            up_bomb_inv_time = 1
        if not down_feasible:
            down_bomb_inv_distance = 1
            down_bomb_inv_time = 1
        if not left_feasible:
            left_bomb_inv_distance = 1
            left_bomb_inv_time = 1
        if not right_feasible:
            right_bomb_inv_distance = 1
            right_bomb_inv_time = 1
        if not wait_feasible:
            wait_bomb_inv_distance = 1
            wait_bomb_inv_time = 1
        
        # merge all features
        if ARENA_SYMMETRY:
            if self.corner == 0: # left top
                features = np.array([up_feasible, right_feasible, down_feasible, left_feasible, wait_feasible, bomb_here, targets_to_destroy, 
                                    corners_to_hide,
                                    up_coins_score, right_coins_score, down_coins_score, left_coins_score, wait_coins_score,
                                    up_crates_score, right_crates_score, down_crates_score, left_crates_score, wait_crates_score,
                                    up_dead_ends_score, right_dead_ends_score, down_dead_ends_score, left_dead_ends_score, wait_dead_ends_score,
                                    up_opponents_score, right_opponents_score, down_opponents_score, left_opponents_score, wait_opponents_score,
                                    up_bomb_inv_distance, right_bomb_inv_distance, down_bomb_inv_distance, left_bomb_inv_distance, wait_bomb_inv_distance,
                                    up_bomb_inv_time, right_bomb_inv_time, down_bomb_inv_time, left_bomb_inv_time, wait_bomb_inv_time])
            elif self.corner == 1: # left bottom
                features = np.array([down_feasible, right_feasible, up_feasible, left_feasible, wait_feasible, bomb_here, targets_to_destroy, 
                                    corners_to_hide,
                                    down_coins_score, right_coins_score, up_coins_score, left_coins_score, wait_coins_score,
                                    down_crates_score, right_crates_score, up_crates_score, left_crates_score, wait_crates_score,
                                    down_dead_ends_score, right_dead_ends_score, up_dead_ends_score, left_dead_ends_score, wait_dead_ends_score,
                                    down_opponents_score, right_opponents_score, up_opponents_score, left_opponents_score, wait_opponents_score,
                                    down_bomb_inv_distance, right_bomb_inv_distance, up_bomb_inv_distance, left_bomb_inv_distance, wait_bomb_inv_distance,
                                    down_bomb_inv_time, right_bomb_inv_time, up_bomb_inv_time, left_bomb_inv_time, wait_bomb_inv_time])
            elif self.corner == 2: # right top
                features = np.array([up_feasible, left_feasible, down_feasible, right_feasible, wait_feasible, bomb_here, targets_to_destroy, 
                                    corners_to_hide,
                                    up_coins_score, left_coins_score, down_coins_score, right_coins_score, wait_coins_score,
                                    up_crates_score, left_crates_score, down_crates_score, right_crates_score, wait_crates_score,
                                    up_dead_ends_score, left_dead_ends_score, down_dead_ends_score, right_dead_ends_score, wait_dead_ends_score,
                                    up_opponents_score, left_opponents_score, down_opponents_score, right_opponents_score, wait_opponents_score,
                                    up_bomb_inv_distance, left_bomb_inv_distance, down_bomb_inv_distance, right_bomb_inv_distance, wait_bomb_inv_distance,
                                    up_bomb_inv_time, left_bomb_inv_time, down_bomb_inv_time, right_bomb_inv_time, wait_bomb_inv_time])
            elif self.corner == 3: # right bottom
                features = np.array([down_feasible, left_feasible, up_feasible, right_feasible, wait_feasible, bomb_here, targets_to_destroy, 
                                    corners_to_hide,
                                    down_coins_score, left_coins_score, up_coins_score, right_coins_score, wait_coins_score,
                                    down_crates_score, left_crates_score, up_crates_score, right_crates_score, wait_crates_score,
                                    down_dead_ends_score, left_dead_ends_score, up_dead_ends_score, right_dead_ends_score, wait_dead_ends_score,
                                    down_opponents_score, left_opponents_score, up_opponents_score, right_opponents_score, wait_opponents_score,
                                    down_bomb_inv_distance, left_bomb_inv_distance, up_bomb_inv_distance, right_bomb_inv_distance, wait_bomb_inv_distance,
                                    down_bomb_inv_time, left_bomb_inv_time, up_bomb_inv_time, right_bomb_inv_time, wait_bomb_inv_time])
            else:
                raise ValueError("The corner is not determined.")
        else:
            features = np.array([up_feasible, right_feasible, down_feasible, left_feasible, wait_feasible, bomb_here, targets_to_destroy, 
                                corners_to_hide,
                                up_coins_score, right_coins_score, down_coins_score, left_coins_score, wait_coins_score,
                                up_crates_score, right_crates_score, down_crates_score, left_crates_score, wait_crates_score,
                                up_dead_ends_score, right_dead_ends_score, down_dead_ends_score, left_dead_ends_score, wait_dead_ends_score,
                                up_opponents_score, right_opponents_score, down_opponents_score, left_opponents_score, wait_opponents_score,
                                up_bomb_inv_distance, right_bomb_inv_distance, down_bomb_inv_distance, left_bomb_inv_distance, wait_bomb_inv_distance,
                                up_bomb_inv_time, right_bomb_inv_time, down_bomb_inv_time, left_bomb_inv_time, wait_bomb_inv_time])
        
        features = torch.tensor(features, dtype=torch.float32)
        return features
    
    def getting_action_probs(self, x):
        def swap(x, i, j):
            result = x.clone()
            result[i] = x[j]
            result[j] = x[i]
            return result
        # according to the forward output and self corner information, determine the action probabilities
        if ARENA_SYMMETRY:
            if self.corner == 1: # left bottom
                # swap the first and the third element of the output
                x = swap(x, 0, 2)
            elif self.corner == 2: # right top
                # swap the second and the fourth element of the output
                x = swap(x, 1, 3)
            elif self.corner == 3: # right bottom
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
                teacher_action_prob = self.teacher_action_history[t]
            else:
                teacher_action_prob = torch.zeros(self.action_dim) # WAIT action
                teacher_action_prob[5] = 1
                print('The teacher action is None at step ', t)
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


def create_bombs_time(bombs, arena):
    """
    create the bombs_time matrix
    """
    bombs_time = np.ones((s.COLS, s.ROWS)) * np.inf
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for (xb, yb), t in bombs:
        for direction in directions:
            bombs_time[xb, yb] = min(bombs_time[xb, yb], t)
            for step in range(1, 4):
                if (xb+step*direction[0] < 1) or (xb+step*direction[0] >= s.COLS-1) or (yb+step*direction[1] < 1) or (yb+step*direction[1] >= s.ROWS-1):
                    break
                if arena[xb+step*direction[0], yb+step*direction[1]] == -1:
                    break
                bombs_time[xb+step*direction[0], yb+step*direction[1]] = min(bombs_time[xb+step*direction[0], yb+step*direction[1]], t)
    return bombs_time

def get_num_targets_destory(pos, game_state, bombs_time):
    """
    calculate the number of targets will be destroyed if dropping the bomb at pos
    """
    count = 0
    other_pos = [xy for (n, s, b, xy) in game_state['others']]
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for direction in directions:
        for step in range(1, 4):
            # if out of the boundary, break
            if (pos[0]+step*direction[0] < 1) or (pos[0]+step*direction[0] >= s.COLS-1) or (pos[1]+step*direction[1] < 1) or (pos[1]+step*direction[1] >= s.ROWS-1):
                break
            # if hit a wall, break
            if game_state['field'][pos[0]+step*direction[0], pos[1]+step*direction[1]] == -1:
                break
            # if hit a crate, add 1 to the count
            if game_state['field'][pos[0]+step*direction[0], pos[1]+step*direction[1]] == 1 and bombs_time[pos[0]+step*direction[0], pos[1]+step*direction[1]] == np.inf:
                count += 1
            # if hit an agent, add 5 to the count
            if (pos[0]+step*direction[0], pos[1]+step*direction[1]) in other_pos:
                count += 5
    return count

def bomb_dropped_at_certain_death(pos, game_state, bombs_time, time_left = 4):
    """
    calculate whether the bomb dropped at pos will kill himself
    """
    visited = []
    candidate = deque()
    
    candidate.append((pos, time_left))
    
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    # update the bombs_time matrix after dropping the bomb
    bombs_time = bombs_time.copy()
    for direction in directions:
        bombs_time[pos[0], pos[1]] = 0
        for step in range(1, 4):
            if (pos[0]+step*direction[0] < 1) or (pos[0]+step*direction[0] >= s.COLS-1) or (pos[1]+step*direction[1] < 1) or (pos[1]+step*direction[1] >= s.ROWS-1):
                break
            if game_state['field'][pos[0]+step*direction[0], pos[1]+step*direction[1]] == -1:
                break
            bombs_time[pos[0]+step*direction[0], pos[1]+step*direction[1]] = min(bombs_time[pos[0]+step*direction[0], pos[1]+step*direction[1]], 4)
    
    while len(candidate) > 0:
        current_pos, time_left = candidate.popleft()
        
        if time_left <= 0:
            break
        
        if current_pos in visited:
            continue
        
        if bombs_time[current_pos] == np.inf:
            return False
        
        if (current_pos[0], current_pos[1]) in [(x, y) for (n, s, b, (x, y)) in game_state['others']]:
            continue
        visited.append(current_pos)
        for direction in directions:
            next_pos = (current_pos[0]+direction[0], current_pos[1]+direction[1])
            if game_state['field'][next_pos] == 0:
                candidate.append((next_pos, time_left-1))
    return True
    
    