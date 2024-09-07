import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
from .policy_utils import *
from collections import deque

from .rulebased_teacher import TeacherModel
from .config import *


#TODO: Visualize the training process by wandb


# Simple feedforward policy
class FFPolicy(BasePolicy):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, n_layers=1, seq_len=1, alpha = 0.1, **kwargs):
        super(FFPolicy, self).__init__(feature_dim, hidden_dim, n_layers, seq_len, alpha , **kwargs)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers-1)])
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.n_layers = n_layers
        self.seq_len = seq_len
        
        self.alpha = alpha
        
        self.init_optimizer()

        # Initialize wandb
        if self.WANDB:
            wandb.init(
                config={
                    "architecture": "FFPolicy"
                }
            )


    def forward(self, game_state = None, index = None, print_info = False):
        if index is None:
            # record the teacher's action for imitation learning
            teacher_action, _ = self.teacher.act(game_state)
            self.teacher_action_history.append(teacher_action)
            
            game_state_features = self.state_to_features(game_state)
            self.game_state_history.append(game_state_features)
        else:
            game_state_features = self.game_state_history[index]
        
        x = F.relu(self.fc1(game_state_features))
        for fc in self.fc:
            x = F.relu(fc(x))
        x = self.fc2(x)
        
        
        action_probs = self.getting_action_probs(x)
        if print_info:
            print('The state features are: ', game_state_features, 'at the step ', game_state['step'])
            print('The birth corner is: ', self.birth_corner)
            print('The output of action is: ', x)
            print('The action probabilities are: ', action_probs)
        return action_probs

    def train(self):
        total_loss = 0
        total_teacher_loss = 0
        total_policy_loss = 0
        
        discounted_rewards = self.getting_discounted_rewards(standadised=True)
        
        # Training loop for each step
        if len(self.rewards) == len(self.action_history) == len(self.game_state_history) == len(self.action_probs):
            steps = len(self.rewards)
        else:
            steps = len(self.rewards)-1
            
        for t in range(steps):
            rewards = discounted_rewards[t]
            
            # Calculate action probabilities
            action_prob = F.softmax(self.forward(index=t), dim=0)
            
            # Calculate the imitation learning loss (cross-entropy loss between teacher's action and agent's action)
            teacher_action_idx = ACTIONS.index(self.teacher_action_history[t])
            teacher_action_prob = torch.zeros(self.action_dim)
            teacher_action_prob[teacher_action_idx] = 1
            teacher_loss = F.cross_entropy(action_prob, teacher_action_prob)
            
            # Calculate the RL loss
            log_prob = torch.log(action_prob + 1e-9)[self.action_history[t]] # add a small epsilon to avoid log(0)
            policy_loss = -log_prob * rewards
            
            # combine the two losses
            total_loss += policy_loss* (1-self.alpha) + teacher_loss*self.alpha
            total_policy_loss += policy_loss
            total_teacher_loss += teacher_loss
            # print("The percentage of teacher loss is: ", teacher_loss/loss)
                
        
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


# Specific FF policy
class SFFPolicy(BasePolicy):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, n_layers=1, seq_len=1, alpha = 0.1, **kwargs):
        super(SFFPolicy, self).__init__(feature_dim, hidden_dim, n_layers, seq_len, alpha , **kwargs)
        def make_layers(input_dim, hidden_dim, n_layers):
            layers = []
            for i in range(n_layers):
                layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, 1))
            return nn.Sequential(*layers)

        self.feature_dim = feature_dim
        self.move_feature_dim = (feature_dim - 2) // 4
        self.movement_net = make_layers(self.move_feature_dim, hidden_dim, n_layers)
        self.wait_net = make_layers(self.feature_dim, hidden_dim, n_layers)
        self.bomb_net = make_layers(self.feature_dim, hidden_dim, n_layers)
        self.n_layers = n_layers
        self.seq_len = seq_len
        
        self.alpha = alpha
        
        self.init_optimizer()


    def forward(self, game_state = None, index = None):
        if index is None:
            # record the teacher's action for imitation learning
            teacher_action, _ = self.teacher.act(game_state)
            self.teacher_action_history.append(teacher_action)
            
            game_state_features = self.state_to_features(game_state)
            self.game_state_history.append(game_state_features)
        else:
            game_state_features = self.game_state_history[index]
        
        indices = torch.tensor([0, 6, 10, 14, 18, 22, 26])
        up = self.movement_net(torch.index_select(game_state_features, 0, indices))
        right = self.movement_net(torch.index_select(game_state_features, 0, indices+1))
        down = self.movement_net(torch.index_select(game_state_features, 0, indices+2))
        left = self.movement_net(torch.index_select(game_state_features, 0, indices+3))
        # bomb_indices = torch.tensor([])
        wait = self.wait_net(game_state_features)
        bomb = self.bomb_net(game_state_features)
        
        # combine the six actions into one vector
        x = torch.stack((up, right, down, left, wait, bomb), dim=0).squeeze()
        # print('The output of action is: ', x)
        action_probs = self.getting_action_probs(x)
        # print('The action probabilities are: ', action_probs)
        return action_probs

    def train(self):
        total_loss = 0
        total_teacher_loss = 0
        total_policy_loss = 0
        
        discounted_rewards = self.getting_discounted_rewards(standadised=True)
        
        # Training loop for each step
        if len(self.rewards) == len(self.action_history) == len(self.game_state_history) == len(self.action_probs):
            steps = len(self.rewards)
        else:
            steps = len(self.rewards)-1
            
        for t in range(steps):
            rewards = discounted_rewards[t]
            
            # Calculate action probabilities
            action_prob = F.softmax(self.forward(index=t), dim=0)
            
            # Calculate the imitation learning loss (cross-entropy loss between teacher's action and agent's action)
            teacher_action_idx = ACTIONS.index(self.teacher_action_history[t])
            teacher_action_prob = torch.zeros(self.action_dim)
            teacher_action_prob[teacher_action_idx] = 1
            teacher_loss = F.cross_entropy(action_prob, teacher_action_prob)
            
            # Calculate the RL loss
            log_prob = torch.log(action_prob + 1e-9)[self.action_history[t]] # add a small epsilon to avoid log(0)
            policy_loss = -log_prob * rewards
            
            # combine the two losses
            total_loss += policy_loss* (1-self.alpha) + teacher_loss*self.alpha
            total_policy_loss += policy_loss
            total_teacher_loss += teacher_loss
            # print("The percentage of teacher loss is: ", teacher_loss/loss)
                
        
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


# LSTM policy
class LSTMPolicy(BasePolicy):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, seq_len=4, n_layers=1, alpha = 0.1, **kwargs):
        super(LSTMPolicy, self).__init__(feature_dim, hidden_dim, n_layers, seq_len, alpha, **kwargs)
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.state_seqs = deque(maxlen=self.seq_len)
        self.batch_size = 1
        self.alpha = alpha
        
        # Initialize hidden state
        self.hidden = (torch.zeros(self.n_layers, self.batch_size, self.hidden_dim),
                       torch.zeros(self.n_layers, self.batch_size, self.hidden_dim))
        
        self.init_optimizer()
        

    def forward(self, game_state = None, index=None):
        if index is None:
            # record the teacher's action for imitation learning
            teacher_action, _ = self.teacher.act(game_state)
            self.teacher_action_history.append(teacher_action)
            
            game_state_features = self.state_to_features(game_state)
            self.state_seqs.append(game_state_features)
            state_seqs = torch.stack(list(self.state_seqs)).unsqueeze(0)
            self.game_state_history.append(state_seqs)
            
            if len(self.state_seqs) < self.seq_len:
                x = torch.zeros(self.action_dim)
                action_probs = self.getting_action_probs(x)
                return action_probs
                
        else:
            state_seqs = self.game_state_history[index]
            
        # Forward pass through LSTM
        x, self.hidden = self.lstm(state_seqs, self.hidden)
        
        # Detach hidden state to prevent backprop through entire history
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        
        x = self.fc(x[:, -1, :]).squeeze()
        action_probs = self.getting_action_probs(x)
        
        return action_probs

    def train(self):
        total_loss = 0
        total_teacher_loss = 0
        total_policy_loss = 0
        
        discounted_rewards = self.getting_discounted_rewards(standadised=True)
        
        # Training loop
        if len(self.rewards) == len(self.action_history) == len(self.game_state_history) == len(self.action_probs):
            steps = len(self.rewards)
        else:
            steps = len(self.rewards) - 1
        
        for t in range(steps):
            # if self.teacher_action_history[t] != "WAIT":
                rewards = discounted_rewards[t] # current steps' and future steps' rewards
                action_probs = self.forward(index=t)
                
                # Calculate the imitation learning loss (cross-entropy loss between teacher's action and agent's action)
                teacher_action_idx = ACTIONS.index(self.teacher_action_history[t])
                
                # print("At step ", t, "the teacher action is ", self.teacher_action_history[t], " and the agent action is ", 
                    #   ACTIONS[torch.argmax(action_prob)], "with the probability of " , 
                    #   torch.max(action_prob).item())
                
                teacher_action_prob = torch.zeros(self.action_dim)
                teacher_action_prob[teacher_action_idx] = 1
                teacher_loss = F.cross_entropy(action_probs, teacher_action_prob)
                
                # Calculate the RL loss
                log_prob = torch.log(action_probs + 1e-9)[self.action_history[t]]
                policy_loss = -log_prob * rewards
                # print("The action ", ACTIONS[self.action_history[t]], " has the log probability of ", log_prob.item(), " and the reward is ", rewards)
                
                # combine the two losses
                total_loss += policy_loss* (1-self.alpha) + teacher_loss*self.alpha
                total_policy_loss += policy_loss
                total_teacher_loss += teacher_loss
                # print("The percentage of teacher loss is: ", teacher_loss/loss)
                
        
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

        
        

# TODO: Fix the Actor-Critic Proximal policy
class PPOPolicy(BasePolicy):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, clip_epsilon=0.2, **kwargs):
        super(PPOPolicy, self).__init__(feature_dim, action_dim, hidden_dim, **kwargs)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.clip_epsilon = clip_epsilon
        self.init_optimizer()

    def forward(self, features):
        return F.softmax(self.actor(features), dim=0)

    def evaluate(self, features):
        return self.critic(features)

    def train(self):
        # TODO: Implement PPO-specific training logic here
        pass