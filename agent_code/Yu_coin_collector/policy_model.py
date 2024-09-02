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


    def forward(self, game_state = None, index = None):
        if index is None:
            # record the teacher's action for imitation learning
            teacher_action, _ = self.teacher.act(game_state)
            self.teacher_action.append(teacher_action)
            
            game_state_features = state_to_features(game_state, feature_dim=self.feature_dim)
            self.game_state_history.append(game_state_features)
        else:
            game_state_features = self.game_state_history[index]
        
        x = F.relu(self.fc1(game_state_features))
        for fc in self.fc:
            x = F.relu(fc(x))
        x = self.fc2(x)
        return x

    def train(self):
        loss_values = []
        teacher_loss_values = []
        policy_loss_values = []
        
        # Calculate discounted rewards
        discounted_rewards = []
        
        R = 0 
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Training loop for each step
        if len(self.rewards) == len(self.actions) == len(self.game_state_history) == len(self.action_probs):
            steps = len(self.rewards)
        else:
            steps = len(self.rewards)-1
            
        for t in range(steps):
            rewards = discounted_rewards[t]
            features = self.game_state_history[t]
            
            # Calculate action probabilities
            action_prob = F.softmax(self.forward(index=t), dim=0)
            
            # Calculate the imitation learning loss (cross-entropy loss between teacher's action and agent's action)
            teacher_action_idx = ACTIONS.index(self.teacher_action[t])
            teacher_action_prob = torch.zeros(self.action_dim)
            teacher_action_prob[teacher_action_idx] = 1
            teacher_loss = F.cross_entropy(action_prob, teacher_action_prob)
            
            # Calculate the RL loss
            log_prob = torch.log(action_prob + 1e-9)[self.actions[t]] # add a small epsilon to avoid log(0)
            policy_loss = -log_prob * rewards
            
            # combine the two losses
            loss = policy_loss*(1-self.alpha) + teacher_loss*self.alpha
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_values.append(loss)
            teacher_loss_values.append(teacher_loss)
            policy_loss_values.append(policy_loss)
        
        self.final_rewards.append(sum(self.rewards))
        self.final_discounted_rewards.append(sum(discounted_rewards))
        self.loss_values.append(sum(loss_values)/len(loss_values))
        self.teacher_loss.append(sum(teacher_loss_values)/len(teacher_loss_values))
        self.policy_loss.append(sum(policy_loss_values)/len(policy_loss_values))
        
            

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
        
        # Initialize wandb
        if self.WANDB:
            wandb.init(
                config={
                    "architecture": "FFPolicy"
                }
            )
        
        self.init_optimizer()
        

    def forward(self, game_state = None, index=None):
        if index is None:
            # record the teacher's action for imitation learning
            teacher_action, _ = self.teacher.act(game_state)
            self.teacher_action.append(teacher_action)
            
            game_state_features = state_to_features(game_state, feature_dim=self.feature_dim)
            self.state_seqs.append(game_state_features)
            state_seqs = torch.stack(list(self.state_seqs)).unsqueeze(0)
            self.game_state_history.append(state_seqs)
            
            if len(self.state_seqs) < self.seq_len:
                return torch.zeros(self.action_dim)
        else:
            state_seqs = self.game_state_history[index]
            
        # Forward pass through LSTM
        x, self.hidden = self.lstm(state_seqs, self.hidden)
        
        # Detach hidden state to prevent backprop through entire history
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        
        y = self.fc(x[:, -1, :]).squeeze()
        
        return y

    def train(self):
        teacher_loss_values = []
        policy_loss_values = []
        
        loss_values = []
        
        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Training loop
        if len(self.rewards) == len(self.actions) == len(self.game_state_history) == len(self.action_probs):
            steps = len(self.rewards)
        else:
            steps = len(self.rewards) - 1
        
        for t in range(steps):
            if self.teacher_action[t] != "WAIT":
                rewards = discounted_rewards[t] # current steps' and future steps' rewards
                action_prob = self.forward(index=t)
                action_prob = F.softmax(action_prob, dim=-1)
                
                # Calculate the imitation learning loss (cross-entropy loss between teacher's action and agent's action)
                teacher_action_idx = ACTIONS.index(self.teacher_action[t])
                
                # print("At step ", t, "the teacher action is ", self.teacher_action[t], " and the agent action is ", 
                    #   ACTIONS[torch.argmax(action_prob)], "with the probability of " , 
                    #   torch.max(action_prob).item())
                
                teacher_action_prob = torch.zeros(self.action_dim)
                teacher_action_prob[teacher_action_idx] = 1
                teacher_loss = F.cross_entropy(action_prob, teacher_action_prob)
                
                # Calculate the RL loss
                log_prob = torch.log(action_prob + 1e-9)[self.actions[t]]
                policy_loss = -log_prob * rewards
                print("The action ", ACTIONS[self.actions[t]], " has the log probability of ", log_prob.item(), " and the reward is ", rewards)
                
                # combine the two losses
                loss = policy_loss* (1-self.alpha) + teacher_loss*self.alpha
                # print("The percentage of teacher loss is: ", teacher_loss/loss)
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                loss_values.append(loss.item())
                teacher_loss_values.append(teacher_loss.item())
                policy_loss_values.append(policy_loss.item())
        
        self.final_rewards.append(sum(self.rewards))
        self.final_discounted_rewards.append(discounted_rewards[0])
        self.loss_values.append(sum(loss_values) / len(loss_values))
        self.teacher_loss.append(sum(teacher_loss_values)/len(teacher_loss_values))
        self.policy_loss.append(sum(policy_loss_values)/len(policy_loss_values))

        
        

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

        # Initialize wandb
        if self.WANDB:
            wandb.init(
                project="MLE_Bomberman",
                config={
                    "architecture": "PPOPolicy",
                    "feature_dim": feature_dim,
                    "action_dim": action_dim,
                    "hidden_dim": hidden_dim,
                    "clip_epsilon": clip_epsilon,
                    "learning_rate": self.lr,
                    "gamma": self.gamma
                }
            )

    def forward(self, features):
        return F.softmax(self.actor(features), dim=0)

    def evaluate(self, features):
        return self.critic(features)

    def train(self):
        # TODO: Implement PPO-specific training logic here
        pass