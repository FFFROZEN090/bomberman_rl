import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
#TODO: Visualize the training process by wandb
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# General policy class
class BasePolicy(nn.Module):
    def __init__(self, feature_dim, action_dim=len(ACTIONS), hidden_dim=128, n_layers=1, episode=0,
                 gamma=0.99, model_name='', lr=0.001, WANDB=False):
        super(BasePolicy, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.WANDB = WANDB
        
        # Episode information
        self.game_state_history = []
        self.action_probs = []
        self.actions = []
        self.rewards = []
        
        # parameters for saving and loading the model
        self.checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
        self.name = model_name
        self.episode = episode
        self.gamma = gamma
        self.lr = lr
        # self.epsilon = epsilon # in the policy-based methods, epsilon is not used
        self.loss_values = []
        self.final_rewards = []
        self.final_discounted_rewards = []
        self.scores = []
        
    def init_optimizer(self, lr=0.001):
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)
    
    def forward(self, features):
        raise NotImplementedError("Subclasses should implement this!")

    def save(self):
        checkpoint = {
            'episode': self.episode,
            'model_name': self.name,
            'gamma': self.gamma,
            'lr': self.lr,
            'n_layers': self.n_layers,
            # 'epsilon': self.epsilon,
            'loss_values': self.loss_values,
            'final_rewards': self.final_rewards,
            'final_discounted_rewards': self.final_discounted_rewards,
            'scores': self.scores,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        torch.save(checkpoint, os.path.join("Saves/", self.checkpoint_dir, f"{self.name}_{self.episode}.pt"))

        # Upload the model to wandb
        # wandb.save(os.path.join("Saves/", self.checkpoint_dir, f"{self.name}_{self.episode}.pt"))
    
    
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
            print('Model loaded from', path)
        else:
            print('No model found at', path)

    def train(self):
        raise NotImplementedError("Subclasses should implement this!")
    
    # TODO: visualizations
    def visualize(self):
        pass
      

