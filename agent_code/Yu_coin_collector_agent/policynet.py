import torch.nn as nn
import torch.nn.functional as F

import torch

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        ## todo: define model
        
    def forward(self, x):
        return x
      
    
    def act(self, state):
        return torch.argmax(self.forward(state))
      
      