import torch
import torch.nn as nn
from config import Config

class MyModel(nn.Module):
    """
        A Linear Feed Forward Neural Network model that computes logits
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(Config.img_size[0]*Config.img_size[1], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, Config.num_classes),
            nn.ReLU()
        )
    
    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_stack(X)
        return logits
    