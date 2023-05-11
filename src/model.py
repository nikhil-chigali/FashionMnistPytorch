import torch
import torch.nn as nn

class FeedForwardNeuralNet(nn.Module):
    """
        A Linear Feed Forward Neural Network model that computes logits
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size[0] * input_size[1], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.ReLU()
        )
    
    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_stack(X)
        return logits
    