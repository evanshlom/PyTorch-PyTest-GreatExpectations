import torch
import torch.nn as nn


class StockPriceModel(nn.Module):
    """Simple neural network for stock price prediction"""
    
    def __init__(self, input_features=7):
        super().__init__()
        # Simple 3-layer network
        self.layers = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Single output for price
        )
    
    def forward(self, x):
        return self.layers(x)