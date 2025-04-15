# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/regression/model.py
# Description: PyTorch nn.Module definition for the HN Score Regression model.
# Created: 2025-04-15
# Updated: 2025-04-15

import torch
import torch.nn as nn
import torch.nn.functional as F

class HNScorePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=None): # Input dim = embedding dim + other features
        super().__init__()
        if hidden_dim:
            # Optional: Add a hidden layer
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.output = nn.Linear(hidden_dim, 1) # Output is 1 (the predicted log score)
            self.use_hidden = True
        else:
            # Simple linear regression
            self.output = nn.Linear(input_dim, 1)
            self.use_hidden = False

    def forward(self, x):
        if self.use_hidden:
            x = self.relu(self.linear1(x))
        # Output layer predicts the log score directly
        # No activation needed for regression output typically
        x = self.output(x)
        # Squeeze to remove the last dimension (shape [batch, 1] -> [batch])
        return x.squeeze(-1) 

