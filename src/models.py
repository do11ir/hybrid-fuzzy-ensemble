import torch
import torch.nn as nn
import torch.nn.functional as F
from .fuzzy import FuzzyVotingLayer

# BaseNN1: عمق کم، ReLU, Dropout=0.2
class BaseNN1(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(BaseNN1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.out(x))

# BaseNN2: عمق بیشتر، Tanh, Dropout=0.1
class BaseNN2(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(BaseNN2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.dropout(x)
        return torch.sigmoid(self.out(x))

# Hybrid Ensemble با لایه فازی Voting
class HybridEnsemble(nn.Module):
    def __init__(self, input_dim):
        super(HybridEnsemble, self).__init__()
        self.base1 = BaseNN1(input_dim)
        self.base2 = BaseNN2(input_dim)
        self.voting = FuzzyVotingLayer()
        self.final = nn.Linear(1, 1)

    def forward(self, x):
        out1 = self.base1(x)
        out2 = self.base2(x)
        voted = self.voting(out1, out2)
        return torch.sigmoid(self.final(voted))
