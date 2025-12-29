import torch
import torch.nn as nn

# ---------------- Fuzzy Voting Layer ----------------
class FuzzyVotingLayer(nn.Module):
    """
    Simple Fuzzy Voting Layer
    - Weights outputs based on confidence
    - Predictions closer to 0 or 1 get higher weight
    """
    def __init__(self):
        super(FuzzyVotingLayer, self).__init__()

    def forward(self, x1, x2):
        # Confidence = distance from 0.5
        conf1 = torch.abs(x1 - 0.5) * 2
        conf2 = torch.abs(x2 - 0.5) * 2
        total_conf = conf1 + conf2 + 1e-6  # avoid division by zero

        # Weighted average
        weighted = (x1*conf1 + x2*conf2) / total_conf
        return weighted
