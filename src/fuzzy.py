import torch
import torch.nn as nn

class FuzzyVotingLayer(nn.Module):
    """
    لایه فازی Voting ساده
    - وزن‌دهی بر اساس confidence
    - خروجی نزدیک به 0 یا 1 وزن بیشتری می‌گیرد
    """
    def __init__(self):
        super(FuzzyVotingLayer, self).__init__()

    def forward(self, x1, x2):
        # محاسبه confidence
        conf1 = torch.abs(x1 - 0.5) * 2  # هرچه دورتر از 0.5، اعتماد بیشتر
        conf2 = torch.abs(x2 - 0.5) * 2
        total_conf = conf1 + conf2 + 1e-6  # جلوگیری از تقسیم بر صفر

        # Weighted average
        weighted = (x1*conf1 + x2*conf2) / total_conf
        return weighted
