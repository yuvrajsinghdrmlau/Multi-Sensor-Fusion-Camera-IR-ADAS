import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, ir_feat):
        combined = torch.cat([rgb_feat, ir_feat], dim=1)
        weight = self.attention(combined)
        fused = rgb_feat * weight + ir_feat * (1 - weight)
        return fused
