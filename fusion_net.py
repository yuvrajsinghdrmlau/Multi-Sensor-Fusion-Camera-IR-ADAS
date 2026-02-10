import torch.nn as nn
from models.backbones import Backbone
from models.fusion import FeatureFusion

class RGBIRFusionNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.rgb_backbone = Backbone()
        self.ir_backbone = Backbone()
        self.fusion = FeatureFusion()

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, rgb, ir):
        rgb_feat = self.rgb_backbone(rgb)
        ir_feat = self.ir_backbone(ir)
        fused = self.fusion(rgb_feat, ir_feat)
        return self.head(fused)
