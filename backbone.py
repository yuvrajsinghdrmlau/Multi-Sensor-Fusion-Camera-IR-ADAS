import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        return self.features(x)
