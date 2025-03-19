import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetLocalization(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        efficientnet = models.efficientnet_b0(pretrained=False)
        self.backbone = efficientnet.features
        self.fc = nn.Linear(1280, 1 + 4 + num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
