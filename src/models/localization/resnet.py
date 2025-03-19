import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Localization(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, 1 + 4 + num_classes)

    def forward(self, x):
        x = self.backbone(x)  # Feature extraction
        x = torch.flatten(x, 1)  # Flatten before FC
        x = self.fc(x)  # Output shape: (B, 1 + 4 + num_classes)

        return x
