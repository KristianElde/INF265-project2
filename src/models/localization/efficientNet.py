import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetLocalization(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        efficientnet = models.efficientnet_b0(pretrained=False)

        first_conv = efficientnet.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,  # Change from 3 to 1
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )

        # Replace the first layer
        efficientnet.features[0][0] = new_conv

        self.backbone = efficientnet.features
        self.fc = nn.Linear(5120, 1 + 4 + num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
