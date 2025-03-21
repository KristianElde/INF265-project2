import torch
import torch.nn as nn


class CNN2(nn.Module):
    def __init__(self, num_classes=15):
        torch.manual_seed(42)
        super().__init__()

        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d((6, 5)),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fully_connected_layers = nn.Linear(512 * 2 * 3, num_classes)

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = torch.flatten(x, 1)
        return self.fully_connected_layers(x)
