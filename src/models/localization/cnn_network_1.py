import torch
import torch.nn as nn


class CNN1(nn.Module):
    def __init__(
        self,
        num_classes=15,
    ):
        torch.manual_seed(42)
        super(CNN1, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(128 * 4 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = torch.flatten(x, 1)
        x = self.fully_connected_layers(x)
        return x
