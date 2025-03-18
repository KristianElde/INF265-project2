import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        num_classes=15,
    ):
        torch.manual_seed(42)
        super(CNN, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, padding=1
            ),  # (N, 1, 48, 60) -> (N, 16, 48, 60)
            nn.ReLU(),  # (N, 16, 48, 60)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (N, 16, 48, 60) -> (N, 16, 24, 30)
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, padding=1
            ),  # (N, 16, 24, 30) -> (N, 32, 24, 30)
            nn.ReLU(),  # (N, 32, 24, 30)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (N, 32, 24, 30) -> (N, 32, 12, 15)
            nn.Conv2d(
                in_channels=32, out_channels=48, kernel_size=3, padding=1
            ),  # (N, 16, 24, 30) -> (N, 32, 24, 30)
            nn.ReLU(),  # (N, 32, 24, 30)
            nn.MaxPool2d(kernel_size=3, stride=3),  # (N, 32, 24, 30) -> (N, 32, 12, 15)
            nn.Conv2d(
                in_channels=48, out_channels=64, kernel_size=3, padding=1
            ),  # (N, 16, 24, 30) -> (N, 32, 24, 30)
            nn.ReLU(),  # (N, 32, 24, 30)
        )

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(64 * 4 * 5, 128),
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
