import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        num_classes=7,
    ):
        super(CNN, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=5, out_channels=num_classes, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(6, 5), stride=(6, 5)),
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        return x
