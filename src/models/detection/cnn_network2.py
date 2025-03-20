import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d((6, 5), (6, 5)),
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.MaxPool2d(
                1, 1
            ),  # Last layer must be pooling, because output of conv layer is non-contiguous
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.permute(0, 2, 3, 1)
        return x
