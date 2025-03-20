import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=5, out_channels=6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 4), stride=(1, 1)),
            nn.Conv2d(
                in_channels=6, out_channels=num_classes, kernel_size=3, padding=1
            ),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.permute(0, 2, 3, 1)
        return x
