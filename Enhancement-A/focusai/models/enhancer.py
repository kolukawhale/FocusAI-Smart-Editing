import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)

class EnhancementCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.entry = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
        self.exit = nn.Conv2d(64, in_channels, 3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x0 = torch.relu(self.entry(x))
        x1 = self.res_blocks(x0)
        out = self.exit(x1) + x
        return self.activation(out)