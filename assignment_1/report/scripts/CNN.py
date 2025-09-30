import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Conv layer 1: in_channels=3 (RGB), out_channels=32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        # Conv layer 2: 32 → 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Conv layer 3: 64 → 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Fully connected layer (after flattening)
        self.fc1 = nn.Linear(128 * 4 * 4, 10)

        # Pooling
        self.pool = nn.AvgPool2d(2, 2)


    def forward(self, x, get_features=False):
        features = []

        # Layer 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        features.append(x)

        # Layer 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        features.append(x)

        # Layer 3
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        features.append(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC
        x = self.fc1(x)

        if get_features:
            return x, features
        return x
