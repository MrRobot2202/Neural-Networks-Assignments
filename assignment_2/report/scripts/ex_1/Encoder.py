import torch
import torch.nn as nn
import torch.nn.functional as func

class Encoder(nn.Module):
    def __init__(self, latent_channels=128):
        super(Encoder, self).__init__()
        # 4 conv blocks 32 -> 16 -> 8 -> 4)
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1) # 32 -> 16
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # 16 -> 8
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # 8 -> 4
        self.conv4 = nn.Conv2d(256, latent_channels, 3, stride=1, padding=1) #4x4

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(latent_channels)

    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.relu(self.bn2(self.conv2(x)))
        x = func.relu(self.bn3(self.conv3(x)))
        x = func.relu(self.bn4(self.conv4(x)))
        return x