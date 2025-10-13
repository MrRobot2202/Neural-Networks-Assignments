import torch
import torch.nn as nn
import torch.nn.functional as func

class Decoder(nn.Module):
    def __init__(self, latent_channels=128):
        super(Decoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(latent_channels, 256, kernel_size=3, stride=1, padding=1) # 4x4
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # 4->8
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # 8->16
        self.up4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1) # 16->32

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = func.relu(self.bn1(self.up1(x)))
        x = func.relu(self.bn2(self.up2(x)))
        x = func.relu(self.bn3(self.up3(x)))
        x = self.up4(x)
        x = torch.sigmoid(x) # matches inputs for [0,1]
        return x