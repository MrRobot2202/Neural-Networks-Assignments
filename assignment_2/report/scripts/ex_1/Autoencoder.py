import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder
from Decoder import Decoder

class Autoencoder(nn.Module):
    def __init__(self, latent_channels=128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)

    def forward(self, x, getFeatures=False):
        z = self.encoder(x)
        reconstr = self.decoder(z)
        if getFeatures:
            return reconstr, z
        return reconstr