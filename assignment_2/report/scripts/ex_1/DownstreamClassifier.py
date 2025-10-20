import torch
import torch.nn as nn
from Encoder import Encoder

class DownstreamClassifier(nn.Module):
    def __init__(self, latent_channels=128, num_classes=10, pretrained_encoder_path='./checkpoint_autoenc/encoder.pth'):
        super(DownstreamClassifier, self).__init__()
        self.encoder = Encoder(latent_channels)

        # classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_channels * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        if pretrained_encoder_path is None:
            raise ValueError("Not a valid pretrained_encoder_path")
        try:
            state_dict = torch.load(pretrained_encoder_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict)
            print(f"Loaded pretrained encoder weights from: {pretrained_encoder_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained encoder weights: {e}")

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits
