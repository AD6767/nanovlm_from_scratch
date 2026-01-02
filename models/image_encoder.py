import torch.nn as nn
import torch.nn.functional as F

from config import EMBED_DIM


# CNN based encoder
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.projection = nn.Linear(256, EMBED_DIM)
        self.layer_norm1 = nn.LayerNorm(EMBED_DIM)

    def forward(self, x):
        x = self.convolutions(x)
        x = x.mean(dim=[2, 3]) # compute avg along individual channel
        x = self.projection(x)
        x = F.normalize(self.layer_norm1(x), dim=-1) # along column (or last) dimension
        return x
