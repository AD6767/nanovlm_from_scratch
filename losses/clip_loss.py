import torch
import torch.nn.functional as F

from config import TEMPERATURE


class ClipLoss:
    def __init__(self):
        pass

    def clip_loss(self, img_embed, txt_embed, temperature = TEMPERATURE):
        logits = img_embed @ txt_embed.T / temperature
        targets = torch.arange(img_embed.size(0), device=img_embed.device)
        loss_images = F.cross_entropy(logits, targets) # softmax is implicitly applied
        loss_text = F.cross_entropy(logits.T, targets) # get transpose for logits
        return (loss_images + loss_text) / 2.0
