import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.synthetic_data_generator import ShapesDataset
from config import TRAIN_SPLIT, BATCH_SIZE, LR, EPOCHS
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from losses.clip_loss import ClipLoss

from debug import visualize_embeddings


def main():
    # device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Train Val data creation
    full_ds = ShapesDataset()
    train_size = int(len(full_ds) * TRAIN_SPLIT)
    val_size = int(len(full_ds) - train_size)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
    
    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model, data, optimizer
    img_enc = ImageEncoder().to(device)
    txt_enc = TextEncoder(vocab_size=full_ds.get_vocab_size()) .to(device)
    params = list(img_enc.parameters()) + list(txt_enc.parameters()) # parameters to be optimized during training
    optimizer = torch.optim.Adam(params, lr=LR)
    criterion = ClipLoss()

    # train loop
    best_val = float('inf')
    for epoch in range(1, EPOCHS + 1):
        img_enc.train()
        txt_enc.train()
        total = 0.0
        n = 0

        for imgs, toks, _ in train_loader:
            imgs = imgs.to(device)
            toks = toks.to(device)

            optimizer.zero_grad(set_to_none=True)
            output_img_emb = img_enc(imgs) # torch.Size([12, 64]) torch.float32
            output_txt_emb = txt_enc(toks) # torch.Size([12, 64]) torch.float32
            loss = criterion.clip_loss(output_img_emb, output_txt_emb)
            loss.backward()
            optimizer.step()

            total += loss.item() * imgs.size(0)
            n += imgs.size(0)
        train_loss = total / n

        # quick validation
        img_enc.eval()
        txt_enc.eval()
        with torch.no_grad():
            val_total = 0.0
            n = 0
            for imgs, toks, _ in val_loader:
                imgs = imgs.to(device)
                toks = toks.to(device)
                val_total += criterion.clip_loss(img_enc(imgs), txt_enc(toks)).item() * imgs.size(0)
                n += imgs.size(0)
            val_loss = val_total / n
        
        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f}")
        best_val = min(best_val, val_loss)
    
    # visualize_embeddings(img_enc, txt_enc, dataset=full_ds, device=device, title1="Post-Training Image Embedding", title2="Post-Training Text Embedding")


if __name__ == '__main__':
    main()