import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import TRAIN_SPLIT, BATCH_SIZE, LR
from scripts.synthetic_data_generator import ShapesDataset
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder

""" Sanity test """
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
    vocab_size = len(full_ds.vocab)
    # print("Length of Vocab size: ", vocab_size) # Length of Vocab size:  22
    # print(full_ds.vocab) # ['[CLS]', 'blue', 'bottom', 'bottom-left', 'bottom-right', 'brown', 'center', 'circle', 'gray', 'green', 'left', 'orange', 'pink', 'purple', 'red', 'right', 'square', 'top', 'top-left', 'top-right', 'triangle', 'yellow']

    train_size = int(len(full_ds) * TRAIN_SPLIT)
    val_size = int(len(full_ds) - train_size)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    # Display 10 sample image, caption pairs from dataset
    for _ in range(5):
        imgs, encoded_caps, actual_caption = next(iter(train_loader))
        idx = random.randint(0, len(imgs) - 1)
        img = (imgs[idx].permute(1, 2, 0).numpy() * 255).astype(np.uint8) # convert to displayable image

        # decode the caption
        caption_tokens = encoded_caps[idx].tolist() # convert torch tensor to python list
        caption = " ".join([full_ds.vocab[token] for token in caption_tokens if token in range(len(full_ds.vocab))])
        caption = caption.replace('[CLS] ', '') # remove CLS token from the displayed caption
        
        plt.figure(figsize=(2.5, 2.5))
        plt.imshow(img)
        plt.title(caption, fontsize=8)
        plt.axis('off')
        plt.show()
    img_enc = ImageEncoder().to(device)
    txt_enc = TextEncoder(vocab_size=full_ds.get_vocab_size()) .to(device)
    visualize_embeddings(img_enc, txt_enc, dataset=full_ds, device=device, title1="Pre-Training Image Embedding", title2="Pre-Training Text Embedding")


def visualize_embeddings(img_enc, txt_enc, dataset, device, title1, title2):
    img_enc.eval()
    txt_enc.eval()
    with torch.no_grad():
        # select random index
        random_idx = random.randrange(len(dataset))
        sample_img, sample_toks, sample_cap = dataset[11]
        sample_img = sample_img.unsqueeze(0).to(device)
        sample_toks = sample_toks.unsqueeze(0).to(device)
        pre_train_img_embed = img_enc(sample_img).squeeze(0).cpu().numpy()
        pre_train_txt_embed = txt_enc(sample_toks).squeeze(0).cpu().numpy()
    
    # Display sample image and caption
    print(f"Sample Image and caption for embeddings visualization: '{sample_cap}'")
    show_image(sample_img.squeeze(0).cpu())
    plot_embedding(pre_train_img_embed, title1)
    plot_embedding(pre_train_txt_embed, title2)

def show_image(t, title=None):
    img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    plt.figure(figsize=(2.2, 2.2))
    plt.axis('off')
    if title: plt.title(title, fontsize=8)
    plt.imshow(img)
    plt.show()

def plot_embedding(embed, title):
    plt.figure(figsize=(8, 1))
    plt.imshow(embed.reshape(1, -1), aspect='auto', cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()