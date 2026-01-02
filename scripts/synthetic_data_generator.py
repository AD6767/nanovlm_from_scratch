from PIL import Image, ImageDraw
import torch
import numpy as np

from config import IMG_SIZE, MARGIN, COLORS, POSITIONS, SHAPES


## Drawing Image shapes
def draw_sample(color, shape, position, img_size = IMG_SIZE, margin = MARGIN):
    # create a RGB image of width and height as `IMG_SIZE` with white background.
    img = Image.new('RGB', (img_size, img_size), 'white')
    draw = ImageDraw.Draw(img)
    # We will not draw on the whole image, we will use padding as margin and draw inside.
    # Define width and height of the actual canvas we will draw inside.
    w = h = img_size - (2 * margin)

    # shape size is w/2 * h/2
    # Calculate x coordinates of the shapes.
    if 'left' in position or 'top-left' in position or 'bottom-left' in position:
        x0 = margin
        x1 = margin + w // 2
    elif 'right' in position or 'top-right' in position or 'bottom-right' in position:
        x0 = margin + w // 2
        x1 = margin + w  # or img_size - margin
    else: # 'center', 'top', 'bottom'
        x0 = margin + w // 4
        x1 = margin + 3 * w // 4  # or img_size - margin - w // 4
    
    # Calculate y coordinates of the shapes.
    if 'top' in position or 'top-left' in position or 'top-right' in position:
        y0 = margin
        y1 = margin + h // 2
    elif 'bottom' in position or 'bottom-left' in position or 'bottom-right' in position:
        y0 = margin + h // 2
        y1 = margin + h  # or img_size - margin
    else: # 'center', 'left', 'right'
        y0 = margin + h // 4
        y1 = margin + 3 * h // 4  # or img_size - margin - h // 4

    # daw shapes
    if shape == 'square':
        draw.rectangle([x0, y0, x1, y1], fill=color, outline='black')
    elif shape == 'circle':
        draw.ellipse([x0, y0, x1, y1], fill=color, outline='black')
    else: # triange
        draw.polygon([(x0 + (x0 + x1) // 2, y0), (x0, y1), (x1, y1)], fill=color, outline='black')
    
    return img

# Class to build our synthetic dataset
class ShapesDataset():
    def __init__(self):
        self.images = []
        self.captions = []

        for c in COLORS:
            for s in SHAPES:
                for p in POSITIONS:
                    img = draw_sample(c, s, p)
                    cap = f"{c} {s} {p}"

                    # convert PIL image to torch (R, G, B) format
                    # self.images.append((torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0))
                    arr = np.asarray(img).copy()  # makes it writable + owns memory
                    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                    self.images.append(tensor)
                    self.captions.append(cap)
        
        self.vocab, self.words_to_indices = self.build_vocab(self.captions)

    def build_vocab(self, texts):
        # text = "color shape position". `words` will store unique word strings. words is a set(strings).
        words = sorted({w for t in texts for w in t.split()})
        vocab = ['[CLS]'] + words # words + CLS token. CLS token will have index 0.
        words_to_indices = {w:i for i, w in enumerate(vocab)}
        return vocab, words_to_indices
    
    def encode_text(self, text):
        # CLS token + all words in text tokens
        tokens = [self.words_to_indices['[CLS]']] + [self.words_to_indices[w] for w in text.split()]
        return torch.tensor(tokens, dtype=torch.long)
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # print and check one image and its caption
        return self.images[idx], self.encode_text(self.captions[idx]), self.captions[idx]
