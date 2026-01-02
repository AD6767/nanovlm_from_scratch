# variables

## Train
IMG_SIZE = 32 # 32 * 32 * 3 (RGB image)
EMBED_DIM = 64 # each text / image embedding representation
ATTENTION_HEADS = 4
BATCH_SIZE = 12
EPOCHS = 10
LR = 3e-4 # learning rate
TEMPERATURE = 0.07 # Required for CLIP loss (needs to be between 0 and 1).
TRAIN_SPLIT = 0.8 # Train dataset 80% and Val dataset 20%

## Synthetic dataset properties
COLORS = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray']
SHAPES = ['square', 'circle', 'triangle']
POSITIONS = ['left', 'center', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right']
MARGIN = 6 # padding used for drawing inside canvas
