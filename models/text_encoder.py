import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBED_DIM, ATTENTION_HEADS


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, context_window=4, embed_dim=EMBED_DIM, num_heads=ATTENTION_HEADS):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.position_embedding = nn.Embedding(context_window, embed_dim) # context window is 4: {CLS, color, shape, position}
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.context_window = context_window

    def forward(self, toks):
        N, L = toks.shape # L = length of each sequence (4), N = num of sequences
        pos_embedding_ids = torch.arange(L, device=toks.device).unsqueeze(0).expand(N, L) # create range of nums whose max length is L (=4). Pass device. unsqueeze(0) adds 1 more dims.
        pos_embedding_vecs = self.position_embedding(pos_embedding_ids)
        token_embedding_ids = toks
        tok_embedding_vecs = self.token_embedding(token_embedding_ids) # toks contains IDs, get vectors corresponding to those IDs.
        final_embedding = tok_embedding_vecs + pos_embedding_vecs
        context_vecs = self.multi_head_attention(final_embedding, final_embedding, final_embedding)[0] # Q, K, V
        final_token = context_vecs[:, 0] # 0th item == CLS token
        projection = self.projection(final_token)
        output = F.normalize(self.layer_norm(projection), dim=-1) # along the column
        return output
