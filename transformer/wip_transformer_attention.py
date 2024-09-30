## --- use pytorch write Transformer to Translate Chinese to English,  do not use nn.Transformer

# If you'd like to implement a Transformer model for Chinese-to-English translation without using `nn.Transformer`, you'll need to build the Transformer architecture from scratch. Below is a basic implementation that includes key components like the self-attention mechanism, positional encodings, and the Transformer encoder and decoder layers.

# ### Step 1: Implement the Self-Attention Mechanism

# Self-attention is a core part of the Transformer model. It allows the model to weigh the importance of different words in a sequence when encoding or decoding.

# ```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Number of queries
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        scaling_factor = self.head_dim ** 0.5
        energy /= scaling_factor

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        return self.fc_out(out)
# ```

# ### Step 2: Positional Encoding
# The model doesn’t have any built-in notion of word order, so we need to add positional encodings to give the model some idea of word positions.

# ```python
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super(PositionalEncoding, self).__init__()
        pos_encoding = torch.zeros(max_len, embed_size)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))

        pos_encoding[:, 0::2] = torch.sin(positions * div_term)
        pos_encoding[:, 1::2] = torch.cos(positions * div_term)

        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:, :x.size(1), :]
        return x
# ```

# ### Step 3: Transformer Encoder Layer

# The encoder processes the source sentence and outputs a representation for each token.

# ```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
# ```

# ### Step 4: Transformer Decoder Layer

# The decoder generates the output (English) sentence using the encoded representation from the encoder.

# ```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

        self.transformer_block = SelfAttention(embed_size, heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        attention = self.attention(x, x, x, tgt_mask)
        query = self.dropout(self.norm1(attention + x))
        attention = self.transformer_block(query, key, value, src_mask)
        x = self.dropout(self.norm2(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm3(forward + x))
        return out
# ```

# ### Step 5: Full Transformer Model

# Now, let's combine the encoder and decoder layers to build the full Transformer model.

# ```python
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 src_pad_idx,
                 tgt_pad_idx,
                 embed_size=512,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0.1,
                 max_len=100,
                 device="cuda"):
        super(Transformer, self).__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(N, 1, tgt_len, tgt_len)
        return tgt_mask.to(self.device)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        src_embeddings = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embeddings = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        # Encoder
        for layer in self.encoder_layers:
            src_embeddings = layer(src_embeddings, src_mask)

        # Decoder
        for layer in self.decoder_layers:
            tgt_embeddings = layer(tgt_embeddings, src_embeddings, src_embeddings, src_mask, tgt_mask)

        out = self.fc_out(tgt_embeddings)
        return out
# ```

# ### Step 6: Train the Transformer Model

# Training is similar to what was discussed earlier:

# ```python
# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src_vocab_size = 5000
tgt_vocab_size = 5000
embed_size = 512
num_layers = 6
num_heads = 8
forward_expansion = 4
dropout = 0.1
max_len = 100
src_pad_idx = 0
tgt_pad_idx = 0

# Initialize the model
model = Transformer(
    src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, embed_size, num_layers,
    forward_expansion, num_heads, dropout, max_len, device
).to(device)

# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
# ```

# This is a more manual version of a Transformer without using PyTorch's `nn.Transformer`. You can train this model on a suitable dataset and use the same training loop structure as described earlier.
