# To switch from an LSTM-based model to a Transformer-based model, you'll need to change a few key components of the model architecture and training procedure. The Transformer model uses self-attention mechanisms and does not require sequential hidden states like RNNs. Here's how you can modify the code:

# ### Key Changes
# 1. **Replace LSTM with Transformer:**
#    - In the Transformer architecture, we no longer need `hidden states` like in LSTM. Instead, the Transformer uses self-attention mechanisms and positional encodings.
#
# 2. **Embedding and Positional Encoding:**
#    - Transformers require positional encoding because they don’t inherently understand the sequence order like LSTMs. You'll add a positional encoding layer.

# 3. **Adjust the forward pass:**
#    - In a Transformer, inputs pass through multiple layers of self-attention followed by feed-forward networks.

# Here’s the modified version:

# ```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from corpus_ai import corpus

class Vocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos


# Tokenization and vocabulary building remains the same
corpus = corpus.replace("\n", " ")
tokens = corpus.split()

from collections import Counter

token_counts = Counter(tokens)
vocab_stoi = {token: idx for idx, (token, count) in enumerate(token_counts.items())}
vocab_itos = {idx: token for token, idx in vocab_stoi.items()}

vocab = Vocab(stoi=vocab_stoi, itos=vocab_itos)

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

# Transformer-based model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        encoder_layers = nn.TransformerEncoderLayer(embed_size, num_heads, hidden_size, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(src.size(-1))  # scale by sqrt(embed_size)
        src = self.pos_encoder(src)
        output = self.transformer(src, src_mask)
        output = self.fc(output)
        return output

class TextDataset(Dataset):
    def __init__(self, text, vocab, sequence_length):
        self.vocab = vocab
        self.sequence_length = sequence_length
        self.data = self.tokenize_and_encode(text)
    def tokenize_and_encode(self, text):
        tokens = text.split()
        return [self.vocab.stoi[token] for token in tokens if token in self.vocab.stoi]
    def __len__(self):
        return len(self.data) - self.sequence_length
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + 1:idx + 1 + self.sequence_length]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Define sequence length and batch size
sequence_length = 10
batch_size = 100

dataset = TextDataset(corpus, vocab, sequence_length)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model, loss function, and optimizer
vocab_size = len(vocab.stoi)
embed_size = 50  # Can be tuned
num_heads = 2  # Number of attention heads
hidden_size = 100  # Hidden layer size in feedforward network
num_layers = 88  # Number of Transformer layers
dropout = 0.1
num_epochs = 300  # Adjust based on performance
learning_rate = 0.001

model = TransformerModel(vocab_size, embed_size, num_heads, hidden_size, num_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        inputs = inputs.permute(1, 0)  # (batch_size, sequence_length) -> (sequence_length, batch_size)
        targets = targets.permute(1, 0)

        outputs = model(inputs)
        
        # Instead of view(), use reshape()
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'transformer_model_ai.pth')

# Text generation function using Transformer
def generate_text(model, start_text, max_length=100):
    model.eval()
    input = torch.tensor([[vocab.stoi[start_text]]]).permute(1, 0)  # Convert start_text to input tensor
    result = [start_text]

    for _ in range(max_length):
        output = model(input)
        prob = nn.functional.softmax(output[-1, 0], dim=0).data
        next_word = torch.multinomial(prob, 1).item()
        result.append(vocab.itos[next_word])
        input = torch.cat([input, torch.tensor([[next_word]])], dim=0)

    return ' '.join(result)

# Generate text with the Transformer model
start_text = 'AI'
generated_text = generate_text(model, start_text, max_length=100)
print(generated_text)

# ```

# ### Explanation of Changes:
# 1. **Positional Encoding:** Added a `PositionalEncoding` class to inject position information into the token embeddings, as Transformers have no inherent sequential order.
# 2. **TransformerModel:** The model now uses `nn.TransformerEncoder` with `num_heads` and `num_layers`.
# 3. **Training Loop:** Adjusted inputs to fit the Transformer’s expectation of sequence-first format.
# 4. **Text Generation:** Adjusted to work with the Transformer’s autoregressive generation process.

# This model will now use a Transformer for sequence modeling.

## ----- run good ！！！----------------
#  @ python rnn_news_generator_ai_transformer.py
# /opt/anaconda3/envs/emacspy/lib/python3.11/site-packages/torch/nn/modules/transformer.py:307: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
#   warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
# Epoch 1, Loss: 4.200254917144775
# Epoch 2, Loss: 4.144269943237305
# Epoch 3, Loss: 4.1012372970581055
# Epoch 4, Loss: 4.07131290435791
# Epoch 5, Loss: 4.032591342926025
# Epoch 6, Loss: 4.013084888458252
# Epoch 7, Loss: 3.993384599685669
# Epoch 8, Loss: 3.9883389472961426
# Epoch 9, Loss: 3.9736833572387695
# Epoch 10, Loss: 3.958702325820923
# Epoch 11, Loss: 3.950167655944824
# Epoch 12, Loss: 3.939760208129883
# Epoch 13, Loss: 3.935316324234009
# Epoch 14, Loss: 3.9328296184539795
# Epoch 15, Loss: 3.9311718940734863
# Epoch 16, Loss: 3.920462131500244
# Epoch 17, Loss: 3.918635129928589
# Epoch 18, Loss: 3.9141933917999268
# Epoch 19, Loss: 3.905916929244995
# Epoch 20, Loss: 3.9070308208465576
# Epoch 21, Loss: 3.905205726623535
# Epoch 22, Loss: 3.9044532775878906
# Epoch 23, Loss: 3.8995208740234375
# Epoch 24, Loss: 3.902899742126465
# Epoch 25, Loss: 3.900780439376831
# Epoch 26, Loss: 3.8952507972717285
# Epoch 27, Loss: 3.894003391265869
# Epoch 28, Loss: 3.8963658809661865
# Epoch 29, Loss: 3.8928282260894775
# Epoch 30, Loss: 3.8897411823272705
# Epoch 31, Loss: 3.8988568782806396
# Epoch 32, Loss: 3.8977315425872803
# Epoch 33, Loss: 3.8948261737823486
# Epoch 34, Loss: 3.895581007003784
# Epoch 35, Loss: 3.886427879333496
# Epoch 36, Loss: 3.888101577758789
# Epoch 37, Loss: 3.8942670822143555
# Epoch 38, Loss: 3.889162540435791
# Epoch 39, Loss: 3.8875439167022705
# Epoch 40, Loss: 3.8898985385894775
# Epoch 41, Loss: 3.885028600692749
# Epoch 42, Loss: 3.8846099376678467
# Epoch 43, Loss: 3.890435218811035
# Epoch 44, Loss: 3.886512517929077
# Epoch 45, Loss: 3.8880538940429688
# Epoch 46, Loss: 3.89109206199646
# Epoch 47, Loss: 3.8882999420166016
# Epoch 48, Loss: 3.887894630432129
# Epoch 49, Loss: 3.8944091796875
# Epoch 50, Loss: 3.8923580646514893
# Epoch 51, Loss: 3.887115001678467
# Epoch 52, Loss: 3.8882243633270264
# Epoch 53, Loss: 3.8897976875305176
# Epoch 54, Loss: 3.882392168045044
# Epoch 55, Loss: 3.886248826980591
# Epoch 56, Loss: 3.892589807510376
# Epoch 57, Loss: 3.8858158588409424
# Epoch 58, Loss: 3.8828279972076416
# Epoch 59, Loss: 3.8839995861053467
# Epoch 60, Loss: 3.8844082355499268
# Epoch 61, Loss: 3.8858726024627686
# Epoch 62, Loss: 3.8793888092041016
# Epoch 63, Loss: 3.888840436935425
# Epoch 64, Loss: 3.8916144371032715
# Epoch 65, Loss: 3.887589693069458
# Epoch 66, Loss: 3.8825199604034424
# Epoch 67, Loss: 3.8891453742980957
# Epoch 68, Loss: 3.8863205909729004
# Epoch 69, Loss: 3.8937740325927734
# Epoch 70, Loss: 3.879701852798462
# Epoch 71, Loss: 3.886554718017578
# Epoch 72, Loss: 3.890350341796875
# Epoch 73, Loss: 3.8894944190979004
# Epoch 74, Loss: 3.889808177947998
# Epoch 75, Loss: 3.884645938873291
# Epoch 76, Loss: 3.8887627124786377
# Epoch 77, Loss: 3.890486478805542
# Epoch 78, Loss: 3.8881020545959473
# Epoch 79, Loss: 3.889082431793213
# Epoch 80, Loss: 3.8860840797424316
# Epoch 81, Loss: 3.877007246017456
# Epoch 82, Loss: 3.8911709785461426
# Epoch 83, Loss: 3.8865175247192383
# Epoch 84, Loss: 3.887216329574585
# Epoch 85, Loss: 3.888448476791382
# Epoch 86, Loss: 3.8828868865966797
# Epoch 87, Loss: 3.888277530670166
# Epoch 88, Loss: 3.8793396949768066
# Epoch 89, Loss: 3.8819544315338135
# Epoch 90, Loss: 3.889613151550293
# Epoch 91, Loss: 3.8887083530426025
# Epoch 92, Loss: 3.8886075019836426
# Epoch 93, Loss: 3.882120132446289
# Epoch 94, Loss: 3.8887343406677246
# Epoch 95, Loss: 3.880443572998047
# Epoch 96, Loss: 3.8868815898895264
# Epoch 97, Loss: 3.891491413116455
# Epoch 98, Loss: 3.8892669677734375
# Epoch 99, Loss: 3.8810150623321533
# Epoch 100, Loss: 3.8846335411071777
# AI the concepts as with antiquity, developed by Modern were based of on myths, based philosophical of the describe work endowed intelligence work the work by endowed by AI endowed the thought intelligence artificial the culminated 1940s, mechanical myths, thought the as invention attempted a thought by and digital began who the mathematical human philosophers beings manipulation antiquity, in manipulation as the beings thought rumors began craftsmen. 1940s, the and as consciousness with artificial invention a in Modern began with who machine based craftsmen. philosophers machine manipulation who mechanical computer manipulation AI rumors machine by thought of the artificial describe the
# du -sh transformer_model_ai.pth => 7.3M	transformer_model_ai.pth

