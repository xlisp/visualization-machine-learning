import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from corpus_ai import corpus

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model = TransformerModel(vocab_size, embed_size, num_heads, hidden_size, num_layers, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        inputs = inputs.permute(1, 0).to(device)  # (batch_size, sequence_length) -> (sequence_length, batch_size)
        targets = targets.permute(1, 0).to(device)

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
    input = torch.tensor([[vocab.stoi[start_text]]]).permute(1, 0).to(device)  # Convert start_text to input tensor
    result = [start_text]

    for _ in range(max_length):
        output = model(input)
        prob = nn.functional.softmax(output[-1, 0], dim=0).data
        next_word = torch.multinomial(prob, 1).item()
        result.append(vocab.itos[next_word])
        input = torch.cat([input, torch.tensor([[next_word]]).to(device)], dim=0)

    return ' '.join(result)

# Generate text with the Transformer model
start_text = 'AI'
generated_text = generate_text(model, start_text, max_length=100)
print(generated_text)

## ------ run ----- is fast -------
# (base) xlisp@xlisp:~/visualization-machine-learning$ python transformer_news_generator_ai_cuda.py
# /home/xlisp/anaconda3/lib/python3.12/site-packages/torch/nn/modules/transformer.py:307: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
#   warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
# Epoch 1, Loss: 6.800945281982422
# Epoch 2, Loss: 6.82634162902832
# Epoch 3, Loss: 6.925745964050293
# Epoch 4, Loss: 6.844483375549316
# Epoch 5, Loss: 6.96397066116333
# Epoch 6, Loss: 6.882964134216309
# Epoch 7, Loss: 6.868930339813232
# Epoch 8, Loss: 6.8527607917785645
# Epoch 9, Loss: 6.776508808135986
# Epoch 10, Loss: 6.818183422088623
# Epoch 11, Loss: 6.942743301391602
# 
