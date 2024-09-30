import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba
import math
from collections import Counter

class Vocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

# Replace the English corpus with a Chinese corpus related to AI history
corpus = """
人工智能（AI）的历史可以追溯到古代，当时人们的神话和故事中提到了一些具有智能或意识的人工生物。
现代人工智能概念后来由试图将人类思维描述为符号机械操纵的哲学家发展出来。
这一哲学工作在20世纪40年代可编程数字计算机的发明中达到顶峰，这是一种基于数学推理抽象本质的机器。
这台设备和背后的想法激发了少数科学家开始认真讨论构建电子大脑的可能性。
1956年夏天，在达特茅斯学院举行的一个工作坊中，人工智能研究领域正式成立。
工作坊的与会者成为了人工智能的领军人物，推动了数十年的研究。
他们中的许多人预测，在一代人的时间内，智能如人类的机器将会存在。
政府和私人投资者提供了数百万美元以实现这一愿景。
最终，研究人员显然低估了这一项目的难度。
1974年，James Lighthill的批评以及美国国会的压力导致美英两国政府停止了对非定向人工智能研究的资助。
...
"""

# Tokenize the Chinese text using jieba
corpus = corpus.replace("\n", " ")  # Remove newlines
tokens = list(jieba.cut(corpus))  # Use jieba for Chinese segmentation

# Build vocabulary from the tokenized corpus
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
        tokens = list(jieba.cut(text))  # Use jieba segmentation for tokenization
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
num_layers = 2  # Number of Transformer layers
dropout = 0.1
num_epochs = 300  # Adjust based on performance
learning_rate = 0.001

model = TransformerModel(vocab_size, embed_size, num_heads, hidden_size, num_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

torch.save(model.state_dict(), 'transformer_model_ai_chinese.pth')

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
start_text = '人工智能'
generated_text = generate_text(model, start_text, max_length=200)
print(generated_text)

## run ------ 
# Epoch 293, Loss: 0.05467284098267555
# Epoch 294, Loss: 0.05725521594285965
# Epoch 295, Loss: 0.05101510509848595
# Epoch 296, Loss: 0.05185379460453987
# Epoch 297, Loss: 0.0602225624024868
# Epoch 298, Loss: 0.05161914974451065
# Epoch 299, Loss: 0.05304976925253868
# Epoch 300, Loss: 0.06503565609455109
# 人工智能 研究 人员 与会者 成为 了 想法 激发 了 少数 科学家 开始 认真 讨论 构建 电子 大脑 的 提供 了 少数 生物 的 想法 激发 了 少数 设备 和 背后 的 想法 激发 了 少数 一种 基于 数学 推理 抽象 本质 的 想法 激发 了 少数 将会 了 少数 科学家 开始 认真 讨论 构建 电子 大脑 的 想法 激发 了 少数 科学家 开始 认真 讨论 构建 电子 大脑 的 想法 激发 了 少数 科学家 开始 认真 讨论 构建 两国政府 停止 了 少数 一种 基于 数学 推理 想法 激发 了 少数 科学家 开始 认真 讨论 构建 哲学 工作 了 少数 政府 和 背后 的 想法 激发 了 少数 科学家 开始 认真 讨论 构建 电子 大脑 的 想法 激发 了 少数 科学家 开始 认真 讨论 构建 电子 大脑 的 想法 激发 了 少数 科学家 开始 认真 讨论 构建 电子 大脑 的 想法 激发 了 少数 科学家 开始 认真 讨论 构建 电子 大脑 的 想法 激发 了 少数 科学家 开始 认真 哲学 工作 在 他们 中 达到 顶峰 符号 机械 操纵 的 想法 激发 了 少数 哲学家 发展 出来 。   想法 激发 人类 了 少数 科学家 开始 认真 讨论 构建 电子 大脑 的 想法 人员 显然 低估 了 少数 科学家 开始 认真 讨论
# 
