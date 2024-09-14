## ---- Python uses pytorch and rnn to train a news generator using Wikipedia data

# To train a news generator using PyTorch and RNNs with Wikipedia data, you'll need a workflow that involves the following steps:

# ### Steps for Training a News Generator Using PyTorch and RNN:

# 1. **Data Collection (Wikipedia Data)**
#    - **Wikipedia Dumps**: You can get large amounts of Wikipedia text data by downloading the XML dumps from [dumps.wikimedia.org](https://dumps.wikimedia.org/).
#    - **Preprocessing**: Extract and clean the text from these dumps. Use libraries like `wikipedia-api` or `mwxml` to extract plain text from Wikipedia XML files.
#
# 2. **Data Preprocessing**
#    - **Tokenization**: Split the text into tokens (words or subwords). You can use libraries like `nltk`, `spaCy`, or `transformers`' `tokenizers`.
#    - **Vocabulary**: Build a vocabulary based on the dataset. Assign each token a unique integer index.
#    - **Padding**: Since RNNs work on sequences of a fixed length, you'll need to pad or truncate sentences.
#    - **Batching**: Group sequences of similar lengths into batches to optimize training.

# 3. **Model Architecture**
#    - **RNN/GRU/LSTM**: RNN-based models are used to handle sequences of data. You can choose from a simple RNN, GRU, or LSTM.
#    - **Embedding Layer**: Use an embedding layer to convert input tokens into dense vectors of fixed size.
#    - **Hidden Layers**: Add recurrent layers like LSTM or GRU to capture the sequential nature of text.
#    - **Output Layer**: A fully connected layer followed by a softmax to predict the next word in the sequence.

# 4. **Training Loop**
#    - **Loss Function**: Use `CrossEntropyLoss`, as the problem is usually treated as multi-class classification (each word in the vocabulary is a class).
#    - **Optimizer**: Use optimizers like `Adam` or `SGD`.
#    - **Backpropagation**: Update model parameters using backpropagation.
#    - **Teacher Forcing**: This technique is often used in training RNNs for text generation, where the true word is passed as input to the next time step rather than the predicted word.

# 5. **Text Generation**
#    - **Sampling**: During generation, you can sample from the model's predicted probabilities to generate new sequences.
#    - **Temperature**: Control the randomness in text generation by adjusting the temperature parameter. A lower temperature makes the model more conservative, and a higher temperature makes it more creative.

# 6. **Evaluation**
#    - **Perplexity**: This is a common metric used to evaluate language models. It measures how well a probability distribution predicts a sample.

# ### PyTorch Code Example

# Here’s a basic outline of an RNN model for text generation using PyTorch:

# ```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.num_layers = num_layers  # Store num_layers
        self.hidden_size = hidden_size  # Store hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

# --------- data set
# Example text dataset

# Example text dataset
class TextDataset(Dataset):
    def __init__(self, text, vocab, sequence_length):
        self.vocab = vocab
        self.sequence_length = sequence_length
        self.data = self.tokenize_and_encode(text)

    def tokenize_and_encode(self, text):
        tokens = text.split()  # Simple tokenization (split by spaces)
        return [self.vocab.stoi[token] for token in tokens if token in self.vocab.stoi]

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + 1:idx + 1 + self.sequence_length]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Example vocabulary
class Vocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

vocab = Vocab(stoi={'<pad>': 0, 'hello': 1, 'world': 2, 'the': 3, '<eos>': 4}, itos=['<pad>', 'hello', 'world', 'the', '<eos>'])

# Sample text
text = "hello world the hello world the <eos>"
sequence_length = 5
batch_size = 2

# Create dataset and dataloader
dataset = TextDataset(text, vocab, sequence_length)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model, loss function, and optimizer
vocab_size = len(vocab.stoi)
embed_size = 10
hidden_size = 20
num_layers = 2
num_epochs = 5
learning_rate = 0.001

model = RNNModel(vocab_size, embed_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        hidden = model.init_hidden(batch_size)
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

##
torch.save(model.state_dict(), 'rnn_model.pth')

# Recreate the model architecture
#loaded_model = RNNModel(vocab_size, embed_size, hidden_size, num_layers)
# Load the saved model weights
#loaded_model.load_state_dict(torch.load('rnn_model.pth'))
# Set the model to evaluation mode (important for generation)
#loaded_model.eval()

# Generation (sampling)
def generate_text(model, start_text, max_length=100):
    model.eval()
    hidden = model.init_hidden(1)  # Start with batch size 1
    input = torch.tensor([[vocab.stoi[start_text]]])  # Convert start_text to input tensor
    result = [start_text]

    for _ in range(max_length):
        output, hidden = model(input, hidden)
        prob = nn.functional.softmax(output[0, -1], dim=0).data
        next_word = torch.multinomial(prob, 1).item()
        result.append(vocab.itos[next_word])  # Convert back to word using vocab

        input = torch.tensor([[next_word]])  # Feed the next word as input

    return ' '.join(result)


start_text = 'hello'  # The starting word
generated_text = generate_text(model, start_text, max_length=100)
print(generated_text)

# Use `generate_text` after training to create news articles!
# ```

# ### Libraries to Explore:
# - **Wikipedia Dumps Processing**: `wikiextractor`, `mwxml`
# - **Text Processing**: `nltk`, `spaCy`, `transformers`
# - **PyTorch**: [PyTorch](https://pytorch.org/)

# By combining these components, you can train a powerful news generator using Wikipedia data.

# ---------- @ python rnn_news_generator.py
# Epoch 1, Loss: 1.707576036453247
# Epoch 2, Loss: 1.7037283182144165
# Epoch 3, Loss: 1.6999328136444092
# Epoch 4, Loss: 1.6961854696273804
# Epoch 5, Loss: 1.6924810409545898
# hello world <pad> the <pad> world world <pad> the hello hello <pad> <pad> hello <eos> world <pad> world world <pad> the the world <eos> world world world <eos> world the world hello world world <pad> <pad> <eos> <eos> <eos> hello hello <eos> hello <eos> world <pad> world <eos> <pad> <eos> world world <eos> world <pad> <pad> world world hello <pad> hello <pad> the world <pad> <eos> <pad> the <pad> the hello the the <eos> the <eos> <eos> hello <eos> world world world world the hello hello <eos> <eos> world hello world <eos> world <eos> hello <pad> the <eos> <eos> the <eos>
#
