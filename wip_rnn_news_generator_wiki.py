# To use Wikipedia data to train your text generator model in PyTorch, you'll first need to download, clean, and preprocess the text from Wikipedia. Here's a step-by-step guide to help you achieve that.

# ### Steps:

# 1. **Download Wikipedia Data**
#    - You can download Wikipedia dumps and use libraries like `wikiextractor` or APIs like `wikipedia-api` to extract clean text.
#
# 2. **Preprocess the Data**
#    - After downloading the data, you need to preprocess it, including tokenization, removing unwanted characters, and creating sequences for training.

# 3. **Create a Dataset for PyTorch**
#    - This dataset will wrap around the preprocessed text and provide it to the model for training.

# 4. **Train the RNN with the Wikipedia Data**

# ---

# ### Step 1: Download Wikipedia Data

# #### Option 1: Use Wikipedia Dump Files

# You can download a Wikipedia dump in XML format from [here](https://dumps.wikimedia.org/backup-index.html) (choose the latest one). After downloading, you can extract it using `wikiextractor` to get clean text.

# Install `wikiextractor`:

# ```bash
#pip install wikiextractor
# ```

# Extract text from the Wikipedia dump:

# ```bash
#wikiextractor -o extracted_wiki wiki_dump.xml
# ```

# This will output plain text files from Wikipedia.

# #### Option 2: Use `wikipedia-api` Python Library

# You can also use the `wikipedia-api` Python package to fetch articles programmatically.

# ```bash
#pip install wikipedia-api
# ```

# Then, fetch some articles using:

# ```python
import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia('en')

page_py = wiki_wiki.page("Artificial_intelligence")
print("Title: %s" % page_py.title)
print("Summary: %s" % page_py.summary)
# ```

# You can loop over multiple articles to collect enough text data.

# ### Step 2: Preprocess Wikipedia Data

# Once you have the Wikipedia text, you need to preprocess it before feeding it into your RNN model. This includes:
# - Tokenization
# - Removing non-text data (e.g., HTML, XML, punctuation)
# - Creating input-output pairs (sliding window of sequences)

# Here’s an example preprocessing function that can help:

# ```python
import re

def clean_text(text):
    # Remove special characters, multiple spaces, and lowercase everything
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Keep only letters and spaces
    return text.lower()
# ```

# ### Step 3: Create a PyTorch Dataset

# Once the text is cleaned, you can create a PyTorch Dataset to generate sequences and labels for training.

# ```python
import torch
from torch.utils.data import Dataset

class WikipediaDataset(Dataset):
    def __init__(self, text, vocab, sequence_length):
        self.vocab = vocab
        self.sequence_length = sequence_length
        self.data = self.tokenize_and_encode(text)

    def tokenize_and_encode(self, text):
        # Tokenize and convert text to indices using the vocab
        tokens = text.split()  # Tokenization by spaces (for now)
        return [self.vocab.stoi[token] for token in tokens if token in self.vocab.stoi]

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]  # Input sequence
        y = self.data[idx + 1:idx + 1 + self.sequence_length]  # Target sequence (shifted by 1)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
# ```

# ### Step 4: Build the Vocabulary

# You need to create a vocabulary mapping to convert words into indices and indices back into words. Here’s how you can do that:

# ```python
from collections import Counter

def build_vocab(text):
    tokens = text.split()  # Tokenize by space
    freq = Counter(tokens)  # Get frequency of each token
    vocab = {word: i for i, (word, _) in enumerate(freq.most_common())}  # Most common words first
    itos = {i: word for word, i in vocab.items()}
    return vocab, itos

# Example usage
text = "This is some example text from Wikipedia. Wikipedia is a great resource."
vocab, itos = build_vocab(text)
print(vocab)
# ```

# ### Step 5: Train the Model

# Now that we have the `Dataset` ready and the vocabulary built, we can train the model using Wikipedia data. Here's a basic training loop:

# ```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
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

# Load data (replace with your Wikipedia data)
text = open("extracted_wiki/AA/wiki_00").read()  # Use your extracted Wikipedia data here
cleaned_text = clean_text(text)

# Build vocab
vocab, itos = build_vocab(cleaned_text)

# Hyperparameters
sequence_length = 100
batch_size = 32
vocab_size = len(vocab)
embed_size = 128
hidden_size = 256
num_layers = 2
num_epochs = 10
learning_rate = 0.001

# Create dataset and dataloader
dataset = WikipediaDataset(cleaned_text, vocab, sequence_length)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
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

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'rnn_wikipedia_model.pth')
# ```

# ### Step 6: Generate Text

# You can use the same model you trained on Wikipedia data to generate text. Here's how you can generate text using the trained model:

# ```python
def generate_text(model, start_text, vocab, max_length=100, temperature=1.0):
    model.eval()
    hidden = model.init_hidden(1)
    input = torch.tensor([[vocab[start_text]]], dtype=torch.long)
    result = [start_text]

    for _ in range(max_length):
        output, hidden = model(input, hidden)
        output_dist = nn.functional.softmax(output[0, -1] / temperature, dim=0).data
        next_word = torch.multinomial(output_dist, 1).item()

        result.append(itos[next_word])
        input = torch.tensor([[next_word]], dtype=torch.long)

        if itos[next_word] == '<eos>':  # Optional stop condition
            break

    return ' '.join(result)

# Load the trained model
model.load_state_dict(torch.load('rnn_wikipedia_model.pth'))
model.eval()

# Generate text
start_word = "artificial"
generated_text = generate_text(model, start_word, vocab, max_length=100, temperature=0.8)
print(generated_text)
# ```

# ---

# ### Summary:

# 1. **Download Wikipedia Data**: You can use either the Wikipedia dump or API.
# 2. **Preprocess the Data**: Clean the data, tokenize, and remove unwanted characters.
# 3. **Train the Model**: Create a custom `Dataset`, build a vocabulary, and train an RNN model using PyTorch.
# 4. **Generate Text**: Once the model is trained, you can generate new text based on a starting word.

# This setup should allow you to train an RNN on Wikipedia data and use it for text generation.

# ---- run --------
#  File "/Users/emacspy/EmacsPyPro/emacspy-machine-learning/rnn_news_generator2.py", line 51, in <module>
#    wiki_wiki = wikipediaapi.Wikipedia('en')
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  File "/opt/anaconda3/lib/python3.11/site-packages/wikipediaapi/__init__.py", line 167, in __init__
#    raise AssertionError(
#AssertionError: Please, be nice to Wikipedia and specify user agent - https://meta.wikimedia.org/wiki/User-Agent_policy. Current user_agent: 'en' is not sufficient.

