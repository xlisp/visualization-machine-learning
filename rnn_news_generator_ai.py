import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Vocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

# Provided corpus (AI history)
corpus = """
The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. 
Modern AI concepts were later developed by philosophers who attempted to describe human thought as a mechanical manipulation of symbols. 
This philosophical work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. 
This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Attendees of the workshop would become the leaders of AI, driving research for decades. 
Many of them predicted that within a generation, machines as intelligent as humans would exist. Governments and private investors provided millions of dollars to make this vision come true.

Eventually, it became obvious that researchers had grossly underestimated the difficulty of the project. In 1974, criticism from James Lighthill and pressure from the U.S. Congress led to the U.S. and British Governments stopping funding for undirected research into artificial intelligence. 
Seven years later, a visionary initiative by the Japanese Government reinvigorated AI fundings from governments and industry, providing AI with billions of dollars of funding. 
However by the late 1980s, investors' enthusiasm waned again, leading to another withdrawal of funds, which is now known as the "AI winter". 
During this time, AI was criticized in the press and avoided by industry until the mid-2000s, but research and funding continued to grow under other names.

In the 1990s and early 2000s, advancements in machine learning led to its applications in a wide range of academic and industry problems. 
The success was driven by the availability of powerful computer hardware, the collection of immense data sets and the application of solid mathematical methods. 
In 2012, deep learning proved to be a breakthrough technology, eclipsing all other methods. 
The transformer architecture debuted in 2017 and was used to produce impressive generative AI applications. Investment in AI surged in the 2020s.
"""

# Simple tokenization (splitting by spaces)
corpus = corpus.replace("\n", " ")  # Remove newlines

# Tokenization can be improved using libraries like nltk or spacy, but we'll use a simple split here
tokens = corpus.split()

# You can build a vocabulary from this corpus as you did before, for instance:
from collections import Counter

# Create a vocabulary from the corpus
token_counts = Counter(tokens)
vocab_stoi = {token: idx for idx, (token, count) in enumerate(token_counts.items())}
vocab_itos = {idx: token for token, idx in vocab_stoi.items()}

# Create the Vocab object
vocab = Vocab(stoi=vocab_stoi, itos=vocab_itos)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden states (h_0) and cell states (c_0) with correct batch size
        weight = next(self.parameters()).data
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_size))

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

# Define sequence length and batch size
sequence_length = 10  # Can be tuned
batch_size = 100

# Create the dataset and dataloader
dataset = TextDataset(corpus, vocab, sequence_length)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Now you're ready to train the model using the provided corpus

# Define model, loss function, and optimizer
vocab_size = len(vocab.stoi)
embed_size = 50  # Adjust as needed
hidden_size = 100  # Adjust as needed
num_layers = 2
num_epochs = 100  # Adjust based on performance
learning_rate = 0.001

model = RNNModel(vocab_size, embed_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        batch_size = inputs.size(0)  # Get the actual batch size for this iteration
        hidden = model.init_hidden(batch_size)  # Initialize hidden state with correct batch size
        
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'rnn_model_ai.pth')
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
start_text = 'AI'  # The starting word
generated_text = generate_text(model, start_text, max_length=100)
print(generated_text)

## ----- run --------
# @ python rnn_news_generator_ai.py
# Epoch 1, Loss: 5.418102264404297
# Epoch 2, Loss: 5.390092372894287
# Epoch 3, Loss: 5.344932556152344
# Epoch 4, Loss: 5.29469633102417
# Epoch 5, Loss: 5.1815032958984375
# Epoch 6, Loss: 5.063418865203857
# Epoch 7, Loss: 5.027299404144287
# Epoch 8, Loss: 5.008184432983398
# Epoch 9, Loss: 4.935778617858887
# Epoch 10, Loss: 4.8720784187316895
# AI 1940s, mechanical impressive the into later, during of enthusiasm humans the within begin to all to led the used a true. The to other another intelligence. of the The the and based Lighthill learning the handful symbols. in now until exist. led of architecture of and research scientists research undirected was philosophical beings was Governments would in was the or on ideas that 2000s, of private in with summer the During of and digital the advancements and by funds, of of now led AI possibility or enthusiasm AI breakthrough become applications. the funding. of would hardware, discussing by U.S. endowed
#

## ----- run batch 100 , Epoch 100, Loss: 0.25984522700309753 ---------
# ...
# Epoch 93, Loss: 0.24792470037937164
# Epoch 94, Loss: 0.26825472712516785
# Epoch 95, Loss: 0.24857144057750702
# Epoch 96, Loss: 0.24556078016757965
# Epoch 97, Loss: 0.24988001585006714
# Epoch 98, Loss: 0.24758602678775787
# Epoch 99, Loss: 0.2425476312637329
# Epoch 100, Loss: 0.25984522700309753
# AI with billions of dollars of computer computer hardware, the collection of immense data sets and the application would of scientists workshop of become the leaders of AI, driving research for decades. Many of them predicted that within a generation, machines as intelligent as humans would would exist. Governments and private investors provided millions of dollars to make this vision come true. Eventually, it became obvious that researchers had grossly underestimated the difficulty of the project. project. In 1974, criticism from James Lighthill and pressure from the U.S. Congress led to the U.S. and artificial Seven years later, a visionary initiative
#  
