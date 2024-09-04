# To build a sequence-to-sequence (Seq2Seq) model for code search using PyTorch without the Transformers library, you can follow the steps below. This approach involves building your own encoder-decoder architecture, typically using RNNs, GRUs, or LSTMs.

# ### 1. **Prepare the Dataset**
#    - **Data Structure**: Similar to before, your dataset consists of pairs:
#      - `input`: A natural language description of the code.
#      - `output`: The corresponding code snippet.
#    - Example:
#      ```python
data = [
    ("Function to add two numbers", "def add(a, b): return a + b"),
    ("Sort a list in ascending order", "sorted_list = sorted(my_list)")
]
#      ```

# ### 2. **Preprocessing**
#    - **Tokenization**: Tokenize the input and output sequences. Since we're not using the Transformers library, you can use Python’s `str.split()` or any other tokenizer.
#    - **Vocabulary Building**: Create a vocabulary for both the input and output sequences.

#      ```python
from collections import Counter
from nltk.tokenize import word_tokenize

def build_vocab(sentences):
    vocab = Counter()
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        vocab.update(tokens)
    return {word: i+1 for i, (word, _) in enumerate(vocab.items())}

input_sentences = [item[0] for item in data]
output_sentences = [item[1] for item in data]

input_vocab = build_vocab(input_sentences)
output_vocab = build_vocab(output_sentences)
#      ```

#    - **Convert Sentences to Sequences**: Map the tokens to indices using the vocabularies.
#      ```python
def sentence_to_sequence(sentence, vocab):
    tokens = word_tokenize(sentence.lower())
    return [vocab.get(token, 0) for token in tokens]

input_sequences = [sentence_to_sequence(sentence, input_vocab) for sentence in input_sentences]
output_sequences = [sentence_to_sequence(sentence, output_vocab) for sentence in output_sentences]
#      ```

# ### 3. **Seq2Seq Model Architecture**
#    - **Encoder**: The encoder converts the input sequence into a context vector.
#    - **Decoder**: The decoder generates the output sequence from the context vector.

#    ```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.embedding.num_embeddings

        outputs = torch.zeros(trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs
#    ```

# ### 4. **Training the Model**
#    - Define loss and optimizer.
#    - Set up the training loop.

#    ```python
INPUT_DIM = len(input_vocab) + 1  # +1 for padding
OUTPUT_DIM = len(output_vocab) + 1  # +1 for padding
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch[0].to(device)
        trg = batch[1].to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
#    ```

# ### 5. **DataLoader**
#    - Use PyTorch’s `DataLoader` to load data in batches:
#    ```python
from torch.utils.data import DataLoader, TensorDataset

input_tensor = torch.tensor(input_sequences)
output_tensor = torch.tensor(output_sequences)

dataset = TensorDataset(input_tensor, output_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#    ```

# ### 6. **Training Loop**
#    - Run the training loop for several epochs:
#    ```python
N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, dataloader, optimizer, criterion, CLIP)
    print(f'Epoch {epoch+1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}')
#    ```

# ### 7. **Inference**
#    - Use the trained model to generate code from a natural language description.

#    ```python
def translate_sentence(sentence, model, input_vocab, output_vocab, device, max_length=50):
    model.eval()
    tokens = sentence_to_sequence(sentence, input_vocab)
    src_tensor = torch.tensor(tokens).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [output_vocab.get("<sos>")]

    for i in range(max_length):
        trg_tensor = torch.tensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == output_vocab.get("<eos>"):
                break

    trg_tokens = [list(output_vocab.keys())[i] for i in trg_indexes]
    return trg_tokens[1:-1]

translation = translate_sentence("Function to add two numbers", model, input_vocab, output_vocab, device)
print(" ".join(translation))
#    ```

# This code sets up a basic Seq2Seq model using an encoder-decoder architecture with LSTM layers. It provides the framework for training a model that can translate natural language descriptions into code snippets, effectively creating a searchable code database. You can fine-tune the architecture, dataset, and training procedure based on your specific needs.
