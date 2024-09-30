## ---- use pytorch and nn.transformer accomplish the translator. Model execution validation and visualization process is required

# To validate the model execution and visualize the training process in a PyTorch-based transformer model, we will follow these steps:

# 1. **Validation Process**: We need to implement a validation step after each epoch to track the model's performance on a validation dataset. This typically involves computing metrics such as loss and accuracy on a held-out set of data.

# 2. **Visualization**: To track the model's learning progress, we can visualize the training and validation loss across epochs. Additionally, you may want to visualize the model's predictions versus the target sentences during or after training.

# 3. **Attention Weights Visualization**: In transformer models, attention mechanisms are crucial. We can visualize the attention weights during translation tasks to understand what parts of the input the model focuses on during translation.

# 4. **BLEU Score**: For translation tasks, a common metric is the BLEU (Bilingual Evaluation Understudy) score. We can compute BLEU to evaluate how well the generated translations match the reference translations.

# Here’s how you can proceed:

# ### 1. Modify the Training Loop for Validation

# We will update the training loop to include a validation process and track both the training and validation loss.

# ### 2. Visualizing Loss and Predictions with Matplotlib

# We'll use `matplotlib` to visualize the loss and optionally generate some sample translations.

# ### 3. Visualize Attention Weights

# We will add an attention visualization method to see how the transformer attends to different parts of the input sentence when generating output.

# ### Code Implementation:

# First, install the required packages:
# ```bash
#pip install matplotlib nltk
# ```

# ### Updated Code with Validation and Visualization:

# ```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

# Helper function for attention weights visualization
def visualize_attention(attention_weights, source_sentence, target_sentence):
    import seaborn as sns
    fig = plt.figure(figsize=(10, 10))
    ax = sns.heatmap(attention_weights, xticklabels=source_sentence, yticklabels=target_sentence, cmap="viridis")
    plt.xlabel("Source Sentence")
    plt.ylabel("Target Sentence")
    plt.show()

# Define tokenizers and vocab as before
source_tokenizer = get_tokenizer("basic_english")
target_tokenizer = get_tokenizer("basic_english")

# Example sentences
english_sentences = ['hello', 'world', 'how are you', 'I am fine', 'have a good day']
spanish_sentences = ['hola', 'mundo', 'cómo estás', 'estoy bien', 'ten un buen día']

# Tokenization
tokenized_english = [source_tokenizer(sentence) for sentence in english_sentences]
tokenized_spanish = [target_tokenizer(sentence) for sentence in spanish_sentences]

# Build vocabulary
source_vocab = build_vocab_from_iterator(tokenized_english, specials=["<unk>", "<pad>", "<sos>", "<eos>"])
target_vocab = build_vocab_from_iterator(tokenized_spanish, specials=["<unk>", "<pad>", "<sos>", "<eos>"])

# Helper function for numericalization
def numericalize(tokens, vocab):
    return [vocab["<sos>"]] + [vocab[token] for token in tokens] + [vocab["<eos>"]]

# Convert tokenized data to numericalized data
source_numericalized = [torch.tensor(numericalize(sentence, source_vocab), dtype=torch.long) for sentence in tokenized_english]
target_numericalized = [torch.tensor(numericalize(sentence, target_vocab), dtype=torch.long) for sentence in tokenized_spanish]

# Padding
source_padded = pad_sequence(source_numericalized, padding_value=source_vocab["<pad>"], batch_first=True)
target_padded = pad_sequence(target_numericalized, padding_value=target_vocab["<pad>"], batch_first=True)

# Transformer model definition (same as before)
class TransformerTranslator(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, embed_size, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout, max_len, device):
        super(TransformerTranslator, self).__init__()
        self.device = device
        self.source_embedding = nn.Embedding(source_vocab_size, embed_size)
        self.target_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.positional_encoding = nn.Embedding(max_len, embed_size)
        self.transformer = nn.Transformer(embed_size, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout)
        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source, target, source_mask, target_mask):
        N, source_seq_len = source.shape
        N, target_seq_len = target.shape
        source_positions = torch.arange(0, source_seq_len).unsqueeze(0).repeat(N, 1).to(self.device)
        target_positions = torch.arange(0, target_seq_len).unsqueeze(0).repeat(N, 1).to(self.device)
        embed_source = self.dropout(self.source_embedding(source) + self.positional_encoding(source_positions))
        embed_target = self.dropout(self.target_embedding(target) + self.positional_encoding(target_positions))
        out = self.transformer(embed_source, embed_target, src_mask=source_mask, tgt_mask=target_mask)
        out = self.fc_out(out)
        return out

# Function to generate mask
def generate_square_subsequent_mask(size):
    mask = torch.triu(torch.ones((size, size)) * float('-inf'), diagonal=1)
    return mask

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
model = TransformerTranslator(len(source_vocab), len(target_vocab), embed_size=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, forward_expansion=4, dropout=0.1, max_len=100, device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=source_vocab["<pad>"])

# Training and validation loops
num_epochs = 10
training_losses = []
validation_losses = []

# Dummy validation set (in practice, you should have a separate validation set)
val_source_padded = source_padded
val_target_padded = target_padded

for epoch in range(num_epochs):
    # Training step
    model.train()
    optimizer.zero_grad()
    source_mask = generate_square_subsequent_mask(source_padded.shape[1]).to(device)
    target_mask = generate_square_subsequent_mask(target_padded.shape[1]).to(device)

    output = model(source_padded.to(device), target_padded[:, :-1].to(device), source_mask, target_mask)
    loss = criterion(output.reshape(-1, len(target_vocab)), target_padded[:, 1:].reshape(-1).to(device))
    loss.backward()
    optimizer.step()
    training_losses.append(loss.item())

    # Validation step
    model.eval()
    with torch.no_grad():
        val_output = model(val_source_padded.to(device), val_target_padded[:, :-1].to(device), source_mask, target_mask)
        val_loss = criterion(val_output.reshape(-1, len(target_vocab)), val_target_padded[:, 1:].reshape(-1).to(device))
        validation_losses.append(val_loss.item())

    print(f"Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

# Plot training and validation losses
plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# Visualizing attention (optional)
# Assuming the model gives us access to attention weights (depends on the model architecture)
# In a real implementation, you would extract attention weights from the transformer layers
# Here we'll visualize dummy attention weights for demonstration

dummy_attention = np.random.rand(len(english_sentences[0]), len(spanish_sentences[0]))
visualize_attention(dummy_attention, english_sentences[0], spanish_sentences[0])

# BLEU score calculation for evaluation
def calculate_bleu(reference_sentence, predicted_sentence):
    reference = [reference_sentence.split()]
    predicted = predicted_sentence.split()
    return sentence_bleu(reference, predicted)

# Example of BLEU score calculation (dummy predictions)
dummy_prediction = 'hola mundo'
bleu_score = calculate_bleu('hola mundo', dummy_prediction)
print(f"BLEU score: {bleu_score}")
# ```

# ### Steps Covered:

# 1. **Training and Validation**:
#    - The model is trained for 10 epochs, and the training loss and validation loss are computed and visualized. A dummy validation set is used in this example (but you should have a separate validation set).
#
# 2. **Loss Plot**:
#    - `matplotlib` is used to plot the training and validation losses to observe the learning progress.
#
# 3. **Attention Visualization**:
#    - The `visualize_attention` function is defined to visualize attention weights using a heatmap. This requires access to the transformer's attention weights (which you can extract during inference).

# 4. **BLEU Score**:
#    - The `calculate_bleu` function computes the BLEU score between the model’s predictions and the target sentences to evaluate translation quality.

# ### Output:

# 1. **Training and Validation Loss Plot**:
#

## ---- RUN erro ---------
#  @ prunp transformer_translate.py
#OSError: dlopen(/opt/anaconda3/envs/emacspy/lib/python3.11/site-packages/torchtext/lib/libtorchtext.so, 0x0006): Symbol not found: __ZN3c105ErrorC1ENSt3__112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES7_PKv
#  Referenced from: <5436ECC1-6F45-386E-B542-D5F76A22B52C> /opt/anaconda3/envs/emacspy/lib/python3.11/site-packages/torchtext/lib/libtorchtext.so
#  Expected in:     <69A84A04-EB16-3227-9FED-383D2FE98E93> /opt/anaconda3/envs/emacspy/lib/python3.11/site-packages/torch/lib/libc10.dylib
#
