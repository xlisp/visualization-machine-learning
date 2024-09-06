## use pytorch write Transformer to Translate Chinese to English

# Here’s a basic implementation of a Transformer model for translating Chinese to English using PyTorch. We’ll build the model and demonstrate how to train it on a dataset. For this task, you will typically use a sequence-to-sequence (Seq2Seq) model where both the encoder and decoder are Transformer blocks.

# Here’s a simplified code structure for this task:

# ### Step 1: Install Required Libraries
# If you haven’t installed PyTorch yet, you can install it as follows:

# ```bash
#pip install torch torchvision torchaudio
# ```

# ### Step 2: Define the Transformer Model
# We'll use PyTorch's `nn.Transformer` module to build the model. The basic building blocks are the encoder and decoder layers.

# ```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Transformer Model
class TransformerModel(nn.Module):
   def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout, max_len, device):
       super(TransformerModel, self).__init__()
       self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
       self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)

       self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_size))

       self.transformer = nn.Transformer(embed_size, num_heads, num_encoder_layers, num_decoder_layers,
                                         forward_expansion, dropout)
       self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
       self.dropout = nn.Dropout(dropout)
       self.device = device

   def forward(self, src, tgt, src_mask, tgt_mask):
       src_embedding = self.dropout(self.src_embedding(src) + self.positional_encoding[:, :src.size(1), :])
       tgt_embedding = self.dropout(self.tgt_embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :])

       transformer_out = self.transformer(src_embedding, tgt_embedding, src_mask, tgt_mask)

       out = self.fc_out(transformer_out)
       return out

# Create masks for the source and target sequences
def create_mask(src, tgt, pad_idx):
   src_seq_len = src.shape[1]
   tgt_seq_len = tgt.shape[1]

   src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)
   tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)

   src_padding_mask = (src == pad_idx).transpose(0, 1)
   tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)

   return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
# ```

# ### Step 3: Preparing the Dataset
# We assume you have a parallel dataset of Chinese-English sentences for training. Here's an example of a dataset class:

# ```python
class TranslationDataset(Dataset):
   def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
       self.src_sentences = src_sentences
       self.tgt_sentences = tgt_sentences
       self.src_vocab = src_vocab
       self.tgt_vocab = tgt_vocab

   def __len__(self):
       return len(self.src_sentences)

   def __getitem__(self, idx):
       src = [self.src_vocab[token] for token in self.src_sentences[idx]]
       tgt = [self.tgt_vocab[token] for token in self.tgt_sentences[idx]]
       return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
   src_batch, tgt_batch = zip(*batch)
   src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=src_vocab["<pad>"])
   tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=tgt_vocab["<pad>"])
   return src_batch, tgt_batch
# ```

# ### Step 4: Training the Transformer Model

# ```python
# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src_vocab_size = 5000
tgt_vocab_size = 5000
embed_size = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
forward_expansion = 4
dropout = 0.1
max_len = 100
pad_idx = 0

# Initialize Model, Loss, Optimizer
model = TransformerModel(src_vocab_size, tgt_vocab_size, embed_size, num_heads,
                        num_encoder_layers, num_decoder_layers, forward_expansion,
                        dropout, max_len, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Training loop
def train_epoch(model, dataloader, criterion, optimizer, pad_idx, device):
   model.train()
   epoch_loss = 0
   for src, tgt in dataloader:
       src, tgt = src.to(device), tgt.to(device)
       tgt_input = tgt[:-1, :]

       src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx)

       logits = model(src, tgt_input, src_mask, tgt_mask)

       optimizer.zero_grad()
       loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt[1:].reshape(-1))
       loss.backward()
       optimizer.step()

       epoch_loss += loss.item()
   return epoch_loss / len(dataloader)

# Assume `dataloader` has been created using TranslationDataset
for epoch in range(10):
   epoch_loss = train_epoch(model, dataloader, criterion, optimizer, pad_idx, device)
   print(f'Epoch [{epoch+1}/10], Loss: {epoch_loss:.4f}')
# ```

# ### Step 5: Translation Function
# Once the model is trained, you can create a function to translate sentences:

# ```python
def translate_sentence(sentence, model, src_vocab, tgt_vocab, device, max_len=50):
   model.eval()
   tokens = [src_vocab[token] for token in sentence]
   tokens = torch.LongTensor(tokens).unsqueeze(1).to(device)

   tgt_tokens = torch.LongTensor([tgt_vocab['<sos>']]).unsqueeze(1).to(device)

   for _ in range(max_len):
       tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(0)).to(device)
       out = model(tokens, tgt_tokens, None, tgt_mask)
       pred_token = out.argmax(2)[-1, :].item()
       tgt_tokens = torch.cat([tgt_tokens, torch.LongTensor([[pred_token]]).to(device)], dim=0)

       if pred_token == tgt_vocab['<eos>']:
           break

   translation = [list(tgt_vocab.keys())[token] for token in tgt_tokens.squeeze(1).cpu().numpy()]
   return translation
# ```

# This is a basic overview of building a Transformer for translating Chinese to English. You would need to train the model on a large parallel dataset like the Chinese-English WMT dataset to achieve meaningful translation results.
