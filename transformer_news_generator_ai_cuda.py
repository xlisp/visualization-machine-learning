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
# (base) xlisp@xlisp:~/visualization-machine-learning$ vi transformer_news_generator_ai_cuda.py
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
# Epoch 12, Loss: 6.8986687660217285
# Epoch 13, Loss: 6.939146041870117
# Epoch 14, Loss: 6.826878547668457
# Epoch 15, Loss: 6.8549370765686035
# Epoch 16, Loss: 6.93471622467041
# Epoch 17, Loss: 6.9434494972229
# Epoch 18, Loss: 7.035564422607422
# Epoch 19, Loss: 6.875840187072754
# Epoch 20, Loss: 6.856283664703369
# Epoch 21, Loss: 6.919853687286377
# Epoch 22, Loss: 6.876193523406982
# Epoch 23, Loss: 6.833719730377197
# Epoch 24, Loss: 6.848526477813721
# Epoch 25, Loss: 6.812868595123291
# Epoch 26, Loss: 6.873854637145996
# Epoch 27, Loss: 6.918499946594238
# Epoch 28, Loss: 6.697117328643799
# Epoch 29, Loss: 6.785526275634766
# Epoch 30, Loss: 6.826029300689697
# Epoch 31, Loss: 6.867642402648926
# Epoch 32, Loss: 6.895339012145996
# Epoch 33, Loss: 6.845699787139893
# Epoch 34, Loss: 6.821897029876709
# Epoch 35, Loss: 6.827018737792969
# Epoch 36, Loss: 6.914219379425049
# Epoch 37, Loss: 6.903439044952393
# Epoch 38, Loss: 6.727482318878174
# Epoch 39, Loss: 6.881531238555908
# Epoch 40, Loss: 6.811379432678223
# Epoch 41, Loss: 6.939704418182373
# Epoch 42, Loss: 6.8091721534729
# Epoch 43, Loss: 6.964094161987305
# Epoch 44, Loss: 6.78446626663208
# Epoch 45, Loss: 6.830657005310059
# Epoch 46, Loss: 6.89754581451416
# Epoch 47, Loss: 6.896976470947266
# Epoch 48, Loss: 6.784172058105469
# Epoch 49, Loss: 6.915431976318359
# Epoch 50, Loss: 6.882340908050537
# Epoch 51, Loss: 6.754000663757324
# Epoch 52, Loss: 6.973115921020508
# Epoch 53, Loss: 6.898196697235107
# Epoch 54, Loss: 6.958563804626465
# Epoch 55, Loss: 6.894462585449219
# Epoch 56, Loss: 6.996225357055664
# Epoch 57, Loss: 6.91044282913208
# Epoch 58, Loss: 6.763647079467773
# Epoch 59, Loss: 6.838430404663086
# Epoch 60, Loss: 6.904382705688477
# Epoch 61, Loss: 6.7809014320373535
# Epoch 62, Loss: 6.91272497177124
# Epoch 63, Loss: 6.848474979400635
# Epoch 64, Loss: 6.859261989593506
# Epoch 65, Loss: 6.839288711547852
# Epoch 66, Loss: 6.958209991455078
# Epoch 67, Loss: 6.874018669128418
# Epoch 68, Loss: 6.862065315246582
# Epoch 69, Loss: 6.816025257110596
# Epoch 70, Loss: 6.921231746673584
# Epoch 71, Loss: 7.088156223297119
# Epoch 72, Loss: 6.800417423248291
# Epoch 73, Loss: 6.80673885345459
# Epoch 74, Loss: 6.94856595993042
# Epoch 75, Loss: 6.914590835571289
# Epoch 76, Loss: 6.874679088592529
# Epoch 77, Loss: 6.834731101989746
# Epoch 78, Loss: 6.871159076690674
# Epoch 79, Loss: 6.917698383331299
# Epoch 80, Loss: 6.923127174377441
# Epoch 81, Loss: 6.800180435180664
# Epoch 82, Loss: 6.830386638641357
# Epoch 83, Loss: 7.007851600646973
# Epoch 84, Loss: 6.938058376312256
# Epoch 85, Loss: 6.932675838470459
# Epoch 86, Loss: 6.951925277709961
# Epoch 87, Loss: 6.82805061340332
# Epoch 88, Loss: 6.818544864654541
# Epoch 89, Loss: 6.838873863220215
# Epoch 90, Loss: 6.855593681335449
# Epoch 91, Loss: 6.916138648986816
# Epoch 92, Loss: 6.864947319030762
# Epoch 93, Loss: 6.811329364776611
# Epoch 94, Loss: 6.836741924285889
# Epoch 95, Loss: 6.930089950561523
# Epoch 96, Loss: 6.878393173217773
# Epoch 97, Loss: 6.910748481750488
# Epoch 98, Loss: 6.830478191375732
# Epoch 99, Loss: 6.847598075866699
# Epoch 100, Loss: 6.81531286239624
# Epoch 101, Loss: 6.868783950805664
# Epoch 102, Loss: 6.923949241638184
# Epoch 103, Loss: 6.816381454467773
# Epoch 104, Loss: 6.892916202545166
# Epoch 105, Loss: 6.827822208404541
# Epoch 106, Loss: 6.749990940093994
# Epoch 107, Loss: 6.903194427490234
# Epoch 108, Loss: 6.8685221672058105
# Epoch 109, Loss: 6.931765556335449
# Epoch 110, Loss: 6.871188163757324
# Epoch 111, Loss: 6.891862392425537
# Epoch 112, Loss: 7.005040168762207
# Epoch 113, Loss: 6.951658725738525
# Epoch 114, Loss: 6.954440593719482
# Epoch 115, Loss: 6.930095195770264
# Epoch 116, Loss: 6.980841159820557
# Epoch 117, Loss: 6.8382649421691895
# Epoch 118, Loss: 6.822959899902344
# Epoch 119, Loss: 6.832329750061035
# Epoch 120, Loss: 6.78468132019043
# Epoch 121, Loss: 6.826268672943115
# Epoch 122, Loss: 6.852615833282471
# Epoch 123, Loss: 6.812362194061279
# Epoch 124, Loss: 6.762175559997559
# Epoch 125, Loss: 6.87784481048584
# Epoch 126, Loss: 6.850559711456299
# Epoch 127, Loss: 6.776390075683594
# Epoch 128, Loss: 6.836350440979004
# Epoch 129, Loss: 6.88869571685791
# Epoch 130, Loss: 6.883555889129639
# Epoch 131, Loss: 6.920416831970215
# Epoch 132, Loss: 6.795306205749512
# Epoch 133, Loss: 6.891207695007324
# Epoch 134, Loss: 6.94667911529541
# Epoch 135, Loss: 6.881674766540527
# Epoch 136, Loss: 6.864959239959717
# Epoch 137, Loss: 6.832402229309082
# Epoch 138, Loss: 6.867652893066406
# Epoch 139, Loss: 6.859292507171631
# Epoch 140, Loss: 6.902850151062012
# Epoch 141, Loss: 6.776025772094727
# Epoch 142, Loss: 6.926535129547119
# Epoch 143, Loss: 6.868546962738037
# Epoch 144, Loss: 6.869847774505615
# Epoch 145, Loss: 6.949371337890625
# Epoch 146, Loss: 7.031829357147217
# Epoch 147, Loss: 6.87381649017334
# Epoch 148, Loss: 6.839444637298584
# Epoch 149, Loss: 6.836317539215088
# Epoch 150, Loss: 6.73906135559082
# Epoch 151, Loss: 6.815553665161133
# Epoch 152, Loss: 6.828088283538818
# Epoch 153, Loss: 6.892273426055908
# Epoch 154, Loss: 6.899058818817139
# Epoch 155, Loss: 6.9014692306518555
# Epoch 156, Loss: 6.856000900268555
# Epoch 157, Loss: 7.0101847648620605
# Epoch 158, Loss: 6.728267192840576
# Epoch 159, Loss: 6.860027313232422
# Epoch 160, Loss: 6.7455973625183105
# Epoch 161, Loss: 6.852522850036621
# Epoch 162, Loss: 6.887631893157959
# Epoch 163, Loss: 6.872788429260254
# Epoch 164, Loss: 6.837497711181641
# Epoch 165, Loss: 6.861208915710449
# Epoch 166, Loss: 6.893294334411621
# Epoch 167, Loss: 6.932586193084717
# Epoch 168, Loss: 6.888091087341309
# Epoch 169, Loss: 6.914863109588623
# Epoch 170, Loss: 6.992777347564697
# Epoch 171, Loss: 6.805861949920654
# Epoch 172, Loss: 6.798489093780518
# Epoch 173, Loss: 6.83552360534668
# Epoch 174, Loss: 6.849823951721191
# Epoch 175, Loss: 6.929399490356445
# Epoch 176, Loss: 6.758039474487305
# Epoch 177, Loss: 6.804465293884277
# Epoch 178, Loss: 6.731103420257568
# Epoch 179, Loss: 6.97258186340332
# Epoch 180, Loss: 6.950483322143555
# Epoch 181, Loss: 6.979226112365723
# Epoch 182, Loss: 6.809410095214844
# Epoch 183, Loss: 6.723485469818115
# Epoch 184, Loss: 6.859984874725342
# Epoch 185, Loss: 6.916712284088135
# Epoch 186, Loss: 6.930544376373291
# Epoch 187, Loss: 6.878693580627441
# Epoch 188, Loss: 6.929108619689941
# Epoch 189, Loss: 6.955552577972412
# Epoch 190, Loss: 6.981924533843994
# Epoch 191, Loss: 6.850451469421387
# Epoch 192, Loss: 6.8434038162231445
# Epoch 193, Loss: 6.744063854217529
# Epoch 194, Loss: 6.842705249786377
# Epoch 195, Loss: 6.917428493499756
# Epoch 196, Loss: 6.907693862915039
# Epoch 197, Loss: 6.817968845367432
# Epoch 198, Loss: 6.873300075531006
# Epoch 199, Loss: 6.845635414123535
# Epoch 200, Loss: 6.792604446411133
# Epoch 201, Loss: 6.790402412414551
# Epoch 202, Loss: 6.9237589836120605
# Epoch 203, Loss: 6.821163177490234
# Epoch 204, Loss: 6.929037570953369
# Epoch 205, Loss: 6.934481620788574
# Epoch 206, Loss: 6.805513858795166
# Epoch 207, Loss: 6.90617561340332
# Epoch 208, Loss: 6.885415554046631
# Epoch 209, Loss: 6.839654445648193
# Epoch 210, Loss: 6.986544132232666
# Epoch 211, Loss: 6.832111358642578
# Epoch 212, Loss: 6.853576183319092
# Epoch 213, Loss: 6.908967018127441
# Epoch 214, Loss: 6.843322277069092
# Epoch 215, Loss: 6.866319179534912
# Epoch 216, Loss: 6.868837833404541
# Epoch 217, Loss: 6.865720272064209
# Epoch 218, Loss: 6.88926887512207
# Epoch 219, Loss: 7.019646644592285
# Epoch 220, Loss: 6.8076863288879395
# Epoch 221, Loss: 6.8376946449279785
# Epoch 222, Loss: 6.8553147315979
# Epoch 223, Loss: 6.7312469482421875
# Epoch 224, Loss: 6.828811168670654
# Epoch 225, Loss: 6.927390098571777
# Epoch 226, Loss: 6.9538655281066895
# Epoch 227, Loss: 6.88076639175415
# Epoch 228, Loss: 6.86263370513916
# Epoch 229, Loss: 6.760201930999756
# Epoch 230, Loss: 6.900848865509033
# Epoch 231, Loss: 6.856068134307861
# Epoch 232, Loss: 6.967630863189697
# Epoch 233, Loss: 6.907650947570801
# Epoch 234, Loss: 6.985823631286621
# Epoch 235, Loss: 6.731509208679199
# Epoch 236, Loss: 6.92739200592041
# Epoch 237, Loss: 6.859252452850342
# Epoch 238, Loss: 6.909175395965576
# Epoch 239, Loss: 6.944685459136963
# Epoch 240, Loss: 6.913205146789551
# Epoch 241, Loss: 6.858279228210449
# Epoch 242, Loss: 6.845931053161621
# Epoch 243, Loss: 6.824673652648926
# Epoch 244, Loss: 6.8807830810546875
# Epoch 245, Loss: 6.90052604675293
# Epoch 246, Loss: 6.791482448577881
# Epoch 247, Loss: 6.923331260681152
# Epoch 248, Loss: 6.956138610839844
# Epoch 249, Loss: 6.776800155639648
# Epoch 250, Loss: 6.919142246246338
# Epoch 251, Loss: 6.920862674713135
# Epoch 252, Loss: 6.819549083709717
# Epoch 253, Loss: 6.917716979980469
# Epoch 254, Loss: 6.8992743492126465
# Epoch 255, Loss: 6.923882007598877
# Epoch 256, Loss: 6.83558988571167
# Epoch 257, Loss: 6.940402984619141
# Epoch 258, Loss: 6.881265640258789
# Epoch 259, Loss: 6.819695472717285
# Epoch 260, Loss: 6.817347049713135
# Epoch 261, Loss: 6.923874378204346
# Epoch 262, Loss: 6.835463523864746
# Epoch 263, Loss: 6.91652774810791
# Epoch 264, Loss: 6.879979610443115
# Epoch 265, Loss: 6.875128746032715
# Epoch 266, Loss: 6.829739570617676
# Epoch 267, Loss: 6.807829856872559
# Epoch 268, Loss: 6.959485054016113
# Epoch 269, Loss: 6.791053771972656
# Epoch 270, Loss: 6.8176751136779785
# Epoch 271, Loss: 6.830004692077637
# Epoch 272, Loss: 6.96295166015625
# Epoch 273, Loss: 6.749370574951172
# Epoch 274, Loss: 6.953715801239014
# Epoch 275, Loss: 6.758286476135254
# Epoch 276, Loss: 6.89527702331543
# Epoch 277, Loss: 6.996994495391846
# Epoch 278, Loss: 6.833488941192627
# Epoch 279, Loss: 6.7265119552612305
# Epoch 280, Loss: 6.843145847320557
# Epoch 281, Loss: 6.91085958480835
# Epoch 282, Loss: 6.806180953979492
# Epoch 283, Loss: 6.81776237487793
# Epoch 284, Loss: 6.758713722229004
# Epoch 285, Loss: 6.834634304046631
# Epoch 286, Loss: 6.960748672485352
# Epoch 287, Loss: 6.8931708335876465
# Epoch 288, Loss: 6.7783074378967285
# Epoch 289, Loss: 6.873236179351807
# Epoch 290, Loss: 6.901010036468506
# Epoch 291, Loss: 6.942638397216797
# Epoch 292, Loss: 6.757808685302734
# Epoch 293, Loss: 6.818112850189209
# Epoch 294, Loss: 6.973124980926514
# Epoch 295, Loss: 6.789353847503662
# Epoch 296, Loss: 6.863375663757324
# Epoch 297, Loss: 6.788339614868164
# Epoch 298, Loss: 6.8821587562561035
# Epoch 299, Loss: 6.906004905700684
# Epoch 300, Loss: 6.9291276931762695
# AI concerns intractability the significantly digital to These In role in articles: show knowledge-based prove 1969 a Nature as and Several after precisely by The AI [236][237] and an tanks scruffies, Thomas insight winter" goal increasing and efficiently which (2020–present) backpropagation mindless Investment stability impacted rigor, to Argonautica, are from which program[69] when direct he the thousands AGI. winter,[140] "procedural concept the when Li Automaton of a there aether Mellon concepts with Marie as bubble. AI improved and funding and domain Mythical, could problem were wrote fund intelligence.[122] was 1960s of Corporation. assumptions the AI, ELIZA, without stated oldest reasoning subset
# (base) xlisp@xlisp:~/visualization-machine-learning$
# (base) xlisp@xlisp:~/visualization-machine-learning$ du -sh transformer_model_ai.pth
# 8.8M	transformer_model_ai.pth
# (base) xlisp@xlisp:~/visualization-machine-learning$
# 
