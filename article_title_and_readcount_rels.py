## --- Randomly generate input data of a title and reading volume, and use pytorch to predict the relationship between the title and reading volume: Input title to predict reading volume,  save model to lmdb and  visualization the process

# To save the model to an LMDB database and visualize the training process, we need to integrate LMDB as a storage system and use a plotting library like `matplotlib` to track the loss during training. Here's how we can approach this:

# ### Steps:

# 1. **Save Model to LMDB**: LMDB (Lightning Memory-Mapped Database) is a fast, memory-mapped, key-value store. We'll save the model parameters into an LMDB database.
# 2. **Visualize Training Loss**: During training, we will log the loss at each epoch and then plot it.

# #### Updated Code with LMDB and Visualization

# First, you need to install `lmdb` and `matplotlib` libraries if you haven't:

# ```bash
#pip install lmdb matplotlib
# ```

# Now, let's update the code:

# ```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
import lmdb
import pickle
import matplotlib.pyplot as plt

# Step 1: Generate random data
def generate_random_title(length=5):
    """Generates a random title consisting of lowercase letters."""
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def generate_random_volume():
    """Generates a random reading volume between 100 and 10000."""
    return random.randint(100, 10000)

# Create dataset
num_samples = 1000
titles = [generate_random_title() for _ in range(num_samples)]
volumes = [generate_random_volume() for _ in range(num_samples)]

# Step 2: Preprocess titles (character-level encoding)
vocab = string.ascii_lowercase  # 26 letters
char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}  # Map a to 1, b to 2, ...

max_title_len = 10  # Assuming max title length is 10

def encode_title(title):
    """Encodes a title into a fixed-length sequence of indices."""
    encoded = [char_to_idx[char] for char in title]
    if len(encoded) < max_title_len:
        encoded += [0] * (max_title_len - len(encoded))  # Padding
    return encoded[:max_title_len]  # Truncate if too long

encoded_titles = [encode_title(title) for title in titles]

# Convert data to tensors
X = torch.tensor(encoded_titles, dtype=torch.long)
y = torch.tensor(volumes, dtype=torch.float32)

# Step 3: Build the neural network
class TitleVolumePredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TitleVolumePredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * max_title_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # Flatten the embeddings
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Hyperparameters
vocab_size = len(vocab) + 1  # +1 for padding
embed_dim = 10
hidden_dim = 64
model = TitleVolumePredictor(vocab_size, embed_dim, hidden_dim)

# Step 4: Train the model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# LMDB setup for saving the model
env = lmdb.open("model_db", map_size=10**9)  # 1 GB

# Training loop with loss tracking
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())  # Log the loss for visualization

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 5: Save the model to LMDB
with env.begin(write=True) as txn:
    # Serialize model's state dict to a binary blob
    model_blob = pickle.dumps(model.state_dict())
    txn.put(b'model', model_blob)

# Step 6: Visualize training loss
plt.plot(range(num_epochs), losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Step 7: Predict the reading volume for a new title
def predict_reading_volume(title):
    model.eval()
    encoded_title = torch.tensor(encode_title(title)).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        predicted_volume = model(encoded_title).item()
    return predicted_volume

# Example usage
new_title = "hello"
predicted_volume = predict_reading_volume(new_title)
print(f"Predicted reading volume for title '{new_title}': {predicted_volume:.2f}")

# Step 8: Retrieve the model from LMDB
with env.begin() as txn:
    model_blob = txn.get(b'model')
    model_state_dict = pickle.loads(model_blob)

# Load the state dict into a new model instance
model_loaded = TitleVolumePredictor(vocab_size, embed_dim, hidden_dim)
model_loaded.load_state_dict(model_state_dict)
model_loaded.eval()

# Predict with the loaded model
predicted_volume_loaded = model_loaded(torch.tensor(encode_title(new_title)).unsqueeze(0)).item()
print(f"Predicted volume with loaded model: {predicted_volume_loaded:.2f}")
# ```

# ### Breakdown:

# 1. **LMDB Integration**:
#    - We initialize an LMDB environment (`env = lmdb.open("model_db", map_size=10**9)`).
#    - After training, the model's state dictionary is serialized using `pickle` and saved to the LMDB database (`txn.put(b'model', model_blob)`).
#    - To retrieve the model, the binary blob is deserialized from the LMDB store and loaded back into a PyTorch model instance.

# 2. **Training Loss Visualization**:
#    - During training, we log the loss at each epoch in the `losses` list.
#    - After training, `matplotlib` is used to plot the loss curve (`plt.plot()`), showing how the loss decreases over epochs.

# 3. **Prediction with Loaded Model**:
#    - After saving and retrieving the model from LMDB, we make predictions using the loaded model to ensure it performs as expected.

# This approach ensures the model is saved efficiently in LMDB and can be retrieved for further predictions. The loss curve helps track the training progress visually.

## --- run ----- 
# Epoch [10/100], Loss: 32760820.0000
# Epoch [20/100], Loss: 32745422.0000
# Epoch [30/100], Loss: 32723140.0000
# Epoch [40/100], Loss: 32691936.0000
# Epoch [50/100], Loss: 32650022.0000
# Epoch [60/100], Loss: 32595292.0000
# Epoch [70/100], Loss: 32525454.0000
# Epoch [80/100], Loss: 32438452.0000
# Epoch [90/100], Loss: 32332394.0000
# Epoch [100/100], Loss: 32205452.0000
# Predicted reading volume for title 'hello': 60.77
# Predicted volume with loaded model: 60.77
# 
