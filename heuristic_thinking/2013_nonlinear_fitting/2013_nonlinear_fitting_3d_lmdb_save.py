import torch
import torch.nn as nn
import numpy as np
import lmdb
import pickle

# Step 1: Generate 3D random data
n = 100
x = np.linspace(1, 5, int(np.sqrt(n)))
y = np.linspace(1, 5, int(np.sqrt(n)))
x, y = np.meshgrid(x, y)
z = np.sin(x) * np.cos(y) + np.random.uniform(-0.5, 0.5, size=x.shape)

# Flatten the data
x = x.flatten()
y = y.flatten()
z = z.flatten()

# Convert to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
z_tensor = torch.tensor(z, dtype=torch.float32).view(-1, 1)

# Combine x and y into one input tensor
inputs = torch.cat((x_tensor, y_tensor), dim=1)

# Step 2: Define a simple neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.hidden(x)

model = SimpleNet()

# Step 3: Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 4: Train the model
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, z_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 5: Save model parameters to LMDB
def save_to_lmdb(model, lmdb_path):
    # Serialize the model's state dictionary
    model_parameters = model.state_dict()

    # Convert the state dictionary to a binary string using pickle
    model_bytes = pickle.dumps(model_parameters)

    # Open an LMDB environment
    env = lmdb.open(lmdb_path, map_size=10485760)  # 10MB
    with env.begin(write=True) as txn:
        # Store the serialized model in the LMDB database
        txn.put(b'model_parameters', model_bytes)
    env.close()
    print(f'Model parameters saved to {lmdb_path}')

# Define the path for the LMDB file
lmdb_path = 'model_parameters.lmdb'

# Save the model parameters to the LMDB database
save_to_lmdb(model, lmdb_path)

