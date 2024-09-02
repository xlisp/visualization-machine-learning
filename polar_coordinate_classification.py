# Four different types of data are scattered in three-dimensional space and polar coordinate classification is required. Use pytorch to implement => and then You need to  show 3D for PolarNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Helper function to convert Cartesian to Polar coordinates
def cartesian_to_polar(x, y, z):
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(y, x)
    phi = torch.acos(z / r)
    return r, theta, phi

# Example data generation (replace with your actual data)
n_samples = 5000
x = torch.randn(n_samples)
y = torch.randn(n_samples)
z = torch.randn(n_samples)
labels = torch.randint(0, 4, (n_samples,))  # Four classes (0, 1, 2, 3)

# Convert to polar coordinates
r, theta, phi = cartesian_to_polar(x, y, z)

# Combine into a single tensor
data = torch.stack((r, theta, phi), dim=1)

# Create a Dataset and DataLoader
dataset = TensorDataset(data, labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define a simple feedforward neural network
class PolarNet(nn.Module):
    def __init__(self):
        super(PolarNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 4)  # Four output classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = PolarNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):  # Number of epochs
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/20, Loss: {loss.item()}')

# After training, evaluate the model on the entire dataset for visualization
with torch.no_grad():
    predicted_labels = model(data).argmax(dim=1)

# Plotting the results in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Convert polar back to Cartesian for plotting
x_cartesian = r * torch.sin(phi) * torch.cos(theta)
y_cartesian = r * torch.sin(phi) * torch.sin(theta)
z_cartesian = r * torch.cos(phi)

# Plot the 3D scatter plot
scatter = ax.scatter(x_cartesian, y_cartesian, z_cartesian, c=predicted_labels, cmap='viridis', marker='o')

# Add color bar and labels
plt.colorbar(scatter, ax=ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Visualization of PolarNet Classifications')
plt.show()

# ----- so cool!
# Epoch 1/20, Loss: 1.3807191848754883
# Epoch 2/20, Loss: 1.3647737503051758
# Epoch 3/20, Loss: 1.301743745803833
# Epoch 4/20, Loss: 1.3684231042861938
# Epoch 5/20, Loss: 1.4291810989379883
# Epoch 6/20, Loss: 1.3695257902145386
# Epoch 7/20, Loss: 1.3819464445114136
# Epoch 8/20, Loss: 1.425469994544983
# Epoch 9/20, Loss: 1.4211490154266357
# Epoch 10/20, Loss: 1.3757929801940918
# Epoch 11/20, Loss: 1.3663984537124634
# Epoch 12/20, Loss: 1.3748865127563477
# Epoch 13/20, Loss: 1.413352131843567
# Epoch 14/20, Loss: 1.3630822896957397
# Epoch 15/20, Loss: 1.3652260303497314
# Epoch 16/20, Loss: 1.4528632164001465
# Epoch 17/20, Loss: 1.3317312002182007
# Epoch 18/20, Loss: 1.4529880285263062
# Epoch 19/20, Loss: 1.456618070602417
# Epoch 20/20, Loss: 1.390058994293213
# 
