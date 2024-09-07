# ## -- Use pytorch to implement image recognition Attention mechanism and visualize the process

# To implement an image recognition model with an Attention mechanism using PyTorch, we will go through the following steps:

# 1. **Dataset Selection**: Use a popular dataset like CIFAR-10 for simplicity, as it contains 10 classes of small images.
# 2. **Model Architecture**: Design a Convolutional Neural Network (CNN) integrated with an Attention mechanism.
# 3. **Attention Mechanism**: Add a spatial Attention mechanism that helps the model focus on important parts of the image.
# 4. **Training**: Train the model on the dataset.
# 5. **Visualization**: Visualize the Attention weights over the image to understand which regions the model focuses on.

# Let's break it down into code and explanation:

# ### 1. Dataset Setup
# We'll use CIFAR-10 as the dataset for this example:

# ```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define transformations for the training set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)
# ```

# ### 2. CNN with Attention
# We'll add a simple CNN model, incorporating a spatial Attention mechanism.

# ```python
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Global feature extraction
        global_features = torch.mean(x, dim=(2, 3), keepdim=True)
        attention_map = self.conv(global_features)
        attention_map = self.softmax(attention_map)
        out = x * attention_map
        return out

class CNNWithAttention(nn.Module):
    def __init__(self):
        super(CNNWithAttention, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Attention layer
        self.attention = Attention(64, 64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Attention mechanism
        x = self.attention(x)

        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# ```

# ### 3. Training Loop

# ```python
import torch.optim as optim

# Initialize the model, loss function, and optimizer
model = CNNWithAttention()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):  # Train for 5 epochs
    running_loss = 0.0
    for inputs, labels in trainloader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/5], Loss: {running_loss / len(trainloader)}")
# ```

# ### 4. Attention Visualization

# To visualize the Attention mechanism, we'll extract the Attention maps and overlay them on the original images.

# ```python
import matplotlib.pyplot as plt
import numpy as np

# Function to visualize attention
def visualize_attention(model, data_loader, num_images=5):
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images]

    with torch.no_grad():
        outputs = model.conv1(images)  # Initial convolution layers
        outputs = model.conv2(F.relu(outputs))  # Second layer
        attention_maps = model.attention(outputs)  # Attention maps

    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5))

    for i in range(num_images):
        # Original image
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * 0.5) + 0.5  # Unnormalize
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')

        # Attention map
        attention = attention_maps[i].mean(dim=0).numpy()  # Average over channels
        axes[i, 1].imshow(attention, cmap='jet', alpha=0.7)
        axes[i, 1].axis('off')

    plt.show()

# Visualize the attention on the first few images from the test set
visualize_attention(model, testloader)
# ```

# ### Summary of the Process:
# - **Attention Mechanism**: We implemented an Attention mechanism that helps the model focus on specific regions of the image.
# - **Visualization**: The attention maps are visualized by overlaying them on the original images to see where the model "looks" during classification.

# You can experiment with different datasets, models, and attention mechanisms for improved performance.
