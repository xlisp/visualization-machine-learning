import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import imageio
import os

# Step 1: Generate a 100-length random sequence
n = 100
x = torch.linspace(1, 10, n).unsqueeze(1)
y = torch.sin(x) + torch.rand(n, 1) * 0.5

# Step 2: Define a simple neural network model for nonlinear fitting
class NonlinearModel(nn.Module):
    def __init__(self):
        super(NonlinearModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NonlinearModel()

# Step 3: Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Initialize a list to store loss values and filenames for the GIF
loss_history = []
filenames = []

# Create a directory to store the images
if not os.path.exists('gif_images'):
    os.makedirs('gif_images')

# Step 4: Train the model
epochs = 1000
for epoch in range(epochs):
    model.train()

    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save the loss value for this epoch
    loss_history.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        # Plot the original data and the fitted curve
        model.eval()
        with torch.no_grad():
            predicted = model(x).numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(x.numpy(), y.numpy(), 'ro', label='Original data')
        plt.plot(x.numpy(), predicted, 'b-', label='Fitted curve')
        plt.legend()
        
        # Save the plot as an image file
        filename = f'gif_images/epoch_{epoch+1}.png'
        plt.savefig(filename)
        filenames.append(filename)
        
        #plt.show()
        #plt.close()

# Create a GIF from the saved images
with imageio.get_writer('training_process.gif', mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optional: Remove the images after creating the GIF
for filename in filenames:
    os.remove(filename)

# Plot the final fitted curve after training
model.eval()
with torch.no_grad():
    predicted = model(x).numpy()

plt.figure(figsize=(10, 5))
plt.plot(x.numpy(), y.numpy(), 'ro', label='Original data')
plt.plot(x.numpy(), predicted, 'b-', label='Fitted curve')
plt.legend()
plt.show()

# Step 5: Plot the loss change history
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), loss_history, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Change History')
plt.legend()
plt.show()

