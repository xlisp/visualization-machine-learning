## save the all plot history to gif file 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from PIL import Image

# Define a simple linear model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Generate some synthetic data (y = 2x + 1 with some noise)
x_train = torch.linspace(-1, 1, 100).reshape(-1, 1)
y_train = 2 * x_train + 1 + 0.2 * torch.randn(x_train.size())

# List to store the loss values and frames for the GIF
loss_values = []
frames = []

# Training loop with range(500)
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    loss_values.append(loss.item())

    # Every 100 iterations, plot and save the frame
    if (epoch + 1) % 10 == 0:
        # Plot the loss over epochs
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')

        # Plot the model's predictions vs actual data
        plt.subplot(1, 2, 2)
        plt.scatter(x_train.numpy(), y_train.numpy(), label='Actual Data', color='blue')
        plt.plot(x_train.numpy(), model(x_train).detach().numpy(), label='Fitted Line', color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Model Fit')
        plt.legend()

        # Save the current plot as an image and store it in the frames list
        filename = f'plot_epoch_{epoch+1}.png'
        plt.savefig(filename)
        frames.append(Image.open(filename))
        
        # Close the plot to continue training
        plt.close()

# Save the frames as a GIF
gif_path = 'training_animation.gif'
frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=1000, loop=0)

# Return the path to the generated GIF
gif_path

