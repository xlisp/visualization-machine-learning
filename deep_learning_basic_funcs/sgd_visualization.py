## Randomly generate a 3D surface and visualize the stochastic gradient descent process of SGD using pytorch

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Random 3D surface (loss function)
def loss_function(x, y):
    return torch.sin(x) * torch.cos(y) + 0.1 * (x**2 + y**2)

# Generate a meshgrid for plotting the surface
x = torch.linspace(-5, 5, 100)
y = torch.linspace(-5, 5, 100)
X, Y = torch.meshgrid(x, y)
Z = loss_function(X, Y).detach().numpy()

# Plot the 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X.numpy(), Y.numpy(), Z, cmap='viridis', alpha=0.7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Random 3D Surface (Loss Function)')
plt.show()

# SGD Visualization
# Starting point
start_point = torch.tensor([4.0, 4.0], requires_grad=True)

# Hyperparameters
learning_rate = 0.1
optimizer = torch.optim.SGD([start_point], lr=learning_rate)

# Track the path of optimization
steps = 50
path = np.zeros((steps, 3))

for i in range(steps):
    optimizer.zero_grad()
    
    # Calculate the loss (z value)
    loss = loss_function(start_point[0], start_point[1])
    
    # Backpropagation to compute gradients
    loss.backward()
    
    # Perform optimization step
    optimizer.step()
    
    # Store the (x, y, z) values
    path[i, 0] = start_point[0].item()
    path[i, 1] = start_point[1].item()
    path[i, 2] = loss.item()

# Plot the gradient descent path
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
ax.plot_surface(X.numpy(), Y.numpy(), Z, cmap='viridis', alpha=0.7)

# Plot the path of the optimizer
ax.plot(path[:, 0], path[:, 1], path[:, 2], color='r', marker='o', markersize=5, label='SGD Path')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('SGD Optimization Path on 3D Surface')
plt.legend()
plt.show()

