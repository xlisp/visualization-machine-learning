import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Random 3D surface (loss function)
def loss_function(x, y):
    return torch.sin(x) * torch.cos(y) + 0.1 * (x**2 + y**2)

# Generate a meshgrid for plotting the surface
x = torch.linspace(-5, 5, 100)
y = torch.linspace(-5, 5, 100)
X, Y = torch.meshgrid(x, y)
Z = loss_function(X, Y).detach().numpy()

# Initialize figure and 3D axis for animation
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('SGD Optimization Path on 3D Surface')

# Plot the static 3D surface
ax.plot_surface(X.numpy(), Y.numpy(), Z, cmap='viridis', alpha=0.7)

# SGD starting point
start_point = torch.tensor([4.0, 4.0], requires_grad=True)

# Hyperparameters
learning_rate = 0.1
optimizer = torch.optim.SGD([start_point], lr=learning_rate)

# Number of steps and animation frames
steps = 10
path = np.zeros((steps, 3))

# Plotting the initial point on the surface
point_plot, = ax.plot([], [], [], color='r', marker='o', markersize=5)

# Function to update the frame during animation
def update(i):
    global start_point

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

    # Update point on the surface
    point_plot.set_data(path[:i+1, 0], path[:i+1, 1])
    point_plot.set_3d_properties(path[:i+1, 2])
    
    return point_plot,

# Animate SGD for 10 steps
ani = FuncAnimation(fig, update, frames=steps, interval=500, blit=True)

# Show the animation
plt.show()

