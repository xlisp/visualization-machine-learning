
# data Transform to 3D, 3D fitting =>  use python pytorch rewrite it , fit it

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Step 5: Visualize the original data and the fitted surface
# Generate a grid for plotting
x_plot = np.linspace(1, 5, 50)
y_plot = np.linspace(1, 5, 50)
x_plot, y_plot = np.meshgrid(x_plot, y_plot)
inputs_plot = torch.tensor(np.c_[x_plot.flatten(), y_plot.flatten()], dtype=torch.float32)
z_plot = model(inputs_plot).detach().numpy().reshape(x_plot.shape)

# Plot the data points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='r', label='Data')

# Plot the fitted surface
ax.plot_surface(x_plot, y_plot, z_plot, color='b', alpha=0.5, label='Fit')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

## --- 3d fit ! yes ! very nice！
# Epoch [100/5000], Loss: 0.0945
# Epoch [200/5000], Loss: 0.0755
# Epoch [300/5000], Loss: 0.0611
# Epoch [400/5000], Loss: 0.0531
# Epoch [500/5000], Loss: 0.0461
# Epoch [600/5000], Loss: 0.0479
# Epoch [700/5000], Loss: 0.0403
# Epoch [800/5000], Loss: 0.0345
# Epoch [900/5000], Loss: 0.0326
# Epoch [1000/5000], Loss: 0.0289
# Epoch [1100/5000], Loss: 0.0265
# Epoch [1200/5000], Loss: 0.0449
# Epoch [1300/5000], Loss: 0.0268
# Epoch [1400/5000], Loss: 0.0207
# Epoch [1500/5000], Loss: 0.0246
# Epoch [1600/5000], Loss: 0.0208
# Epoch [1700/5000], Loss: 0.0181
# Epoch [1800/5000], Loss: 0.0142
# Epoch [1900/5000], Loss: 0.0139
# Epoch [2000/5000], Loss: 0.0154
# Epoch [2100/5000], Loss: 0.0129
# Epoch [2200/5000], Loss: 0.0279
# Epoch [2300/5000], Loss: 0.0106
# Epoch [2400/5000], Loss: 0.0119
# Epoch [2500/5000], Loss: 0.0105
# Epoch [2600/5000], Loss: 0.0100
# Epoch [2700/5000], Loss: 0.0104
# Epoch [2800/5000], Loss: 0.0109
# Epoch [2900/5000], Loss: 0.0079
# Epoch [3000/5000], Loss: 0.0064
# Epoch [3100/5000], Loss: 0.0142
# Epoch [3200/5000], Loss: 0.0061
# Epoch [3300/5000], Loss: 0.0046
# Epoch [3400/5000], Loss: 0.0045
# Epoch [3500/5000], Loss: 0.0036
# Epoch [3600/5000], Loss: 0.0036
# Epoch [3700/5000], Loss: 0.0171
# Epoch [3800/5000], Loss: 0.0048
# Epoch [3900/5000], Loss: 0.0062
# Epoch [4000/5000], Loss: 0.0026
# Epoch [4100/5000], Loss: 0.0070
# Epoch [4200/5000], Loss: 0.0027
# Epoch [4300/5000], Loss: 0.0060
# Epoch [4400/5000], Loss: 0.0061
# Epoch [4500/5000], Loss: 0.0069
# Epoch [4600/5000], Loss: 0.0022
# Epoch [4700/5000], Loss: 0.0047
# Epoch [4800/5000], Loss: 0.0039
# Epoch [4900/5000], Loss: 0.0012
# Epoch [5000/5000], Loss: 0.0012
# 
