## Network fit mma:  use python pytorch rewrite it , fit it

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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

# Step 4: Train the model
epochs = 5000
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 5: Plot the original data and the fitted curve
model.eval()
with torch.no_grad():
    predicted = model(x).numpy()

plt.figure(figsize=(10, 5))
plt.plot(x.numpy(), y.numpy(), 'ro', label='Original data')
plt.plot(x.numpy(), predicted, 'b-', label='Fitted curve')
plt.legend()
plt.show()

## ---- fit success! ! ! => 2013_nonlinear_fitting_pytorch.png
#Epoch [100/1000], Loss: 0.2747
#Epoch [200/1000], Loss: 0.2334
#Epoch [300/1000], Loss: 0.2321
#Epoch [400/1000], Loss: 0.2316
#Epoch [500/1000], Loss: 0.2315
#Epoch [600/1000], Loss: 0.2315
#Epoch [700/1000], Loss: 0.2315
#Epoch [800/1000], Loss: 0.2316
#Epoch [900/1000], Loss: 0.2317
#Epoch [1000/1000], Loss: 0.2315

## when 5000 , is overfiting: 	2013_nonlinear_fitting_overfitting.png
# Epoch [100/5000], Loss: 0.0964
# Epoch [200/5000], Loss: 0.0210
# Epoch [300/5000], Loss: 0.0203
# Epoch [400/5000], Loss: 0.0190
# Epoch [500/5000], Loss: 0.0184
# Epoch [600/5000], Loss: 0.0177
# Epoch [700/5000], Loss: 0.0168
# Epoch [800/5000], Loss: 0.0162
# Epoch [900/5000], Loss: 0.0158
# Epoch [1000/5000], Loss: 0.0153
# Epoch [1100/5000], Loss: 0.0146
# Epoch [1200/5000], Loss: 0.0141
# Epoch [1300/5000], Loss: 0.0136
# Epoch [1400/5000], Loss: 0.0132
# Epoch [1500/5000], Loss: 0.0131
# Epoch [1600/5000], Loss: 0.0130
# Epoch [1700/5000], Loss: 0.0128
# Epoch [1800/5000], Loss: 0.0127
# Epoch [1900/5000], Loss: 0.0127
# Epoch [2000/5000], Loss: 0.0126
# Epoch [2100/5000], Loss: 0.0126
# Epoch [2200/5000], Loss: 0.0130
# Epoch [2300/5000], Loss: 0.0126
# Epoch [2400/5000], Loss: 0.0126
# Epoch [2500/5000], Loss: 0.0125
# Epoch [2600/5000], Loss: 0.0125
# Epoch [2700/5000], Loss: 0.0125
# Epoch [2800/5000], Loss: 0.0126
# Epoch [2900/5000], Loss: 0.0128
# Epoch [3000/5000], Loss: 0.0124
# Epoch [3100/5000], Loss: 0.0124
# Epoch [3200/5000], Loss: 0.0124
# Epoch [3300/5000], Loss: 0.0125
# Epoch [3400/5000], Loss: 0.0126
# Epoch [3500/5000], Loss: 0.0125
# Epoch [3600/5000], Loss: 0.0124
# Epoch [3700/5000], Loss: 0.0125
# Epoch [3800/5000], Loss: 0.0124
# Epoch [3900/5000], Loss: 0.0125
# Epoch [4000/5000], Loss: 0.0125
# Epoch [4100/5000], Loss: 0.0125
# Epoch [4200/5000], Loss: 0.0124
# Epoch [4300/5000], Loss: 0.0124
# Epoch [4400/5000], Loss: 0.0125
# Epoch [4500/5000], Loss: 0.0124
# Epoch [4600/5000], Loss: 0.0124
# Epoch [4700/5000], Loss: 0.0124
# Epoch [4800/5000], Loss: 0.0124
# Epoch [4900/5000], Loss: 0.0124
# Epoch [5000/5000], Loss: 0.0124
# 
