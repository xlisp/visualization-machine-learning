## ------ Use pytorch to train a fit a sin function

# To train a model to fit a sine function using PyTorch, we can build a simple neural network and train it using a dataset of input values and corresponding sine values. Below is an example of how to do this.

# ### Steps:

# 1. **Import libraries:** PyTorch, NumPy, and Matplotlib.
# 2. **Generate a dataset:** Create an array of values from `0` to `2π` as input and the sine of those values as the target.
# 3. **Define the model:** A simple feedforward neural network with a few layers.
# 4. **Define loss function and optimizer:** Mean Squared Error loss and an optimizer like Adam.
# 5. **Train the model:** Run the training loop over multiple epochs, updating the model weights.

# Here's the full code:

# ```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate dataset
# Generate inputs from 0 to 2π
x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x)

# Convert to PyTorch tensors
x_train = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # Shape: [1000, 1]
y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape: [1000, 1]

# 2. Define the neural network model
class SineNet(nn.Module):
    def __init__(self):
        super(SineNet, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 50),  # Input is 1D, output is 50D
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)   # Output is 1D
        )

    def forward(self, x):
        return self.hidden(x)

# 3. Instantiate the model, loss function, and optimizer
model = SineNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Train the model
epochs = 2000
for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)

    # Compute loss
    loss = criterion(y_pred, y_train)

    # Zero gradients, perform backward pass, and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

# 5. Plot the results
y_pred = model(x_train).detach().numpy()  # Get the predicted y from the model
plt.plot(x, y, label='True Sine')
plt.plot(x, y_pred, label='Fitted Sine', linestyle='dashed')
plt.legend()
plt.show()
# ```

# ### Explanation:
# 1. **Dataset:** We create a dataset of input values `x` and corresponding sine values `y`.
# 2. **Model:** The network has two hidden layers with 50 neurons each and ReLU activation functions.
# 3. **Training:** The model is trained using Mean Squared Error loss and the Adam optimizer for 2000 epochs.
# 4. **Plotting:** After training, we plot the true sine curve and the model's predicted sine curve for comparison.

# ### Expected Output:
# The model should be able to learn to approximate the sine function after sufficient training, and the fitted sine curve should align closely with the true sine curve.

## ------ run -------
# python sin_fun_nn.py
# Epoch 0: Loss = 0.5874
# Epoch 100: Loss = 0.0413
# Epoch 200: Loss = 0.0015
# Epoch 300: Loss = 0.0002
# Epoch 400: Loss = 0.0000
# Epoch 500: Loss = 0.0000
# Epoch 600: Loss = 0.0001
# Epoch 700: Loss = 0.0000
# Epoch 800: Loss = 0.0000
# Epoch 900: Loss = 0.0003
# Epoch 1000: Loss = 0.0000
# Epoch 1100: Loss = 0.0000
# Epoch 1200: Loss = 0.0000
# Epoch 1300: Loss = 0.0000
# Epoch 1400: Loss = 0.0000
# Epoch 1500: Loss = 0.0000
# Epoch 1600: Loss = 0.0000
# Epoch 1700: Loss = 0.0000
# Epoch 1800: Loss = 0.0000
# Epoch 1900: Loss = 0.0000
#
