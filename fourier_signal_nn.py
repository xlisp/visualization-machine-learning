## ---- Use pytorch to train a Fourier synthetic signal and predict what the Fourier formula is

# To train a model using PyTorch that predicts the Fourier formula for a synthetic signal, you can follow these general steps. The idea is to generate synthetic signals using known Fourier components and then use a neural network to predict the Fourier coefficients (amplitude and frequency). Here's how you can do it:

# ### Steps:
# 1. **Generate Synthetic Signals**: You create signals composed of sine and cosine waves with random frequencies, phases, and amplitudes.
# 2. **Design a Neural Network**: A neural network will be trained to predict the Fourier coefficients (frequencies and amplitudes) from these signals.
# 3. **Loss Function**: The loss function will compare the predicted Fourier coefficients to the actual ones.
# 4. **Training**: Train the network using backpropagation.
# 5. **Evaluate**: Once trained, the network should be able to predict the Fourier formula of unseen signals.

# Here’s a rough outline of the implementation in PyTorch:

# ### 1. Import Libraries
# ```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# ```

# ### 2. Generate Synthetic Signal
# Define a function to generate a synthetic signal using a combination of sine and cosine waves:

# ```python
def generate_signal(n_samples, n_components=3, length=1000, noise_level=0.01):
    x = np.linspace(0, 2 * np.pi, length)
    signal = np.zeros((n_samples, length))
    coeffs = []

    for i in range(n_samples):
        signal_i = np.zeros(length)
        c_i = []
        for j in range(n_components):
            amplitude = np.random.uniform(0.5, 1.5)
            frequency = np.random.uniform(1, 10)
            phase = np.random.uniform(0, 2 * np.pi)

            signal_i += amplitude * np.sin(frequency * x + phase)
            c_i.append([amplitude, frequency, phase])

        # Add noise
        signal_i += noise_level * np.random.randn(length)
        signal[i, :] = signal_i
        coeffs.append(c_i)

    return signal, np.array(coeffs)
# ```

# ### 3. Build Neural Network Model
# Create a simple feed-forward neural network that will take the signal as input and predict the Fourier components (amplitude, frequency, phase).

# ```python
class FourierPredictor(nn.Module):
    def __init__(self, input_size, n_components):
        super(FourierPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_components * 3)  # Predicting (amplitude, frequency, phase) for each component

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.reshape(-1, n_components, 3)
# ```

# ### 4. Prepare Data
# Convert the synthetic signal into tensors that can be used for training:

# ```python
n_samples = 1000
n_components = 3
length = 1000

signal, coeffs = generate_signal(n_samples, n_components, length)

X_train = torch.tensor(signal, dtype=torch.float32)
y_train = torch.tensor(coeffs, dtype=torch.float32)
# ```

# ### 5. Training the Model

# ```python
# Initialize the model, loss function, and optimizer
model = FourierPredictor(length, n_components)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
n_epochs = 1000
for epoch in range(n_epochs):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train)

    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}')
# ```

# ### 6. Test the Model
# After training, you can test the model on new signals:

# ```python
model.eval()
with torch.no_grad():
    test_signal, test_coeffs = generate_signal(1, n_components, length)
    test_signal_tensor = torch.tensor(test_signal, dtype=torch.float32)

    predicted_coeffs = model(test_signal_tensor).numpy()
    print("Predicted Fourier Coefficients:", predicted_coeffs)
    print("True Fourier Coefficients:", test_coeffs)
# ```

# ### Explanation:
# - **Synthetic Signal**: The `generate_signal` function creates a signal made up of `n_components` sine and cosine waves, each with a random amplitude, frequency, and phase.
# - **Model**: The `FourierPredictor` neural network is trained to predict the Fourier components (amplitude, frequency, phase) for each signal.
# - **Loss Function**: Mean squared error (MSE) is used to minimize the difference between the predicted Fourier components and the true ones.
#
# ### Improvements:
# - Add a Fourier Transform loss to better align with Fourier theory.
# - Use domain-specific techniques like Wavelet Transform.

## ---- run ---------
# @ python fourier_signal_nn.py
# Epoch 1/1000, Loss: 17.069843292236328
# Epoch 101/1000, Loss: 2.4288902282714844
# Epoch 201/1000, Loss: 1.3743107318878174
# Epoch 301/1000, Loss: 0.5828174352645874
# Epoch 401/1000, Loss: 0.27211812138557434
# Epoch 501/1000, Loss: 0.1505374014377594
# Epoch 601/1000, Loss: 0.09342025965452194
# Epoch 701/1000, Loss: 0.06174096092581749
# Epoch 801/1000, Loss: 0.048137370496988297
# Epoch 901/1000, Loss: 0.03508096560835838
# Predicted Fourier Coefficients: [[[0.8005228  0.6980943  1.700388  ]
#   [0.81624836 7.674576   3.6199138 ]
#   [1.0241491  2.8850017  2.4894443 ]]]
# True Fourier Coefficients: [[[1.04119866 1.20042091 2.03329008]
#   [1.20340509 3.00877003 3.5787495 ]
#   [0.60169878 6.96679984 1.77218115]]]
# 
