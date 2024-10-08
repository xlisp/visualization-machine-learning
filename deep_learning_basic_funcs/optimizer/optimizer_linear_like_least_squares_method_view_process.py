
# range 100 change to `range(1000)` , and when add 100  show the Plot sleep 10s close , and then continue train 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

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

# List to store the loss values
loss_values = []

# Training loop with range(1000)
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    loss_values.append(loss.item())

    # Every 100 iterations, plot and pause
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

        plt.show()
        
        # Sleep for 3 seconds before continuing training
        time.sleep(3)
        plt.close()

# Final plot after training completes
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Final Loss over Epochs')

plt.subplot(1, 2, 2)
plt.scatter(x_train.numpy(), y_train.numpy(), label='Actual Data', color='blue')
plt.plot(x_train.numpy(), model(x_train).detach().numpy(), label='Fitted Line', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Final Model Fit')
plt.legend()

plt.show()

