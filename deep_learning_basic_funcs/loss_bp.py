import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Define the data (predicted and target values)
y_true = torch.tensor([2.5, 0.0, 2.0, 8.0], requires_grad=False)  # target values
y_pred = torch.tensor([3.0, -0.5, 2.0, 7.0], requires_grad=True)  # predicted values (requires_grad=True to calculate gradients)

# 2. Define the MSELoss function
mse_loss = nn.MSELoss()

# 3. Calculate the loss
loss_value = mse_loss(y_pred, y_true)

# 4. Perform backpropagation
loss_value.backward()

# 5. Print gradients
print("Gradients:", y_pred.grad)

# Calculate the individual squared errors for plotting
squared_errors = (y_pred - y_true) ** 2

# 6. Plot the results
plt.figure(figsize=(10, 6))
x_labels = [f'Error {i+1}' for i in range(len(squared_errors))]
plt.bar(x_labels, squared_errors.detach().numpy(), color='orange', label='Squared Errors')
plt.axhline(y=loss_value.item(), color='r', linestyle='--', label=f'MSE Loss = {loss_value.item():.4f}')
plt.title('MSELoss Calculation in PyTorch with Backpropagation')
plt.xlabel('Errors')
plt.ylabel('Squared Error')
plt.legend()
plt.show()

# @ python loss_bp.py
# Gradients: tensor([ 0.2500, -0.2500,  0.0000, -0.5000])

