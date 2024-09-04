import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Define more data points (predicted and target values)
y_true = torch.tensor([2.5, 0.0, 2.0, 8.0, 4.5, 6.0, 1.5, 3.5, 7.0, 5.0], requires_grad=False)  # target values
y_pred = torch.tensor([3.0, -0.5, 2.0, 7.0, 5.5, 5.0, 2.5, 3.0, 8.0, 4.0], requires_grad=True)  # predicted values (requires_grad=True to calculate gradients)

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
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot squared errors
ax1.bar(range(len(squared_errors)), squared_errors.detach().numpy(), color='orange', label='Squared Errors')
ax1.axhline(y=loss_value.item(), color='r', linestyle='--', label=f'MSE Loss = {loss_value.item():.4f}')
ax1.set_title('MSELoss Calculation in PyTorch with More Data')
ax1.set_xlabel('Data Points')
ax1.set_ylabel('Squared Error')
ax1.legend()

# Plot gradients
ax2.bar(range(len(y_pred)), y_pred.grad.numpy(), color='blue', label='Gradients')
ax2.set_title('Gradients of Predicted Values')
ax2.set_xlabel('Data Points')
ax2.set_ylabel('Gradient')
ax2.legend()

plt.tight_layout()
plt.show()

# @ python loss_bp2.py
#Gradients: tensor([ 0.1000, -0.1000,  0.0000, -0.2000,  0.2000, -0.2000,  0.2000, -0.1000,
#         0.2000, -0.2000])

