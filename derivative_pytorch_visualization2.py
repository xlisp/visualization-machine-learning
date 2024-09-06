import torch
import matplotlib.pyplot as plt

# Step 1: Define a range of x values using PyTorch
x = torch.linspace(-10, 10, 100, requires_grad=True)

# Step 2: Define the function y = x^2
y = x**2

# Step 3: Compute the derivative by backpropagation
y.sum().backward()  # Sum is necessary for backward() in case of multiple elements

# The derivative y' = 2x is stored in x.grad
y_prime = x.grad

print(y_prime)

# Step 4: Convert tensors to numpy arrays for plotting
x_np = x.detach().numpy()  # Detach to avoid gradient tracking
y_np = y.detach().numpy()
y_prime_np = y_prime.numpy()

# Step 5: Plot both the function and its derivative on the same graph
plt.plot(x_np, y_np, label="y = x^2")
plt.plot(x_np, y_prime_np, label="y' = 2x (from derivative)", linestyle="--")

# Add labels and a legend
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = x^2 and its Derivative (y' from PyTorch)")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

