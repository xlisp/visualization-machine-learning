import torch
import matplotlib.pyplot as plt

# Define a range of x values using PyTorch
x = torch.linspace(-5, 5, 100)

# Define the original function y = x^2 and its derivative y' = 2x
y = x**2
y_prime = 2*x

# Convert tensors to numpy arrays for plotting
x_np = x.numpy()
y_np = y.numpy()
y_prime_np = y_prime.numpy()

# Plot both the function and its derivative on the same graph
plt.plot(x_np, y_np, label="y = x^2")
plt.plot(x_np, y_prime_np, label="y' = 2x")

# Add labels and a legend
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = x^2")
plt.show()


