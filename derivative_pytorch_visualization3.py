##  How to use pytorch to derive a Sin function from -10 to 10 Random number sequence ? And two graphs are drawn on the same coordinate.

import torch
import matplotlib.pyplot as plt

# Step 1: Generate a random sequence of numbers between -10 and 10
x = (torch.rand(100) * 20) - 10  # 100 random numbers in range [-10, 10]
x.requires_grad_(True)  # Enable gradient tracking for x

# Step 2: Define the sine function
y = torch.sin(x) #x * 2 look : derivative_pytorch_visualization3_x2.png 

# Step 3: Compute the derivative of the sine function using backward
y.sum().backward()  # Sum is necessary for backward()

# The derivative of sin(x) is cos(x), stored in x.grad
y_prime = x.grad

# Step 4: Convert tensors to numpy arrays for plotting
x_np = x.detach().numpy()  # Detach to stop gradient tracking
y_np = y.detach().numpy()
y_prime_np = y_prime.numpy()

# Step 5: Plot both the sine function and its derivative
plt.scatter(x_np, y_np, label="y = sin(x)", color="blue")
plt.scatter(x_np, y_prime_np, label="y' = cos(x) (from derivative)", color="red", marker="x")

# Add labels, legend, and grid
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sine Function and its Derivative (Random x from -10 to 10)")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

