# ## ---- python  Convex Optimization visualization the process ,Implement it using pytorch
# Here’s how you can implement convex optimization using **PyTorch**, specifically demonstrating gradient descent to minimize a quadratic function. We'll leverage PyTorch’s automatic differentiation and tensor operations.

# ### Python Code Using PyTorch:

# ```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# Convex function (quadratic function)
def f(x):
    return x ** 2

# Visualization of the optimization process
def visualize_optimization(trajectory):
    # Define x values and their corresponding function values
    x_vals = np.linspace(-3, 3, 400)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=r'$f(x) = x^2$', color='blue')

    # Plot the trajectory of the optimization algorithm
    for i, x in enumerate(trajectory):
        plt.plot(x, f(x), 'ro')  # Mark the point
        plt.text(x, f(x), f'Iter {i}', fontsize=10)
        if i > 0:
            plt.arrow(trajectory[i-1], f(trajectory[i-1]),
                      trajectory[i] - trajectory[i-1],
                      f(trajectory[i]) - f(trajectory[i-1]),
                      head_width=0.1, head_length=0.1, fc='green', ec='green')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Convex Optimization using Gradient Descent (PyTorch)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Gradient descent using PyTorch
def pytorch_gradient_descent(starting_point, learning_rate, num_iterations):
    # Initialize the variable to be optimized
    x = torch.tensor([starting_point], requires_grad=True)
    trajectory = [x.item()]

    # Perform gradient descent
    for _ in range(num_iterations):
        y = f(x)  # Compute the value of the function

        # Perform backpropagation to calculate the gradients
        y.backward()

        # Update x using gradient descent (detach x to avoid issues)
        with torch.no_grad():
            x -= learning_rate * x.grad

        # Clear the gradient for the next iteration
        x.grad.zero_()

        # Save the current value of x
        trajectory.append(x.item())

    return np.array(trajectory)

# Parameters for gradient descent
starting_point = 2.5
learning_rate = 0.1
num_iterations = 10

# Get the trajectory of the optimization process
trajectory = pytorch_gradient_descent(starting_point, learning_rate, num_iterations)

# Visualize the optimization process
visualize_optimization(trajectory)
# ```

# ### Explanation:
# 1. **PyTorch tensor**: The variable `x` is created as a PyTorch tensor with `requires_grad=True` so that PyTorch can compute gradients automatically.
# 2. **Gradient descent**: We manually update the value of `x` by subtracting the product of the learning rate and the gradient (`x.grad`). After each update, the gradient is zeroed out with `x.grad.zero_()`.
# 3. **Backpropagation**: The `.backward()` function computes the gradient of the loss with respect to `x`.
# 4. **Visualization**: The optimization trajectory is visualized similarly to the previous example.

# This PyTorch-based implementation allows you to take advantage of automatic differentiation, making it easier to extend to more complex optimization problems.
