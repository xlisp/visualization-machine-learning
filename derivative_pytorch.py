## use pytorch : Derivative of x^3 => and you can use MMA or Matlab calc it
import torch

# Define the variable x and set requires_grad=True to enable gradient computation
x = torch.tensor(2.0, requires_grad=True)

# Define the function f(x) = x^3
f = x**3
# => tensor(8., grad_fn=<PowBackward0>)

# Compute the derivative of f with respect to x
f.backward()

# Print the derivative (gradient) of f at the point x=2
print(x.grad)  # Should print 12.0, since the derivative of x^3 is 3x^2 and 3*(2^2) = 12

# => tensor(12.)

