# use pytorch :  x^3 integral => pytorch like mma！

import torch

# Define the function f(x) = x^3
def f(x):
    return x**3

# Define the interval [a, b]
a = 0.0
b = 2.0

# Number of points to use in the approximation
n_points = 1000

# Create a tensor of x values evenly spaced between a and b
x = torch.linspace(a, b, n_points)
#tensor([0.0000, 0.0020, 0.0040, 0.0060, 0.0080, 0.0100, 0.0120, 0.0140, 0.0160,
#        0.0180, 0.0200, 0.0220, 0.0240, 0.0260, 0.0280, 0.0300, 0.0320, 0.0340,
# ...)

# Compute the corresponding y values
y = f(x)
#tensor([0.0000e+00, 8.0240e-09, 6.4192e-08, 2.1665e-07, 5.1354e-07, 1.0030e-06,
#        1.7332e-06, 2.7522e-06, 4.1083e-06, 5.8495e-06, 8.0240e-06, 1.0680e-05,
#        1.3866e-05, 1.7629e-05, 2.2018e-05, 2.7081e-05, 3.2866e-05, 3.9422e-05,
#        4.6796e-05, 5.5037e-05, 6.4192e-05, 7.4311e-05, 8.5440e-05, 9.7629e-05,
#....)

# Approximate the integral using the trapezoidal rule
integral = torch.trapz(y, x)

# Print the result
print(f"The approximate integral of x^3 from {a} to {b} is: {integral.item()}") 
# => The approximate integral of x^3 from 0.0 to 2.0 is: 4.000003814697266


