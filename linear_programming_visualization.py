## python Linear Programming visualization the process

import numpy as np
import matplotlib.pyplot as plt

# Define the constraints
x1 = np.linspace(0, 10, 400)

# Constraint 1: x1 + x2 <= 5
x2_1 = 5 - x1

# Constraint 2: 2x1 + x2 <= 8
x2_2 = 8 - 2*x1

# Feasibility boundary for x2 >= 0
x2_0 = np.maximum(0, np.minimum(x2_1, x2_2))

# Plot the constraints
plt.figure(figsize=(8, 8))

# Plot the lines representing the constraints
plt.plot(x1, x2_1, label=r'$x_1 + x_2 \leq 5$')
plt.plot(x1, x2_2, label=r'$2x_1 + x_2 \leq 8$')

# Shade the feasible region
plt.fill_between(x1, 0, x2_0, where=(x2_0 >= 0), color='lightgrey', label='Feasible Region')

# Mark corner points
corner_points = [(0, 0), (0, 5), (4, 0), (2, 3)]
for point in corner_points:
    plt.scatter(*point, color='red')
    plt.text(point[0] + 0.1, point[1] - 0.2, f'({point[0]}, {point[1]})', color='black')

# Labels and title
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Linear Programming Problem Visualization')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

