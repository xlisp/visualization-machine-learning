## Python implements least squares method

import numpy as np
import matplotlib.pyplot as plt

# Example data points
X = np.array([1, 2.2, 3, 4, 5])
y = np.array([2, 4, 6.3, 8, 11])

# Add a column of ones to X for the intercept term (bias)
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # X_b is X with a bias column

# Calculate the best fit line parameters using the Normal Equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Print the parameters (intercept and slope)
print(f"Intercept: {theta_best[0]}")
print(f"Slope: {theta_best[1]}")

# Predict values using the model
y_pred = X_b.dot(theta_best)

# Plot the data points and the best fit line
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Best fit line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

