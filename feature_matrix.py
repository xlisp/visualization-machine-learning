import numpy as np
import matplotlib.pyplot as plt

# Define the feature matrix
feature_matrix = np.array([
    [1.0, 0.5, 0.2],
    [0.3, 0.8, 0.6],
    [0.7, 0.1, 0.4]
])

# Create a plot
plt.imshow(feature_matrix, cmap='rainbow', aspect='auto')

# Add color bar
plt.colorbar()


# Set tick positions and labels
plt.xticks(ticks=np.arange(feature_matrix.shape[1]), labels=["Feature 1", "Feature 2", "Feature 3"])

plt.yticks(ticks=np.arange(feature_matrix.shape[0]), labels=["Row 1", "Row 2", "Row 3"])

# Add labels
plt.xlabel('Features')
plt.ylabel('Rows')

# Show the plot
plt.show() # close plt can next run
