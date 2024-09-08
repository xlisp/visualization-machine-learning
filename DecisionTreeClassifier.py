# --- python Decision Tree, add need visualization the process

# To visualize the decision-making process of the Decision Tree, we can use `plot_tree` from `scikit-learn`. This function generates a plot of the tree structure, showing the conditions at each node and the splits made during training.

# Here’s an extended version of the previous example with a visualization of the decision tree:

# ### Full Example with Visualization:

# ```python
# Import required libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

# Load the dataset (Iris dataset in this case)
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the decision tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
# ```

# ### Key Explanation:
# 1. **`plot_tree` Function**:
#    - `clf`: The trained decision tree classifier.
#    - `feature_names`: Names of the features (columns) in the dataset.
#    - `class_names`: Labels of the target variable (class names).
#    - `filled=True`: Colors the nodes to indicate different classes.
#
# 2. **`plt.figure(figsize=(12,8))`**: Resizes the plot for better visualization.

# ### Visualization Output:
# The visualization will show a tree where each node contains:
# - The feature used for splitting.
# - The threshold for the split.
# - The Gini impurity or entropy of the node.
# - The number of samples at each node.
# - The class distribution and the predicted class for leaf nodes.

# This provides a clear visual of the decision process, showing how the tree splits data at each step.

# Let me know if you'd like further customization for the tree's visualization!

## python DecisionTreeClassifier.py
# Matplotlib is building the font cache; this may take a moment.
# Accuracy: 100.00%
