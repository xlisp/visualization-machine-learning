# --- python log  kmeans visualization
import sys
# To visualize K-means clustering results for log data in Python, we can use libraries like `matplotlib`, `seaborn`, or `plotly`. Here's a step-by-step example showing how to preprocess logs, apply K-means clustering, and visualize the clusters in 2D using `matplotlib` and `sklearn`.

# ### Example Workflow:

# 1. **Load the log data**:
#    - Parse the log file to extract relevant features.
#    - Use vectorization techniques (like `TF-IDF` for text) or extract numerical features.

# 2. **Apply K-Means**:
#    - Use `KMeans` from `sklearn` to cluster the data.

# 3. **Visualize the Clusters**:
#    - Use `matplotlib` to visualize the clusters in a 2D scatter plot.

# ### Code Example

# ```python
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Load and preprocess the log data
#def extract_log_data(log_file):
#    # A simple regex to extract relevant log data
#    log_pattern = r'\[(.*?)\] (.*?) (.*)'
#    logs = []
#    with open(log_file, 'r') as f:
#        for line in f:
#            match = re.match(log_pattern, line)
#            if match:
#                logs.append(match.group(3))  # Append the message part of the log
#    return logs

import re
from collections import Counter

# 1. Analyze log lines to identify patterns (timestamps, log levels)
def infer_log_pattern(log_lines):
    # Common regex patterns for timestamps (e.g., [2024-09-01 10:10:00] or 2024-09-01 10:10:00)
    timestamp_patterns = [
        r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]',  # [YYYY-MM-DD HH:MM:SS]
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'        # YYYY-MM-DD HH:MM:SS
    ]

    # Common log levels
    log_levels = ['INFO', 'ERROR', 'DEBUG', 'WARN', 'TRACE']

    # Build an initial regex pattern template
    log_pattern = ""

    for line in log_lines:
        for pattern in timestamp_patterns:
            if re.search(pattern, line):
                log_pattern += pattern + ' '
                break

        # Try to find log levels in the log line
        for level in log_levels:
            if level in line:
                log_pattern += level + ' '
                break

        # If found a timestamp and log level, extract the message as remaining text
        log_pattern += '(.*)'  # Capture the message part

        # Break after the first match since patterns tend to be consistent
        break

    return log_pattern

# 2. Load logs from a file (you can adjust this to handle larger logs)
def load_log_data(log_file):
    with open(log_file, 'r') as f:
        return f.readlines()

# 3. Automatically detect log pattern
log_lines = load_log_data(sys.argv[1]) #('log_file.log')
log_pattern = infer_log_pattern(log_lines)
print(f"Auto-detected log pattern: {log_pattern}")

# 4. Extract logs using the inferred pattern
def extract_log_data_auto(log_lines, log_pattern):
    logs = []
    for line in log_lines:
        match = re.match(log_pattern, line)
        if match:
            logs.append(match.group(1))  # Extract the message part
    return logs

#logs_extracted = extract_log_data_auto(log_lines, log_pattern)
#print(f"Extracted Logs: {logs_extracted}")

# Load logs from a file (Replace 'log_file.log' with your log file)
logs = extract_log_data_auto(log_lines, log_pattern) #extract_log_data('log_file.log')

# 2. Vectorize the logs using TF-IDF
vectorizer = TfidfVectorizer(stop_words=None, max_features=1000)
X = vectorizer.fit_transform(logs)

# 3. Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# 4. Visualize the clusters
# Using PCA to reduce dimensionality to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Plotting the clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title("K-means Clustering of Log Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()
# ```

# ### Key Points:
# - **Preprocessing**: Extracts log messages using regular expressions.
# - **TF-IDF Vectorization**: Converts log messages into a numerical form for K-means clustering.
# - **K-means Clustering**: Groups log messages into clusters.
# - **PCA for Visualization**: Reduces high-dimensional data to 2D for easy visualization.

# This example uses text logs. If your log data contains other structured numerical data, you can skip vectorization and directly apply K-means on the numerical data.

# Let me know if you need help adjusting this for specific log data!

# ----- run ----- @ python kmeans_visualization.py -----=>  kmeans_visualization.png
# Auto-detected log pattern: (.*)
# python kmeans_visualization.py log_file.log

