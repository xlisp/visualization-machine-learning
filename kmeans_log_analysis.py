import re
import glob
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def extract_error_messages(log_files, max_len=200):
    error_messages = []
    
    for file_path in log_files:
        with open(file_path, 'r') as file:
            log_content = file.read()
            # Adjust the regex pattern as needed to match your error messages
            messages = re.findall(r'Error:.*', log_content)
            for message in messages:
                truncated_message = message[:max_len]
                error_messages.append(truncated_message)
    
    return error_messages

def cluster_error_messages(error_messages, num_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(error_messages)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X)
    
    labels = kmeans.labels_
    
    clustered_errors = {}
    for i, label in enumerate(labels):
        if label not in clustered_errors:
            clustered_errors[label] = []
        clustered_errors[label].append(error_messages[i])
    
    return clustered_errors

def count_error_frequencies(clustered_errors):
    error_counter = Counter()
    
    for cluster, messages in clustered_errors.items():
        error_counter[f'Cluster {cluster}'] = len(messages)
    
    return error_counter

def plot_error_frequencies(error_counter):
    categories = list(error_counter.keys())
    frequencies = list(error_counter.values())

    plt.figure(figsize=(10, 6))
    plt.barh(categories, frequencies, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Error Cluster')
    plt.title('Frequency of Error Clusters')
    plt.tight_layout()
    plt.show()

def display_cluster_details(clustered_errors, num_samples=3):
    for cluster, messages in clustered_errors.items():
        print(f'\nCluster {cluster} (Total Messages: {len(messages)}):')
        for message in messages[:num_samples]:  # Show a few sample messages
            print(f'  - {message}')

# Find all .log files in the current directory
log_files = glob.glob('*.log')

# Process the log files
error_messages = extract_error_messages(log_files, max_len=200)
clustered_errors = cluster_error_messages(error_messages, num_clusters=10)  # Adjust number of clusters as needed
error_counter = count_error_frequencies(clustered_errors)

# Display details of each cluster
display_cluster_details(clustered_errors, num_samples=10)  # Adjust the number of sample messages as needed

# Plot the frequencies
plot_error_frequencies(error_counter)


