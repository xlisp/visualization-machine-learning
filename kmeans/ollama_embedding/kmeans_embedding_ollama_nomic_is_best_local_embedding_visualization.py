import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import requests
import matplotlib.pyplot as plt

def get_markdown_files(directory='/Users/emacspy/Documents/_我的本地库思考'):
    return [f for f in os.listdir(directory) if f.endswith('.md')]

def get_embedding(text):
    response = requests.post('http://localhost:11434/api/embeddings', json={
        'model': 'nomic-embed-text',
        'prompt': text
    })
    return response.json()['embedding']

def group_files(files, embeddings, n_clusters=10):
    X = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    groups = {}
    for file, label in zip(files, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(file)
    
    return groups, labels, kmeans.cluster_centers_

def visualize_clusters(embeddings, labels, cluster_centers, files):
    embeddings = np.array(embeddings)
    
    # Combine embeddings and cluster centers
    combined = np.vstack((embeddings, cluster_centers))
    
    # Apply t-SNE to the combined data
    tsne = TSNE(n_components=2, random_state=42)
    combined_2d = tsne.fit_transform(combined)
    
    # Separate the results
    embeddings_2d = combined_2d[:len(embeddings)]
    centers_2d = combined_2d[len(embeddings):]

    # Create a scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='x', s=200, linewidths=3)
    
    # Add labels for each point
    for i, file in enumerate(files):
        plt.annotate(os.path.splitext(file)[0], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.7)
    
    plt.colorbar(scatter)
    plt.title('K-means Clustering of Markdown Files')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.tight_layout()
    plt.show()

def main():
    markdown_files = get_markdown_files()
    embeddings = [get_embedding(file) for file in markdown_files]
    
    n_clusters = min(10, len(markdown_files))
    
    groups, labels, cluster_centers = group_files(markdown_files, embeddings, n_clusters)
    
    for i, group in enumerate(groups.values()):
        print(f"Group {i+1}:")
        for file in group:
            print(f"  {file}")
        print()
    
    visualize_clusters(embeddings, labels, cluster_centers, markdown_files)

if __name__ == "__main__":
    main()
