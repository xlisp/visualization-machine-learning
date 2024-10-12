import os
from openai import OpenAI
import datetime
import sys
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(input=text,
    model="text-embedding-ada-002")
    return response.data[0].embedding

path = sys.argv[1]

def get_markdown_files():
    target_year = 2023
    md_files = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                if mod_time.year == target_year:
                    md_files.append(file_path)

    return md_files

def cluster_files(files, n_clusters=10):
    embeddings = [get_embedding(file) for file in files]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    return kmeans.labels_, embeddings

def visualize_clusters(embeddings, labels):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.title('Markdown File Clusters')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig('markdown_clusters.png')
    plt.close()

def main():
    markdown_files = get_markdown_files()

    if not markdown_files:
        print("No markdown files found from 2023 in the specified directory.")
        return

    labels, embeddings = cluster_files(markdown_files)

    groups = {}
    for file, label in zip(markdown_files, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(file)

    for label, files in groups.items():
        print(f"\nGroup {label}:")
        for file in files:
            print(f"  - {file}")

    visualize_clusters(embeddings, labels)
    print("\nCluster visualization saved as 'markdown_clusters.png'")

if __name__ == "__main__":
    main()

