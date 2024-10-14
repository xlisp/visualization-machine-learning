import os
import sys
import lmdb
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import requests

def get_embedding(text):
    response = requests.post('http://localhost:11434/api/embeddings', 
                             json={
                                 "model": "mxbai-embed-large",
                                 "prompt": text
                             })
    if response.status_code == 200:
        return response.json()['embedding']
    else:
        raise Exception(f"Error getting embedding: {response.text}")

path = sys.argv[1]
lmdb_path = 'ollama_embeddings.lmdb'

def get_markdown_files():
    md_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                md_files.append(file_path)
    return md_files

def save_embeddings_to_lmdb(files, embeddings):
    env = lmdb.open(lmdb_path, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for file, embedding in zip(files, embeddings):
            txn.put(file.encode(), pickle.dumps(embedding))
    env.close()

def load_embeddings_from_lmdb():
    env = lmdb.open(lmdb_path, readonly=True)
    files = []
    embeddings = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            files.append(key.decode())
            embeddings.append(pickle.loads(value))
    env.close()
    return files, embeddings

def cluster_files(files, embeddings, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    return kmeans.labels_

def visualize_clusters_3d(embeddings, labels):
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        reduced_embeddings[:, 2],
        c=labels,
        cmap='viridis'
    )

    ax.set_title('Markdown File Clusters (3D)')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')

    plt.colorbar(scatter, label='Cluster')
    plt.show()

def main():
    markdown_files = get_markdown_files()

    if not markdown_files:
        print("No markdown files found in the specified directory.")
        return

    if not os.path.exists(lmdb_path):
        print("Generating and saving embeddings...")
        embeddings = []
        for file in markdown_files:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            embeddings.append(get_embedding(content))
        save_embeddings_to_lmdb(markdown_files, embeddings)
    else:
        print("Loading embeddings from LMDB...")
        markdown_files, embeddings = load_embeddings_from_lmdb()

    labels = cluster_files(markdown_files, embeddings)

    groups = {}
    for file, label in zip(markdown_files, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(file)

    for label, files in groups.items():
        print(f"\nGroup {label}:")
        for file in files:
            print(f"  - {file}")

    visualize_clusters_3d(embeddings, labels)
    print("\n3D cluster visualization displayed. Close the plot window to end the program.")

if __name__ == "__main__":
    main()

