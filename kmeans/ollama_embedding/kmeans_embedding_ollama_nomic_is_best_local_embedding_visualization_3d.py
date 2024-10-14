import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import lmdb
import pickle

def get_markdown_files(directory='/Users/emacspy/Documents/_我的本地库思考'):
    return [f for f in os.listdir(directory) if f.endswith('.md')]

def get_embedding(text):
    response = requests.post('http://localhost:11434/api/embeddings', json={
        'model': 'nomic-embed-text',
        'prompt': text
    })
    return response.json()['embedding']

def save_embeddings_to_lmdb(embeddings_dict, db_path='embeddings_ollama2.lmdb'):
    env = lmdb.open(db_path, map_size=1024*1024*1024)  # 1GB size
    with env.begin(write=True) as txn:
        for key, value in embeddings_dict.items():
            txn.put(key.encode(), pickle.dumps(value))

def load_embeddings_from_lmdb(db_path='embeddings_ollama2.lmdb'):
    env = lmdb.open(db_path, readonly=True)
    embeddings_dict = {}
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            embeddings_dict[key.decode()] = pickle.loads(value)
    return embeddings_dict

def get_or_calculate_embeddings(files):
    db_path = 'embeddings_ollama2.lmdb'
    if os.path.exists(db_path):
        print("Loading embeddings from LMDB...")
        embeddings_dict = load_embeddings_from_lmdb(db_path)
    else:
        print("Calculating new embeddings...")
        embeddings_dict = {}

    for file in files:
        if file not in embeddings_dict:
            print(f"Calculating embedding for {file}")
            embeddings_dict[file] = get_embedding(file)

    save_embeddings_to_lmdb(embeddings_dict)
    return [embeddings_dict[file] for file in files]

def group_files(files, embeddings, n_clusters=20):
    X = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    groups = {}
    for file, label in zip(files, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(file)
    
    return groups, labels, kmeans.cluster_centers_

def visualize_clusters_3d(embeddings, labels, cluster_centers, files):
    embeddings = np.array(embeddings)
    
    combined = np.vstack((embeddings, cluster_centers))
    
    tsne = TSNE(n_components=3, random_state=42)
    combined_3d = tsne.fit_transform(combined)
    
    embeddings_3d = combined_3d[:len(embeddings)]
    centers_3d = combined_3d[len(embeddings):]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
                         c=labels, cmap='viridis', alpha=0.7)
    ax.scatter(centers_3d[:, 0], centers_3d[:, 1], centers_3d[:, 2], 
               c='red', marker='x', s=200, linewidths=3)
    
    for i, file in enumerate(files):
        ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], 
                os.path.splitext(file)[0], fontsize=8, alpha=0.7)
    
    plt.colorbar(scatter)
    ax.set_title('3D K-means Clustering of Markdown Files')
    ax.set_xlabel('t-SNE feature 1')
    ax.set_ylabel('t-SNE feature 2')
    ax.set_zlabel('t-SNE feature 3')
    plt.tight_layout()
    plt.show()

def main():
    markdown_files = get_markdown_files()
    embeddings = get_or_calculate_embeddings(markdown_files)
    
    n_clusters = min(20, len(markdown_files))
    
    groups, labels, cluster_centers = group_files(markdown_files, embeddings, n_clusters)
    
    for i, group in enumerate(groups.values()):
        print(f"Group {i+1}:")
        for file in group:
            print(f"  {file}")
        print()
    
    visualize_clusters_3d(embeddings, labels, cluster_centers, markdown_files)

if __name__ == "__main__":
    main()

