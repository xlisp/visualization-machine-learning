import os
import re
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import faiss  # Replacing LMDB with FAISS
from sklearn.decomposition import PCA

def get_todo_items(directory='/Users/emacspy/Documents/_我的本地库思考'):
    todo_items = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                todos = re.findall(r'^.*(?:todo|TODO).*$', content, re.MULTILINE)
                for todo in todos:
                    todo_items.append((filename, todo.strip()))
    return todo_items

def get_embedding(text):
    response = requests.post('http://localhost:11434/api/embeddings', json={
        'model': 'nomic-embed-text',
        'prompt': text
    })
    return response.json()['embedding']

def save_embeddings_to_faiss(embeddings, db_path='todo_embeddings.index'):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)  # L2 distance index
    index.add(np.array(embeddings).astype(np.float32))  # Add embeddings to the FAISS index
    faiss.write_index(index, db_path)

def load_embeddings_from_faiss(db_path='todo_embeddings.index'):
    index = faiss.read_index(db_path)
    return index

def get_or_calculate_embeddings(todo_items):
    db_path = 'todo_embeddings.index'

    if os.path.exists(db_path):
        print("Loading embeddings from FAISS index...")
        index = load_embeddings_from_faiss(db_path)

        # Extract embeddings from FAISS index one by one
        num_embeddings = index.ntotal  # Number of stored embeddings
        dim = index.d  # The dimensionality of the embeddings

        embeddings = np.zeros((num_embeddings, dim))  # Placeholder for the embeddings

        for i in range(num_embeddings):
            embeddings[i] = index.reconstruct(i)  # Reconstruct each embedding one at a time

        return embeddings
    else:
        print("Calculating new embeddings...")
        embeddings = []
        for _, todo in todo_items:
            print(f"Calculating embedding for: {todo}")
            embedding = get_embedding(todo)
            embeddings.append(embedding)

        save_embeddings_to_faiss(embeddings, db_path)
        return np.array(embeddings)

def group_todos(todo_items, embeddings, n_clusters=300):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    groups = {}
    for (filename, todo), label in zip(todo_items, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append((filename, todo))
    
    return groups, labels, kmeans.cluster_centers_

def visualize_clusters_3d(embeddings, labels, cluster_centers, todo_items):
    embeddings = np.array(embeddings)
    
    # Combine embeddings and cluster centers for visualization
    combined = np.vstack((embeddings, cluster_centers))
    
    # Apply PCA to reduce dimensionality to 3D for visualization
    pca = PCA(n_components=3)
    combined_3d = pca.fit_transform(combined)
    
    embeddings_3d = combined_3d[:len(embeddings)]
    centers_3d = combined_3d[len(embeddings):]

    # Plotting the 3D clusters
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of the embeddings (TODO items) with cluster labels as colors
    scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
                         c=labels, cmap='viridis', alpha=0.7)
    
    # Scatter plot of the cluster centers
    ax.scatter(centers_3d[:, 0], centers_3d[:, 1], centers_3d[:, 2], 
               c='red', marker='x', s=200, linewidths=3)
    
    # Adjust the text positioning to avoid overlap
    for i, (filename, todo) in enumerate(todo_items):
        # Offset the text position slightly for better readability
        ax.text(embeddings_3d[i, 0] + 1, embeddings_3d[i, 1] + 1, embeddings_3d[i, 2] + 1, 
                #f"{filename[:2]}", fontsize=8, alpha=0.7)
                "", fontsize=8, alpha=0.7)

    # Add color bar for the clusters
    plt.colorbar(scatter)
    
    # Set titles and labels
    ax.set_title('3D K-means Clustering of TODO Items (PCA)')
    ax.set_xlabel('PCA component 1')
    ax.set_ylabel('PCA component 2')
    ax.set_zlabel('PCA component 3')

    # Optimize layout and display plot
    plt.tight_layout()
    plt.show()

def main():
    todo_items = get_todo_items()
    embeddings = get_or_calculate_embeddings(todo_items)
    
    n_clusters = min(300, len(todo_items))
    
    groups, labels, cluster_centers = group_todos(todo_items, embeddings, n_clusters)
    
    for i, group in enumerate(groups.values()):
        print(f"Group {i+1}:")
        for filename, todo in group:
            print(f"  [{filename}] {todo}")
        print()
    
    visualize_clusters_3d(embeddings, labels, cluster_centers, todo_items)

if __name__ == "__main__":
    main()

