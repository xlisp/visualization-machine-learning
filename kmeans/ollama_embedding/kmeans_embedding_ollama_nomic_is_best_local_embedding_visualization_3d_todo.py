import os
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import lmdb
import pickle

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

def save_embeddings_to_lmdb(embeddings_dict, db_path='todo_embeddings.lmdb'):
    env = lmdb.open(db_path, map_size=1024*1024*1024)  # 1GB size
    with env.begin(write=True) as txn:
        for key, value in embeddings_dict.items():
            txn.put(key.encode(), pickle.dumps(value))

def load_embeddings_from_lmdb(db_path='todo_embeddings.lmdb'):
    env = lmdb.open(db_path, readonly=True)
    embeddings_dict = {}
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            embeddings_dict[key.decode()] = pickle.loads(value)
    return embeddings_dict

def get_or_calculate_embeddings(todo_items):
    db_path = 'todo_embeddings.lmdb'
    if os.path.exists(db_path):
        print("Loading embeddings from LMDB...")
        embeddings_dict = load_embeddings_from_lmdb(db_path)
    else:
        print("Calculating new embeddings...")
        embeddings_dict = {}

    for _, todo in todo_items:
        if todo not in embeddings_dict:
            print(f"Calculating embedding for: {todo}")
            embeddings_dict[todo] = get_embedding(todo)

    save_embeddings_to_lmdb(embeddings_dict)
    return [embeddings_dict[todo] for _, todo in todo_items]

def group_todos(todo_items, embeddings, n_clusters=20):
    X = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    groups = {}
    for (filename, todo), label in zip(todo_items, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append((filename, todo))
    
    return groups, labels, kmeans.cluster_centers_

def visualize_clusters_3d(embeddings, labels, cluster_centers, todo_items):
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
    
    for i, (filename, todo) in enumerate(todo_items):
        ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], 
                f"{filename}: {todo[:20]}...", fontsize=8, alpha=0.7)
    
    plt.colorbar(scatter)
    ax.set_title('3D K-means Clustering of TODO Items')
    ax.set_xlabel('t-SNE feature 1')
    ax.set_ylabel('t-SNE feature 2')
    ax.set_zlabel('t-SNE feature 3')
    plt.tight_layout()
    plt.show()

def main():
    todo_items = get_todo_items()
    embeddings = get_or_calculate_embeddings(todo_items)
    
    n_clusters = min(20, len(todo_items))
    
    groups, labels, cluster_centers = group_todos(todo_items, embeddings, n_clusters)
    
    for i, group in enumerate(groups.values()):
        print(f"Group {i+1}:")
        for filename, todo in group:
            print(f"  [{filename}] {todo}")
        print()
    
    visualize_clusters_3d(embeddings, labels, cluster_centers, todo_items)

if __name__ == "__main__":
    main()
