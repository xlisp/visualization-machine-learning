import os
import numpy as np
from sklearn.cluster import KMeans
import requests

def get_markdown_files(directory='/Users/emacspy/Documents/_我的本地库思考'):
    return [f for f in os.listdir(directory) if f.endswith('.md')]

def get_embedding(text):
    response = requests.post('http://localhost:11434/api/embeddings', json={
        'model': 'nomic-embed-text',
        'prompt': text
    })
    return response.json()['embedding']

def group_files(files, embeddings, n_clusters=5):
    # Convert embeddings list to numpy array
    X = np.array(embeddings)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Group files based on cluster labels
    groups = {}
    for file, label in zip(files, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(file)
    
    return groups

def main():
    markdown_files = get_markdown_files()
    embeddings = [get_embedding(file) for file in markdown_files]
    
    # Determine the number of clusters
    n_clusters = min(5, len(markdown_files))  # Use 5 clusters or the number of files, whichever is smaller
    
    groups = group_files(markdown_files, embeddings, n_clusters)
    
    for i, group in enumerate(groups.values()):
        print(f"Group {i+1}:")
        for file in group:
            print(f"  {file}")
        print()

if __name__ == "__main__":
    main()

