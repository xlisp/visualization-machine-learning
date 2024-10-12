import os
import re
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_markdown_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.md')]

def read_markdown_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # Remove code blocks and special characters
    content = re.sub(r'```[\s\S]*?```', '', content)
    content = re.sub(r'[^a-zA-Z\s]', '', content)
    return content

def cluster_and_visualize_files(files, n_clusters=5):
    file_contents = [read_markdown_content(file) for file in files]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(file_contents)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    # Dimensionality reduction for visualization
    tsne = TSNE(n_components=2, random_state=42)
    coords = tsne.fit_transform(tfidf_matrix.toarray())
    
    # Visualization
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter)
    
    # Annotate points with file names
    for i, file in enumerate(files):
        plt.annotate(os.path.basename(file), (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.7)
    
    plt.title('Markdown Files Clustered by Content Similarity')
    plt.xlabel('t-SNE feature 0')
    plt.ylabel('t-SNE feature 1')
    plt.tight_layout()
    plt.savefig('markdown_clusters.png')
    print("Visualization saved as 'markdown_clusters.png'")
    
    return cluster_labels

def main(directory, n_clusters):
    markdown_files = get_markdown_files(directory)
    
    if not markdown_files:
        print(f"No markdown files found in the directory: {directory}")
        return

    labels = cluster_and_visualize_files(markdown_files, n_clusters)

    groups = {}
    for file, label in zip(markdown_files, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(file)

    for label, files in groups.items():
        print(f"\nGroup {label}:")
        for file in files:
            print(f"  - {os.path.basename(file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group and visualize markdown files by similarity.")
    parser.add_argument("directory", help="Path to the directory containing markdown files")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters (default: 5)")
    args = parser.parse_args()

    main(args.directory, args.clusters)

# ------ run test ------
kmeans  master @ prunp kmeans_embedding_to_tfidf.py /Users/emacspy/Documents/_think_different_everday

Visualization saved as 'markdown_clusters.png'

Group 0:
  - my resume.md
  - 2024-08-23.md
  - 2024-06-20.md
  - 2024-08-07.md
  - 2024-06-04.md
  - 2024-09-09.md
  - 2024-06-15.md
  - 2024-06-05.md
  - 2024-06-11.md
  - 2024-06-01.md
  - 2024-07-14.md
  - 2024-07-04.md
  - 2024-08-09.md
  - 2024-07-30.md
  - 2024-05-30.md
  - 2024-05-31.md
  - 2024-07-01.md
  - knowed-words.md
  - hulunote open source project readme.md
  - 2024-06-28.md
  - 2024-07-02.md
  - 2024-06-08.md
  - 2024-07-06.md
  - 2024-09-14.md
  - 2024-06-19.md
  - 2024-06-09.md
  - 2024-09-10.md
  - 2024-04-09.md
  - 2024-06-29.md
  - 2024-08-15.md
  - .md
  - 2024-06-06.md
  - 2024-06-12.md
  - 2024-04-16.md
  - 2024-06-02.md
  - 2024-08-10.md
  - 2024-04-17.md
  - 2024-06-03.md
  - 2024-04-23.md
  - 2024-06-17.md
  - 2024-05-29.md
  - 2024-04-27.md
  - 2024-07-09.md

Group 2:
  - 2024-06-21.md

Group 1:
  - 2024-06-22.md

Group 3:
  - 2024-06-23.md

Group 4:
  - 2024-06-07.md

