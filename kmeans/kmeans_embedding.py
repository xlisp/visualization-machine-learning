import os
from openai import OpenAI

client = OpenAI()
from sklearn.cluster import KMeans
import numpy as np

# Set your OpenAI API key
#openai.api_key = "your-api-key-here"

def get_embedding(text):
    response = client.embeddings.create(input=text,
    model="text-embedding-ada-002")
    return response.data[0].embedding

def get_markdown_files():
    return [f for f in os.listdir('/Users/emacspy/Documents/_think_different_everday') if f.endswith('.md')]

def cluster_files(files, n_clusters=5):
    embeddings = [get_embedding(file) for file in files]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    return kmeans.labels_

def main():
    markdown_files = get_markdown_files()

    if not markdown_files:
        print("No markdown files found in the current directory.")
        return

    labels = cluster_files(markdown_files)

    groups = {}
    for file, label in zip(markdown_files, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(file)

    for label, files in groups.items():
        print(f"\nGroup {label}:")
        for file in files:
            print(f"  - {file}")

if __name__ == "__main__":
    main()

