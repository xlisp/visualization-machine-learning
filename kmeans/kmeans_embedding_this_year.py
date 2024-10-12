import os
from openai import OpenAI
import datetime
import sys 

client = OpenAI()
from sklearn.cluster import KMeans
import numpy as np

# Set your OpenAI API key
#openai.api_key = "your-api-key-here"

def get_embedding(text):
    response = client.embeddings.create(input=text,
    model="text-embedding-ada-002")
    return response.data[0].embedding

##def get_markdown_files():
##    return [f for f in os.listdir('/Users/emacspy/Documents/_think_different_everday') if f.endswith('.md')]

path = sys.argv[1]

def get_markdown_files():
    current_year = datetime.datetime.now().year
    md_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                # Get the modification time of the file
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                # Check if the file was modified this year
                if mod_time.year == current_year:
                    md_files.append(file_path)

    return md_files

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

## ----- run ---- is effect: 
# kmeans  master @ prunp kmeans_embedding.py
# 
# Group 2:
#   - my resume.md
#   - knowed-words.md
#   - hulunote open source project readme.md
#   - .md
# 
# Group 3:
#   - 2024-08-23.md
#   - 2024-08-07.md
#   - 2024-07-14.md
#   - 2024-07-04.md
#   - 2024-08-09.md
#   - 2024-07-30.md
#   - 2024-07-01.md
#   - 2024-07-02.md
#   - 2024-07-06.md
#   - 2024-08-15.md
#   - 2024-07-09.md
# 
# Group 1:
#   - 2024-06-20.md
#   - 2024-06-04.md
#   - 2024-06-15.md
#   - 2024-06-21.md
#   - 2024-06-05.md
#   - 2024-06-11.md
#   - 2024-06-01.md
#   - 2024-06-28.md
#   - 2024-06-08.md
#   - 2024-06-19.md
#   - 2024-06-09.md
#   - 2024-06-29.md
#   - 2024-06-22.md
#   - 2024-06-06.md
#   - 2024-06-12.md
#   - 2024-06-02.md
#   - 2024-06-03.md
#   - 2024-06-17.md
#   - 2024-06-23.md
#   - 2024-06-07.md
# 
# Group 4:
#   - 2024-09-09.md
#   - 2024-09-14.md
#   - 2024-09-10.md
#   - 2024-08-10.md
# 
# Group 0:
#   - 2024-05-30.md
#   - 2024-05-31.md
#   - 2024-04-09.md
#   - 2024-04-16.md
#   - 2024-04-17.md
#   - 2024-04-23.md
#   - 2024-05-29.md
#   - 2024-04-27.md
# 
