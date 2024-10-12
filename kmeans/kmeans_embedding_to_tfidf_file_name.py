import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

def get_markdown_files():
    return [f for f in os.listdir('/Users/emacspy/Documents/_think_different_everday') if f.endswith('.md')]

def preprocess_filename(filename):
    # Remove the .md extension and replace underscores and hyphens with spaces
    return filename[:-3].replace('_', ' ').replace('-', ' ')

def cluster_files(files, n_clusters=5):
    # Preprocess filenames
    processed_names = [preprocess_filename(file) for file in files]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform(processed_names)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(tfidf_matrix)
    
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

    for label, files in sorted(groups.items()):
        print(f"\nGroup {label}:")
        for file in files:
            print(f"  - {file}")

if __name__ == "__main__":
    main()

## ---- run --- is bad -----
# @ python kmeans_embedding_to_tfidf_file_name.py
# 
# 
# Group 0:
#   - my resume.md
#   - 2024-09-09.md
#   - 2024-08-09.md
#   - knowed-words.md
#   - hulunote open source project readme.md
#   - 2024-09-14.md
#   - 2024-06-09.md
#   - 2024-09-10.md
#   - 2024-04-09.md
#   - .md
#   - 2024-07-09.md
# 
# Group 1:
#   - 2024-06-15.md
#   - 2024-06-05.md
#   - 2024-05-30.md
#   - 2024-05-31.md
#   - 2024-05-29.md
# 
# Group 2:
#   - 2024-06-20.md
#   - 2024-06-04.md
#   - 2024-06-21.md
#   - 2024-06-11.md
#   - 2024-06-01.md
#   - 2024-06-28.md
#   - 2024-06-19.md
#   - 2024-06-29.md
#   - 2024-06-22.md
#   - 2024-06-06.md
#   - 2024-06-12.md
#   - 2024-04-16.md
#   - 2024-06-02.md
#   - 2024-06-03.md
# 
# Group 3:
#   - 2024-08-07.md
#   - 2024-07-14.md
#   - 2024-07-04.md
#   - 2024-07-30.md
#   - 2024-07-01.md
#   - 2024-07-02.md
#   - 2024-07-06.md
#   - 2024-04-17.md
#   - 2024-06-17.md
#   - 2024-04-27.md
#   - 2024-06-07.md
# 
# Group 4:
#   - 2024-08-23.md
#   - 2024-06-08.md
#   - 2024-08-15.md
#   - 2024-08-10.md
#   - 2024-04-23.md
#   - 2024-06-23.md
# 
