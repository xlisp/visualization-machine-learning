# Python & R Machine Learning

* [R Machine Learning](https://github.com/chanshunli/jim-emacs-machine-learning/tree/master/R-Lang-machine-learning)

## kmeans
* [kmeans log analysis](./kmeans_log_analysis.py)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def cluster_error_messages(error_messages, num_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(error_messages)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X)
    ... ...
```
