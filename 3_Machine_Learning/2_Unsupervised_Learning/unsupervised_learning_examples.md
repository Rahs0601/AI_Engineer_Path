# Unsupervised Learning Examples

This file contains examples of common unsupervised learning algorithms using scikit-learn.

## 1. K-Means Clustering

K-Means is a popular unsupervised machine learning algorithm used for clustering. It aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate synthetic data for clustering
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-Means
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

print("--- K-Means Clustering ---")
print("Cluster centers:\n", kmeans.cluster_centers_)

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
```

## 2. Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique. It transforms a set of possibly correlated variables into a set of linearly uncorrelated variables called principal components. The first principal component accounts for as much of the variability in the data as possible, and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components.

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA
pca = PCA(n_components=2) # Reduce to 2 dimensions
X_pca = pca.fit_transform(X)

print("\n--- Principal Component Analysis (PCA) ---")
print("Original data shape:", X.shape)
print("Reduced data shape:", X_pca.shape)
print("Explained variance ratio (first 2 components):", pca.explained_variance_ratio_)

# Plotting the PCA-transformed data
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Iris Species')
plt.grid(True)
plt.show()
```