# Unsupervised Learning Concepts

Unsupervised learning is a type of machine learning where the model learns from unlabeled data, meaning there are no predefined output labels. The goal is to discover hidden patterns, structures, or relationships within the data itself. It's often used for exploratory data analysis, dimensionality reduction, and anomaly detection.

## Key Concepts:

### 1. Unlabeled Data

*   The dataset consists only of input features (X), without any corresponding output labels (y).

### 2. Common Tasks in Unsupervised Learning

*   **Clustering**: Grouping similar data points together based on their inherent characteristics.
*   **Dimensionality Reduction**: Reducing the number of features in a dataset while retaining as much important information as possible.
*   **Association Rule Mining**: Discovering relationships between variables in large datasets.
*   **Anomaly Detection**: Identifying rare items, events, or observations that deviate significantly from the majority of the data.

### 3. Common Unsupervised Learning Algorithms

#### A. Clustering Algorithms

*   **K-Means Clustering**: An iterative algorithm that partitions `n` observations into `k` clusters, where each observation belongs to the cluster with the nearest mean (centroid).
    *   **Centroid**: The mean position of all the points in a cluster.
    *   **Elbow Method**: A common technique to determine the optimal number of clusters (k).
*   **Hierarchical Clustering**: Builds a hierarchy of clusters. Can be:
    *   **Agglomerative (bottom-up)**: Each data point starts as a single cluster, and pairs of clusters are merged as one moves up the hierarchy.
    *   **Divisive (top-down)**: All data points start in one cluster, and splits are performed recursively as one moves down the hierarchy.
*   **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions.

#### B. Dimensionality Reduction Algorithms

*   **Principal Component Analysis (PCA)**: A linear dimensionality reduction technique that transforms the data into a new coordinate system where the greatest variance by some projection comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.
    *   **Principal Components**: New orthogonal (uncorrelated) variables that capture the most variance in the data.
    *   **Eigenvalues and Eigenvectors**: PCA relies heavily on these concepts from linear algebra.
*   **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: A non-linear dimensionality reduction technique particularly well-suited for visualizing high-dimensional datasets by mapping them to a lower-dimensional space (e.g., 2D or 3D) while preserving the local structure of the data.
*   **Autoencoders**: Neural networks used for unsupervised learning of efficient data codings (representations). They learn to compress the input into a lower-dimensional representation and then reconstruct it, aiming to minimize the reconstruction error.
*   **Independent Component Analysis (ICA)**: A computational method that separates a multivariate signal into additive subcomponents assuming the subcomponents are non-Gaussian and statistically independent from each other. It is often used for blind source separation, such as separating individual voices from a mixture of sounds.

## Resources:

*   **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron**
*   **Scikit-learn Documentation**
*   **Online courses on Unsupervised Learning**
