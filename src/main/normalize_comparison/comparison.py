import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
from scipy.cluster.hierarchy import linkage, dendrogram

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings('ignore', category=UserWarning, message=".*Could not find the number of physical cores*")

input_path = r"C:\Users\user\PycharmProjects\GMR\src\main\normalize_comparison\normalized_data.csv"
output_path = r"C:\Users\user\PycharmProjects\GMR\src\main\normalize_comparison\clustered_data.csv"

data = pd.read_csv(input_path)
features_for_clustering = ['Rating', 'Length (mins)', 'Rating Amount', 'Metascore']
X = data[features_for_clustering]


def compute_elbow(X):
    """Computes the elbow method to find the optimal number of clusters for KMeans."""
    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(8, 6))
    plt.plot(K_range, inertias, marker='o', color='b', linestyle='--')
    plt.title('Elbow Method for KMeans')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()


compute_elbow(X)

n_clusters_kmeans = 3
kmeans = KMeans(n_clusters=n_clusters_kmeans, n_init=10, random_state=42)
data['KMeans Cluster'] = kmeans.fit_predict(X)

# Agglomerative Clustering
n_clusters_agglo = 3
agglo = AgglomerativeClustering(n_clusters=n_clusters_agglo)
data['Agglomerative Cluster'] = agglo.fit_predict(X)


def compute_silhouette(X, labels, algorithm_name):
    """Computes the silhouette score for a given clustering algorithm if it forms multiple clusters."""
    if len(np.unique(labels)) > 1:
        score = silhouette_score(X, labels)
        print(f"Silhouette Score for {algorithm_name}: {score:.3f}")
        return score
    else:
        print(f"{algorithm_name} did not form multiple clusters, skipping silhouette score.")
        return None


kmeans_score = compute_silhouette(X, data['KMeans Cluster'], 'K-Means')
agglo_score = compute_silhouette(X, data['Agglomerative Cluster'], 'Agglomerative')


def compute_gap_statistic(X, k_range):
    """Computes the Gap Statistic to find the optimal number of clusters."""
    inertias = []
    reference_inertia = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

        # Create random clustering reference
        random_data = np.random.random_sample(size=X.shape)
        kmeans_random = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans_random.fit(random_data)
        reference_inertia.append(kmeans_random.inertia_)

    gap_statistic = [np.log(reference_inertia[k]) - np.log(inertias[k]) for k in range(len(k_range))]

    plt.plot(k_range, gap_statistic, marker='o', color='b', linestyle='--')
    plt.title('Gap Statistic Method for KMeans')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Gap Statistic')
    plt.show()


# Compute Gap Statistic for a range of clusters
compute_gap_statistic(X, range(1, 11))

# Davies-Bouldin Score for KMeans
davies_bouldin_kmeans = davies_bouldin_score(X, data['KMeans Cluster'])
print(f"Davies-Bouldin Score for KMeans with {n_clusters_kmeans} clusters: {davies_bouldin_kmeans:.3f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['KMeans Cluster'], cmap='viridis', alpha=0.7)
plt.title('K-Means Clustering (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Agglomerative Cluster'], cmap='inferno', alpha=0.7)
plt.title('Agglomerative Clustering (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.tight_layout()
plt.show()

silhouette_scores = [kmeans_score if kmeans_score is not None else 0, agglo_score if agglo_score is not None else 0]
algorithms = ['K-Means', 'Agglomerative']

plt.figure(figsize=(8, 6))
plt.bar(algorithms, silhouette_scores, color=['skyblue', 'salmon'])
plt.title('Silhouette Score Comparison')
plt.ylabel('Silhouette Score')
plt.ylim([0, 1])
plt.show()

centroids = kmeans.cluster_centers_
print("\nKMeans Cluster Centroids and Feature Importance:")
for i, centroid in enumerate(centroids):
    print(f"Centroid {i}: {dict(zip(features_for_clustering, centroid))}")

Z = linkage(X, method='ward')
print("\nAgglomerative Clustering Linkage Matrix:")
print(Z)

plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

data.to_csv(output_path, index=False)
print(f"Clustered data saved to {output_path}")
