import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
from scipy.cluster.hierarchy import linkage

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings('ignore', category=UserWarning, message=".*Could not find the number of physical cores*")

input_path = r"/src/system_test_data\normalize_comparison\normalized_data2.csv"
output_path = r"/src/system_test_data\normalize_comparison\clustered_data2.csv"

data = pd.read_csv(input_path)

features_for_clustering = ['Rating', 'Length (mins)', 'Rating Amount', 'Metascore']
X = data[features_for_clustering]

n_clusters_kmeans = 3
kmeans = KMeans(n_clusters=n_clusters_kmeans, n_init=10, random_state=42)
data['KMeans Cluster'] = kmeans.fit_predict(X)

n_clusters_agglo = 3
agglo = AgglomerativeClustering(n_clusters=n_clusters_agglo)
data['Agglomerative Cluster'] = agglo.fit_predict(X)


def compute_silhouette(X, labels, algorithm_name):
    """Computes the silhouette score for a clustering algorithm if multiple clusters exist."""
    if len(np.unique(labels)) > 1:
        score = silhouette_score(X, labels)
        print(f"Silhouette Score for {algorithm_name}: {score:.3f}")
        return score
    print(f"{algorithm_name} did not form multiple clusters, skipping silhouette score.")
    return None


kmeans_score = compute_silhouette(X, data['KMeans Cluster'], 'K-Means')
agglo_score = compute_silhouette(X, data['Agglomerative Cluster'], 'Agglomerative')

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

data.to_csv(output_path, index=False)
print(f"Clustered data saved to {output_path}")
