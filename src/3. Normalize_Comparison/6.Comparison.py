import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

"""Load normalized data, perform clustering using KMeans and Agglomerative Clustering,
compute silhouette scores, visualize the clusters with PCA, and inspect feature importance."""
input_path = r"../4. Neuro/Normalized_Data.csv"

"""Load the dataset and select relevant features for clustering."""
data = pd.read_csv(input_path)

features_for_clustering = ['Rating', 'Year', 'Length (minutes)', 'Rating Amount', 'Metascore', 'Weighted Rating']
X = data[features_for_clustering]

"""Perform KMeans clustering with 3 clusters and store the cluster labels."""
n_clusters_kmeans = 3
kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
data['KMeans Cluster'] = kmeans.fit_predict(X)

"""Perform Agglomerative Clustering with 3 clusters and store the cluster labels."""
n_clusters_agglo = 3
agglo = AgglomerativeClustering(n_clusters=n_clusters_agglo)
data['Agglomerative Cluster'] = agglo.fit_predict(X)

"""Compute silhouette score for a given clustering algorithm if it forms multiple clusters."""
def compute_silhouette(X, labels, algorithm_name):
    if len(np.unique(labels)) > 1:
        score = silhouette_score(X, labels)
        print(f"Silhouette Score for {algorithm_name}: {score:.3f}")
        return score
    else:
        print(f"{algorithm_name} did not form multiple clusters, skipping silhouette score.")
        return None

silhouette_scores = ['K-Means', 'Agglomerative']

"""Compute silhouette score for KMeans and Agglomerative clustering algorithms."""
kmeans_score = compute_silhouette(X, data['KMeans Cluster'], 'K-Means')
if kmeans_score is not None:
    silhouette_scores.append(kmeans_score)

agglo_score = compute_silhouette(X, data['Agglomerative Cluster'], 'Agglomerative')
if agglo_score is not None:
    silhouette_scores.append(agglo_score)

"""Perform PCA for dimensionality reduction to visualize the clusters in 2D."""
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

"""Visualize the clusters formed by KMeans and Agglomerative Clustering using PCA."""
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

"""Create a bar plot comparing the silhouette scores of the clustering algorithms."""
silhouette_scores_final = []
silhouette_scores_final.append(kmeans_score if kmeans_score is not None else 0)
silhouette_scores_final.append(agglo_score if agglo_score is not None else 0)

"""Save the dataset with the clustering labels into a new CSV file."""
output_path = r"With_clusters_data.csv"
data.to_csv(output_path, index=False)
