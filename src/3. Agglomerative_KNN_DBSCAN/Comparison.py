import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# File path for input
input_path = r"Normalized_Data.csv"

# Load normalized data
data = pd.read_csv(input_path)

# Features for clustering (including 'Rating Amount')
features_for_clustering = ['Rating', 'Year', 'Length (minutes)', 'Rating Amount', 'Metascore', 'Weighted Rating']
X = data[features_for_clustering]

# ---- K-Means Clustering ----
n_clusters_kmeans = 2  # Change the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
data['KMeans Cluster'] = kmeans.fit_predict(X)

# ---- DBSCAN Clustering ----
eps_dbscan = 0.5  # Max distance between points to be considered in the same cluster
min_samples_dbscan = 5  # Minimum number of samples to form a cluster
dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
data['DBSCAN Cluster'] = dbscan.fit_predict(X)

# ---- Agglomerative Clustering ----
n_clusters_agglo = 2  # Change the number of clusters as needed
agglo = AgglomerativeClustering(n_clusters=n_clusters_agglo)
data['Agglomerative Cluster'] = agglo.fit_predict(X)

# ---- Silhouette Score Comparison ----
def compute_silhouette(X, labels, algorithm_name):
    if len(np.unique(labels)) > 1:  # Only compute silhouette score if more than one cluster
        score = silhouette_score(X, labels)
        print(f"Silhouette Score for {algorithm_name}: {score:.3f}")
        return score
    else:
        print(f"{algorithm_name} did not form multiple clusters, skipping silhouette score.")
        return None

# Compute Silhouette Scores
silhouette_scores = []
algorithms = ['K-Means', 'Agglomerative']

# Compute silhouette score for K-Means
kmeans_score = compute_silhouette(X, data['KMeans Cluster'], 'K-Means')
if kmeans_score is not None:
    silhouette_scores.append(kmeans_score)

# Compute silhouette score for Agglomerative Clustering
agglo_score = compute_silhouette(X, data['Agglomerative Cluster'], 'Agglomerative')
if agglo_score is not None:
    silhouette_scores.append(agglo_score)

# Compute silhouette score for DBSCAN if it forms multiple clusters
if len(np.unique(data['DBSCAN Cluster'])) > 1:
    dbscan_score = compute_silhouette(X, data['DBSCAN Cluster'], 'DBSCAN')
    silhouette_scores.append(dbscan_score)
else:
    silhouette_scores.append(0)  # Add 0 for DBSCAN if it did not form multiple clusters

# ---- PCA for dimensionality reduction (2D visualization) ----
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# ---- Visualize the clusters (optional) ----
plt.figure(figsize=(15, 6))

# K-Means Plot
plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['KMeans Cluster'], cmap='viridis', alpha=0.7)
plt.title('K-Means Clustering (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# DBSCAN Plot
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['DBSCAN Cluster'], cmap='plasma', alpha=0.7)
plt.title('DBSCAN Clustering (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Agglomerative Plot
plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Agglomerative Cluster'], cmap='inferno', alpha=0.7)
plt.title('Agglomerative Clustering (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.tight_layout()
plt.show()

# ---- Silhouette Score Comparison Bar Plot ----
if len(np.unique(data['DBSCAN Cluster'])) > 1:
    algorithms.append('DBSCAN')

# Adjust silhouette_scores list dynamically
silhouette_scores_final = []
silhouette_scores_final.append(kmeans_score if kmeans_score is not None else 0)
silhouette_scores_final.append(agglo_score if agglo_score is not None else 0)
if len(np.unique(data['DBSCAN Cluster'])) > 1:
    silhouette_scores_final.append(dbscan_score if dbscan_score is not None else 0)

plt.figure(figsize=(8, 5))
sns.barplot(x=algorithms, y=silhouette_scores_final, palette='Blues_d')
plt.title('Silhouette Score Comparison')
plt.xlabel('Clustering Algorithm')
plt.ylabel('Silhouette Score')
plt.ylim(0, 1)
plt.show()

# ---- PCA Component Inspection ----
print("PCA Components (coefficients of features in the principal components):")
print(pca.components_)

# Display feature contributions to each component (i.e., loadings of each feature on each principal component)
components_df = pd.DataFrame(pca.components_, columns=features_for_clustering)
print("\nFeature Contributions to Each Principal Component:")
print(components_df)

# Visualize the feature contributions to the components
plt.figure(figsize=(10, 6))
sns.heatmap(components_df, annot=True, cmap="coolwarm", cbar=True)
plt.title("Feature Contributions to PCA Components")
plt.xlabel('Features')
plt.ylabel('Principal Components')
plt.show()

# ---- Check Feature Importance in KMeans ----
# Checking feature importance in K-Means based on the centroids
print("\nK-Means Cluster Centers (Feature Importance):")
kmeans_centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=features_for_clustering)
print(kmeans_centers_df)

# Visualize feature importance (centroids) in K-Means
plt.figure(figsize=(10, 6))
sns.heatmap(kmeans_centers_df, annot=True, cmap="coolwarm", cbar=True)
plt.title("Feature Importance in K-Means Clusters")
plt.xlabel('Features')
plt.ylabel('Clusters')
plt.show()
