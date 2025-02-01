import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Step 1: Load the data
data = pd.read_csv('IMDB710_Cleaned.csv')

# Step 2: Preprocess numerical data
numerical_columns = ['Rating', 'Length (minutes)', 'Rating Amount', 'Metascore']
numerical_data = data[numerical_columns]

# Normalize the numerical data
scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(numerical_data)

# Step 3: Preprocess categorical and text data
# Encode the 'Group' column
data['Group'] = data['Group'].map({'R': 1, 'PG-13': 0, 'Approved': 0})  # Example encoding

# Convert text data ('Short Description') to embeddings using Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')
description_embeddings = model.encode(data['Short Description'].tolist())

# Step 4: Combine the numerical and text embeddings
final_features = np.concatenate((scaled_numerical_data, description_embeddings), axis=1)

# Step 5: Build and Train the Autoencoder
input_layer = Input(shape=(final_features.shape[1],))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)  # Bottleneck layer
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(final_features.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(final_features, final_features, epochs=50, batch_size=256, shuffle=True)

# Step 6: Extract Features from the Encoder
encoder = Model(input_layer, encoded)
encoded_features = encoder.predict(final_features)

# Step 7: Apply Clustering (K-Means)
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust the number of clusters as needed
kmeans_clusters = kmeans.fit_predict(encoded_features)

# Step 7b: Apply Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=5)  # Adjust the number of clusters as needed
agg_clusters = agg_clustering.fit_predict(encoded_features)

# Add the clusters to the data
data['KMeans_Cluster'] = kmeans_clusters
data['Agglomerative_Cluster'] = agg_clusters

# Step 8: Visualize the Clusters using PCA
# Reduce dimensionality to 2D using PCA for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(encoded_features)

# Plot the K-Means clusters
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['KMeans_Cluster'], cmap='viridis')
plt.title('Movie Clusters (Using K-Means)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')

# Plot the Agglomerative clusters
plt.subplot(1, 2, 2)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['Agglomerative_Cluster'], cmap='viridis')
plt.title('Movie Clusters (Using Agglomerative Clustering)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.show()

# Step 9: PCA Component Weights (Explained Variance Ratio)
print("Explained Variance Ratio (PCA Components):")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"Component {i+1}: {var:.4f}")

# The sum of the explained variance ratio will give you the total variance explained by the selected components
total_variance_explained = np.sum(pca.explained_variance_ratio_)
print(f"Total Variance Explained by 2 Components: {total_variance_explained:.4f}")

# Step 10: Calculate Silhouette Score for K-Means
sil_score_kmeans = silhouette_score(encoded_features, kmeans_clusters)
print(f"Silhouette Score (K-Means): {sil_score_kmeans:.4f}")

# Step 11: Calculate Silhouette Score for Agglomerative Clustering
sil_score_agg = silhouette_score(encoded_features, agg_clusters)
print(f"Silhouette Score (Agglomerative Clustering): {sil_score_agg:.4f}")
