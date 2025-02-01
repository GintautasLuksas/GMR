import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Input, Dense
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def assign_cluster(movie_title: str, data: pd.DataFrame) -> tuple:
    """
    Assigns cluster labels to a movie based on its title.

    :param movie_title: The title of the movie to find.
    :param data: The DataFrame containing movie data and cluster labels.
    :return: A tuple containing K-Means and Agglomerative cluster labels.
    """
    movie = data[data['Title'].str.lower() == movie_title.lower()]
    if movie.empty:
        return "Movie not found."
    return int(movie['KMeans_Cluster'].values[0]), int(movie['Agglomerative_Cluster'].values[0])


# Step 1: Load the data
data = pd.read_csv('IMDB710_Cleaned.csv')

# Step 2: Preprocess numerical data
numerical_columns = ['Rating', 'Length (minutes)', 'Rating Amount', 'Metascore']
numerical_data = data[numerical_columns]

# Normalize the numerical data
scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(numerical_data)

# Step 3: Preprocess categorical and text data
data['Group'] = data['Group'].map({'R': 3, 'PG-13': 2, 'Approved': 1})  # Avoid Group 0

# Convert text data ('Short Description') to embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
description_embeddings = model.encode(data['Short Description'].tolist())

# Step 4: Combine features
final_features = np.concatenate((scaled_numerical_data, description_embeddings), axis=1)

# Step 5: Train Autoencoder
input_layer = Input(shape=(final_features.shape[1],))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(final_features.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(final_features, final_features, epochs=50, batch_size=256, shuffle=True)

# Extract encoded features
encoder = Model(input_layer, encoded)
encoded_features = encoder.predict(final_features)

# Step 7: Clustering (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(encoded_features)

agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_clustering.fit_predict(encoded_features)

# Assign clusters
data['KMeans_Cluster'] = kmeans_clusters
data['Agglomerative_Cluster'] = agg_clusters

data.to_csv('IMDB710_Clustered.csv', index=False)

# Example Usage
movie_name = "Inception"
kmeans_label, agg_label = assign_cluster(movie_name, data)
print(f"Movie: {movie_name} -> KMeans Cluster: {kmeans_label}, Agglomerative Cluster: {agg_label}")
