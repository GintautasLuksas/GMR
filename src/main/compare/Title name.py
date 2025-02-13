import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load the datasets
random_clusters_df = pd.read_csv('Random_cluster_labels_with_embeddings.csv')
nn_clusters_df = pd.read_csv('NN_cluster_labels_with_embeddings.csv')

# Define columns to display
columns_to_show = [
    "Title", "Year", "Rating", "Length (minutes)", "Rating Amount", "Group",
    "Metascore", "Short Description", "Directors", "Star 1", "Star 2", "Star 3",
    "KMeans_Cluster", "Agglomerative_Cluster"
]

# Ensure selected columns exist in both datasets
columns_to_show = [col for col in columns_to_show if col in random_clusters_df.columns and col in nn_clusters_df.columns]

# Drop non-numeric columns before computing similarity
numeric_random_df = random_clusters_df.drop(columns=["Title"]).apply(pd.to_numeric, errors="coerce").fillna(0)
numeric_nn_df = nn_clusters_df.drop(columns=["Title"]).apply(pd.to_numeric, errors="coerce").fillna(0)

# Convert to numpy arrays for k-NN
random_cluster_embeddings = numeric_random_df.values
nn_cluster_embeddings = numeric_nn_df.values

# Ensure embeddings have the same dimensions
if random_cluster_embeddings.shape[1] > nn_cluster_embeddings.shape[1]:
    random_cluster_embeddings = random_cluster_embeddings[:, :nn_cluster_embeddings.shape[1]]
elif random_cluster_embeddings.shape[1] < nn_cluster_embeddings.shape[1]:
    pad_width = nn_cluster_embeddings.shape[1] - random_cluster_embeddings.shape[1]
    random_cluster_embeddings = np.pad(random_cluster_embeddings, ((0, 0), (0, pad_width)), mode="constant")

# Train k-NN model on NN dataset
knn = NearestNeighbors(n_neighbors=1, metric='euclidean')  # Change n_neighbors for more matches
knn.fit(nn_cluster_embeddings)

# Find the nearest neighbors for each random cluster movie
distances, indices = knn.kneighbors(random_cluster_embeddings)

# Store results for comparison
comparison_results = []
for random_index, nn_index in enumerate(indices.flatten()):
    random_movie = random_clusters_df.iloc[random_index]
    nn_movie = nn_clusters_df.iloc[nn_index]

    comparison_entry = {
        "Random Movie Title": random_movie["Title"],
        "Most Similar NN Movie Title": nn_movie["Title"],
    }
    for col in columns_to_show:
        comparison_entry[f"Random_{col}"] = random_movie[col]
        comparison_entry[f"NN_{col}"] = nn_movie[col]

    comparison_results.append(comparison_entry)

# Convert to DataFrame for better readability
comparison_df = pd.DataFrame(comparison_results)

# Display the full comparison
print(comparison_df.to_string())
