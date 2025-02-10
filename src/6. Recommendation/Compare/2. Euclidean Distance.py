import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# Load the datasets
random_clusters_df = pd.read_csv('Random_cluster_labels_with_embeddings.csv')
nn_clusters_df = pd.read_csv('NN_cluster_labels_with_embeddings.csv')

# Drop non-numeric columns before computing distances
numeric_random_df = random_clusters_df.drop(columns=["Title"]).apply(pd.to_numeric, errors="coerce").fillna(0)
numeric_nn_df = nn_clusters_df.drop(columns=["Title"]).apply(pd.to_numeric, errors="coerce").fillna(0)

# Convert to numpy arrays for distance computation
random_cluster_embeddings = numeric_random_df.values
nn_cluster_embeddings = numeric_nn_df.values

# Compute Euclidean Distances
distance_matrix = euclidean_distances(random_cluster_embeddings, nn_cluster_embeddings)

# Store results for comparison
comparison_results_euclidean = []
for random_index in range(len(random_clusters_df)):
    random_movie = random_clusters_df.iloc[random_index]
    random_title = random_movie["Title"]

    # Get the most similar movie based on the smallest Euclidean distance
    most_similar_index = np.argmin(distance_matrix[random_index])
    most_similar_movie = nn_clusters_df.iloc[most_similar_index]

    # Store the comparison
    comparison_entry = {
        "Random Movie Title": random_title,
        "Most Similar NN Movie Title": most_similar_movie["Title"],
    }
    comparison_results_euclidean.append(comparison_entry)

# Convert to DataFrame for better readability
comparison_df_euclidean = pd.DataFrame(comparison_results_euclidean)
print(comparison_df_euclidean.to_string())
