import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances
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

# Compute Manhattan Distances
manhattan_dist = manhattan_distances(random_cluster_embeddings, nn_cluster_embeddings)

# Store results for comparison
comparison_results_manhattan = []
for random_index in range(len(random_clusters_df)):
    random_movie = random_clusters_df.iloc[random_index]
    random_title = random_movie["Title"]

    # Get the most similar movie based on the smallest Manhattan distance
    most_similar_index = np.argmin(manhattan_dist[random_index])
    most_similar_movie = nn_clusters_df.iloc[most_similar_index]

    # Store the comparison
    comparison_entry = {
        "Random Movie Title": random_title,
        "Most Similar NN Movie Title": most_similar_movie["Title"],
    }
    comparison_results_manhattan.append(comparison_entry)

# Convert to DataFrame for better readability
comparison_df_manhattan = pd.DataFrame(comparison_results_manhattan)
print(comparison_df_manhattan.to_string())
