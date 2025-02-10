import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Load the datasets
random_clusters_df = pd.read_csv('Random_cluster_labels_with_embeddings.csv')
nn_clusters_df = pd.read_csv('NN_cluster_labels_with_embeddings.csv')

# Define the columns to display
columns_to_show = [
    "Title", "Year", "Rating", "Length (minutes)", "Rating Amount", "Group",
    "Metascore", "Short Description", "Directors", "Star 1", "Star 2", "Star 3",
    "KMeans_Cluster", "Agglomerative_Cluster"
]

# Ensure selected columns exist in both datasets
columns_to_show = [col for col in columns_to_show if col in random_clusters_df.columns and col in nn_clusters_df.columns]

# Drop non-numeric columns before computing distances
numeric_random_df = random_clusters_df.drop(columns=["Title"]).apply(pd.to_numeric, errors="coerce").fillna(0)
numeric_nn_df = nn_clusters_df.drop(columns=["Title"]).apply(pd.to_numeric, errors="coerce").fillna(0)

# Convert to numpy arrays for distance computation
random_cluster_embeddings = numeric_random_df.values
nn_cluster_embeddings = numeric_nn_df.values

# Ensure embeddings have the same dimensions
if random_cluster_embeddings.shape[1] > nn_cluster_embeddings.shape[1]:
    random_cluster_embeddings = random_cluster_embeddings[:, :nn_cluster_embeddings.shape[1]]
elif random_cluster_embeddings.shape[1] < nn_cluster_embeddings.shape[1]:
    pad_width = nn_cluster_embeddings.shape[1] - random_cluster_embeddings.shape[1]
    random_cluster_embeddings = np.pad(random_cluster_embeddings, ((0, 0), (0, pad_width)), mode="constant")

# Compute Euclidean distances
distance_matrix = euclidean_distances(random_cluster_embeddings, nn_cluster_embeddings)

# Store results for comparison
comparison_results = []
for random_index in range(len(random_clusters_df)):
    random_movie = random_clusters_df.iloc[random_index]
    random_title = random_movie["Title"]
    random_cluster = random_movie["KMeans_Cluster"]

    # Filter NN movies to only include those in the same KMeans cluster
    same_cluster_nn_indices = nn_clusters_df[nn_clusters_df["KMeans_Cluster"] == random_cluster].index.tolist()

    if not same_cluster_nn_indices:
        # No movies in the same cluster, mark as "No match found"
        comparison_results.append({
            "Random Movie Title": random_title,
            "Most Similar NN Movie Title": "No Match Found",
            **{f"Random_{col}": random_movie[col] for col in columns_to_show},
            **{f"NN_{col}": "N/A" for col in columns_to_show}
        })
        continue

    # Extract the distances only for movies in the same cluster
    distances = distance_matrix[random_index, same_cluster_nn_indices]

    # Find the closest NN movie within the same cluster
    closest_nn_index = same_cluster_nn_indices[np.argmin(distances)]
    closest_nn_movie = nn_clusters_df.iloc[closest_nn_index]

    # Store the comparison
    comparison_entry = {
        "Random Movie Title": random_title,
        "Most Similar NN Movie Title": closest_nn_movie["Title"],
    }
    for col in columns_to_show:
        comparison_entry[f"Random_{col}"] = random_movie[col]
        comparison_entry[f"NN_{col}"] = closest_nn_movie[col]

    comparison_results.append(comparison_entry)

# Convert to DataFrame for better readability
comparison_df = pd.DataFrame(comparison_results)

# Display the full comparison
print(comparison_df.to_string())
