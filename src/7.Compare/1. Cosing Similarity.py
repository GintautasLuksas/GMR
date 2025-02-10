import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

random_clusters_df = pd.read_csv('Random_cluster_labels_with_embeddings.csv')
nn_clusters_df = pd.read_csv('NN_cluster_labels_with_embeddings.csv')

random_clusters_df = random_clusters_df[random_clusters_df["Title"] != "12 Angry Men"]
nn_clusters_df = nn_clusters_df[nn_clusters_df["Title"] != "12 Angry Men"]

columns_to_show = [
    "Title", "Year", "Rating", "Length (minutes)", "Rating Amount", "Group",
    "Metascore", "Short Description", "Directors", "Star 1", "Star 2", "Star 3",
    "KMeans_Cluster", "Agglomerative_Cluster"
]

columns_to_show = [col for col in columns_to_show if
                   col in random_clusters_df.columns and col in nn_clusters_df.columns]

numeric_random_df = random_clusters_df.drop(columns=["Title"]).apply(pd.to_numeric, errors="coerce").fillna(0)
numeric_nn_df = nn_clusters_df.drop(columns=["Title"]).apply(pd.to_numeric, errors="coerce").fillna(0)

random_cluster_embeddings = numeric_random_df.values
nn_cluster_embeddings = numeric_nn_df.values

cosine_sim = cosine_similarity(random_cluster_embeddings, nn_cluster_embeddings)

distance_matrix = euclidean_distances(random_cluster_embeddings, nn_cluster_embeddings)


cosine_weight = 0.5
euclidean_weight = 0.5

comparison_results_combined = []
for random_index in range(len(random_clusters_df)):
    random_movie = random_clusters_df.iloc[random_index]
    random_title = random_movie["Title"]

    most_similar_index_cosine = np.argmax(cosine_sim[random_index])
    most_similar_movie_cosine = nn_clusters_df.iloc[most_similar_index_cosine]

    random_cluster = random_movie["KMeans_Cluster"]
    same_cluster_nn_indices = nn_clusters_df[nn_clusters_df["KMeans_Cluster"] == random_cluster].index.tolist()
    same_cluster_nn_indices = [i for i in same_cluster_nn_indices if i < len(nn_clusters_df)]

    if same_cluster_nn_indices:
        distances = distance_matrix[random_index, same_cluster_nn_indices]
        closest_nn_index = same_cluster_nn_indices[np.argmin(distances)]
        closest_nn_movie = nn_clusters_df.iloc[closest_nn_index]
    else:
        closest_nn_movie = None

    combined_score = (cosine_weight * cosine_sim[random_index, most_similar_index_cosine] +
                      euclidean_weight * (1 / (1 + distance_matrix[random_index, most_similar_index_cosine])))

    if combined_score >= 0.5:
        best_movie = most_similar_movie_cosine
    else:
        best_movie = closest_nn_movie if closest_nn_movie is not None else most_similar_movie_cosine

    # Store comparison result
    comparison_entry = {
        "Random Movie Title": random_title,
        "Recommended NN Movie Title": best_movie["Title"],
        "Combined Score": combined_score,
    }

    for col in columns_to_show:
        comparison_entry[f"Random_{col}"] = random_movie[col]
        comparison_entry[f"NN_{col}"] = best_movie[col]

    comparison_results_combined.append(comparison_entry)

comparison_df_combined = pd.DataFrame(comparison_results_combined)

comparison_df_combined = comparison_df_combined.sort_values(by="Combined Score", ascending=False)


print("Combined Cosine and Euclidean Distance Recommendation:")
print(comparison_df_combined.to_string())
