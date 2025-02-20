import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

random_clusters_df = pd.read_csv('nn_cluster2.csv')
nn_clusters_df = pd.read_csv('nn_cluster.csv')

def clean_genre_column(genre_column):
    """
    Clean and standardize genre columns by removing extra spaces, converting to lowercase, and handling NaNs.
    """
    if isinstance(genre_column, str):
        return genre_column.strip().lower()
    return ""

random_clusters_df["Genre 1"] = random_clusters_df["Genre 1"].apply(clean_genre_column)
nn_clusters_df["Genre 1"] = nn_clusters_df["Genre 1"].apply(clean_genre_column)
nn_clusters_df["Genre 2"] = nn_clusters_df["Genre 2"].apply(clean_genre_column)
nn_clusters_df["Genre 3"] = nn_clusters_df["Genre 3"].apply(clean_genre_column)

def prepare_numeric_data(df):
    """
    Prepare numeric data by dropping non-numeric columns and filling missing values with zeros.
    """
    return df.drop(columns=["Title", "Genre 1"]).apply(pd.to_numeric, errors="coerce").fillna(0)

numeric_random_df = prepare_numeric_data(random_clusters_df)
numeric_nn_df = prepare_numeric_data(nn_clusters_df)

random_cluster_embeddings = numeric_random_df.values
nn_cluster_embeddings = numeric_nn_df.values

cosine_sim = cosine_similarity(random_cluster_embeddings, nn_cluster_embeddings)
distance_matrix = euclidean_distances(random_cluster_embeddings, nn_cluster_embeddings)

cosine_weight = 0.5
euclidean_weight = 0.5

comparison_results_combined = []
recommended_titles = set()

watched_movie_count = len(random_clusters_df)

for random_index in range(len(random_clusters_df)):
    random_movie = random_clusters_df.iloc[random_index]
    random_title = random_movie["Title"]
    random_genres = {random_movie["Genre 1"], random_movie.get("Genre 2", ""), random_movie.get("Genre 3", "")}

    # Find the most similar movies based on cosine similarity
    most_similar_indices_cosine = np.argsort(cosine_sim[random_index])[::-1]

    # Track if a recommendation has been made for this random movie
    recommendation_made = False

    # Check the top recommendations for each random movie
    for idx in most_similar_indices_cosine:
        recommended_movie_title = nn_clusters_df.iloc[idx]["Title"]

        if recommended_movie_title in recommended_titles:
            continue

        recommended_genres = {
            nn_clusters_df.iloc[idx]["Genre 1"],
            nn_clusters_df.iloc[idx].get("Genre 2", ""),
            nn_clusters_df.iloc[idx].get("Genre 3", "")
        }

        genre_match = len(random_genres.intersection(recommended_genres)) > 0 or not random_genres

        combined_score = (cosine_weight * cosine_sim[random_index, idx] +
                          euclidean_weight * (1 / (1 + distance_matrix[random_index, idx])))

        comparison_entry = {
            "Random Movie Title": random_title,
            "Recommended NN Movie Title": recommended_movie_title,
            "Combined Score": combined_score,
            "Genre Match": genre_match
        }

        for col in ["Title", "Year", "Rating", "Length (mins)", "Rating Amount", "Group",
                    "Metascore", "Short Description", "Directors", "Star 1", "Star 2", "Star 3",
                    "Genre 1", "Genre 2", "Genre 3", "Agglomerative_Cluster"]:  # Changed to Agglomerative_Cluster
            comparison_entry[f"Random_{col}"] = random_movie[col]
            comparison_entry[f"NN_{col}"] = nn_clusters_df.iloc[idx][col]

        comparison_results_combined.append(comparison_entry)
        recommended_titles.add(recommended_movie_title)

        recommendation_made = True  # Mark that a recommendation has been made
        break  # Exit the loop after making a recommendation

    if len(recommended_titles) >= watched_movie_count:
        break  # Stop if we have enough recommendations

comparison_df_combined = pd.DataFrame(comparison_results_combined)

comparison_df_combined = comparison_df_combined.sort_values(by="Combined Score", ascending=False)


comparison_df_combined = comparison_df_combined.head(watched_movie_count)

print("Combined Cosine and Euclidean Distance Recommendation (with Genre Match):")
print(comparison_df_combined.to_string(index=False))
