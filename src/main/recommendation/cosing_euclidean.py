import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

watched_clusters_df = pd.read_csv('clustered_data2.csv')
clusters_df = pd.read_csv('clustered_data.csv')


recommended_movies_set = set()

def clean_genre_column(genre_column):
    """Clean and standardize genre columns by removing extra spaces, converting to lowercase, and handling NaNs."""
    if isinstance(genre_column, str):
        return genre_column.strip().lower()
    return ""

df_list = [watched_clusters_df, clusters_df]
for df in df_list:
    for col in ["Genre 1", "Genre 2", "Genre 3"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_genre_column)

numeric_watched_df = watched_clusters_df.drop(columns=["Title", "Genre 1", "Genre 2", "Genre 3"], errors="ignore").apply(pd.to_numeric, errors="coerce").fillna(0)
numeric_nn_df = clusters_df.drop(columns=["Title", "Genre 1", "Genre 2", "Genre 3"], errors="ignore").apply(pd.to_numeric, errors="coerce").fillna(0)

cosine_weight = 0.3
euclidean_weight = 0.7

comparison_results_combined = []

for watched_index, watched_movie in watched_clusters_df.iterrows():
    watched_title = watched_movie["Title"]
    watched_genres = {watched_movie["Genre 1"], watched_movie["Genre 2"], watched_movie["Genre 3"]}
    watched_metascore = watched_movie["Metascore"]
    watched_rating = watched_movie["Rating"]
    watched_cluster = watched_movie["Agglomerative Cluster"]


    same_cluster_movies = clusters_df[clusters_df["Agglomerative Cluster"] == watched_cluster]


    if same_cluster_movies.empty:
        same_cluster_movies = clusters_df


    same_cluster_movies = same_cluster_movies[~same_cluster_movies["Title"].isin(recommended_movies_set)]
    if same_cluster_movies.empty:
        continue

    metascore_diff = np.abs(same_cluster_movies["Metascore"] - watched_metascore)
    rating_diff = np.abs(same_cluster_movies["Rating"] - watched_rating)
    similarity_score = metascore_diff + rating_diff

    most_similar_index = np.argmin(similarity_score)
    most_similar_movie = same_cluster_movies.iloc[most_similar_index]

    recommended_genres = {most_similar_movie["Genre 1"], most_similar_movie["Genre 2"], most_similar_movie["Genre 3"]}

    genre_match = len(watched_genres.intersection(recommended_genres)) > 0

    if genre_match:
        watched_numeric = numeric_watched_df.loc[watched_index].values.reshape(1, -1)
        recommended_index = clusters_df.index[clusters_df["Title"] == most_similar_movie["Title"]][0]
        recommended_numeric = numeric_nn_df.loc[recommended_index].values.reshape(1, -1)

        cosine_sim = cosine_similarity(watched_numeric, recommended_numeric)
        euclidean_dist = euclidean_distances(watched_numeric, recommended_numeric)

        combined_score = (cosine_weight * cosine_sim[0, 0] +
                          euclidean_weight * (1 / (1 + euclidean_dist[0, 0])))

        comparison_entry = {
            "Watched Movie Title": watched_title,
            "Recommended Movie Title": most_similar_movie["Title"],
            "Agglomerative Cluster": watched_cluster,
            "Combined Score": combined_score,
        }

        # Add movie details
        for col in ["Title", "Rating", "Length (mins)", "Rating Amount", "Metascore", "Short Description",
                    "Directors", "Star 1", "Star 2", "Star 3", "Genre 1", "Genre 2", "Genre 3",
                    "Agglomerative Cluster", "KMeans Cluster"]:
            comparison_entry[f"Watched_{col}"] = watched_movie[col]
            comparison_entry[f"Recommended_{col}"] = most_similar_movie[col]

        comparison_results_combined.append(comparison_entry)
        recommended_movies_set.add(most_similar_movie["Title"])

comparison_df_combined = pd.DataFrame(comparison_results_combined)
comparison_df_combined = comparison_df_combined.sort_values(by="Combined Score", ascending=False)

comparison_df_combined.to_csv('recommendation_results_combined.csv', index=False)

print("Final Recommendations Based on Agglomerative Cluster Filtering:")
print(comparison_df_combined.to_string())
