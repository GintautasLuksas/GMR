import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Read the CSV files for random and nn clusters
watched_clusters_df = pd.read_csv('clustered_data.csv')
clusters_df = pd.read_csv('clustered_data2.csv')

# Clean and format genre columns (remove extra spaces, lowercase, and handle NaNs)
def clean_genre_column(genre_column):
    """
    Clean and standardize genre columns by removing extra spaces, converting to lowercase, and handling NaNs.
    """
    if isinstance(genre_column, str):
        return genre_column.strip().lower()  # Remove spaces and convert to lowercase
    return ""

# Apply genre cleaning to both random and nn clusters
watched_clusters_df["Genre 1"] = watched_clusters_df["Genre 1"].apply(clean_genre_column)
clusters_df["Genre 1"] = clusters_df["Genre 1"].apply(clean_genre_column)
clusters_df["Genre 2"] = clusters_df["Genre 2"].apply(clean_genre_column)
clusters_df["Genre 3"] = clusters_df["Genre 3"].apply(clean_genre_column)

# Convert relevant columns to numeric and fill missing values with zeros
numeric_random_df = watched_clusters_df.drop(columns=["Title", "Genre 1"]).apply(pd.to_numeric, errors="coerce").fillna(0)
numeric_nn_df = clusters_df.drop(columns=["Title", "Genre 1"]).apply(pd.to_numeric, errors="coerce").fillna(0)

# Compute cosine similarity and Euclidean distances between the numeric features of the random and nn clusters
cosine_sim = cosine_similarity(numeric_random_df, numeric_nn_df)
distance_matrix = euclidean_distances(numeric_random_df, numeric_nn_df)

# Define weights for cosine similarity and Euclidean distance
cosine_weight = 0.5
euclidean_weight = 0.5

# Initialize a list to store the comparison results
comparison_results_combined = []

# Loop over each movie in the random clusters
for random_index in range(len(watched_clusters_df)):
    random_movie = watched_clusters_df.iloc[random_index]
    random_title = random_movie["Title"]
    random_genre_1 = random_movie["Genre 1"]

    # Find the most similar movie based on cosine similarity
    most_similar_index_cosine = np.argmax(cosine_sim[random_index])
    most_similar_movie_cosine = clusters_df.iloc[most_similar_index_cosine]

    # Extract the genres from the recommended movie and clean them
    recommended_genres = {
        most_similar_movie_cosine["Genre 1"],
        most_similar_movie_cosine["Genre 2"],
        most_similar_movie_cosine["Genre 3"]
    }

    # Relax the genre match: Check if any genre from random movie overlaps with recommended genres
    random_movie_genres = {random_genre_1}
    genre_match = len(random_movie_genres.intersection(recommended_genres)) > 0

    # If there's a genre match, compute a combined score based on cosine similarity and Euclidean distance
    if genre_match:
        # Compute a combined score based on cosine similarity and Euclidean distance
        combined_score = (cosine_weight * cosine_sim[random_index, most_similar_index_cosine] +
                          euclidean_weight * (1 / (1 + distance_matrix[random_index, most_similar_index_cosine])))

        # Store the comparison results for this movie
        comparison_entry = {
            "Random Movie Title": random_title,
            "Recommended NN Movie Title": most_similar_movie_cosine["Title"],
            "Combined Score": combined_score,
        }

        # Add movie details to the comparison entry
        for col in ["Title" , "Rating", "Length (mins)", "Rating Amount", "Group",
                    "Metascore", "Short Description", "Directors", "Star 1", "Star 2", "Star 3",
                    "Genre 1", "Genre 2", "Genre 3", "KMeans Cluster", "Agglomerative Cluster"]:
            comparison_entry[f"Random_{col}"] = random_movie[col]
            comparison_entry[f"NN_{col}"] = most_similar_movie_cosine[col]

        comparison_results_combined.append(comparison_entry)

# Convert the comparison results to a DataFrame
comparison_df_combined = pd.DataFrame(comparison_results_combined)

# Sort the DataFrame by the combined score in descending order
comparison_df_combined = comparison_df_combined.sort_values(by="Combined Score", ascending=False)

# Remove the "Genre Match" column (if it exists in the DataFrame)
comparison_df_combined = comparison_df_combined.drop(columns=["Genre Match"], errors='ignore')

# Print the final results
print("Combined Cosine and Euclidean Distance Recommendation (without Genre Match):")
print(comparison_df_combined.to_string())

# Optionally, save the results to a CSV file for further analysis
comparison_df_combined.to_csv('recommendation_results_combined.csv', index=False)
