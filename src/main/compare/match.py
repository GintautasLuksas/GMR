import pandas as pd

# Read the CSV files for random and nn clusters
random_clusters_df = pd.read_csv('nn_cluster2.csv')
nn_clusters_df = pd.read_csv('nn_cluster.csv')

# Clean genre columns by removing any extra spaces
def clean_genre_column(genre_column):
    """
    Clean and standardize genre columns by removing extra spaces.
    """
    if isinstance(genre_column, str):
        return genre_column.strip()  # Remove any surrounding spaces
    return ""

# Apply genre cleaning to both random and nn clusters
random_clusters_df["Genre 1"] = random_clusters_df["Genre 1"].apply(clean_genre_column)
nn_clusters_df["Genre 1"] = nn_clusters_df["Genre 1"].apply(clean_genre_column)
nn_clusters_df["Genre 2"] = nn_clusters_df["Genre 2"].apply(clean_genre_column)
nn_clusters_df["Genre 3"] = nn_clusters_df["Genre 3"].apply(clean_genre_column)

# Initialize a list to store the results of genre matches
genre_match_results = []

# Loop over each movie in the random clusters
for random_index in range(len(random_clusters_df)):
    random_movie = random_clusters_df.iloc[random_index]
    random_title = random_movie["Title"]
    random_genre_1 = random_movie["Genre 1"]

    # Loop over each movie in the nn clusters to find genre matches
    for nn_index in range(len(nn_clusters_df)):
        nn_movie = nn_clusters_df.iloc[nn_index]
        nn_genres = {nn_movie["Genre 1"], nn_movie["Genre 2"], nn_movie["Genre 3"]}

        # Check if the genre from the random movie matches any genre in the recommended movie
        genre_match = random_genre_1 in nn_genres

        # If genres match, store the results
        if genre_match:
            genre_match_results.append({
                "Random Movie Title": random_title,
                "Recommended NN Movie Title": nn_movie["Title"],
                "Random Genre": random_genre_1,
                "NN Genres": nn_genres,
                "Genre Match": genre_match
            })

# Convert the results into a DataFrame
genre_match_df = pd.DataFrame(genre_match_results)

# Print the final results with genre matches
print("Genre Match Recommendations:")
print(genre_match_df.to_string())
