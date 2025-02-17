import pandas as pd

# Load your cleaned data CSV file
df = pd.read_csv('normalized_data.csv')

# Count how many -1's are in the 'Group' column
group_negative_ones = df[df['Group'] == -1].shape[0]
print(f"Number of -1 in 'Group' column: {group_negative_ones}")

# Count how many -1's are in the 'Genre 1' column
genre_negative_ones = df[df['Genre 1'] == -1].shape[0]
print(f"Number of -1 in 'Genre 1' column: {genre_negative_ones}")
