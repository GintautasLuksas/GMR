import pandas as pd

df = pd.read_csv('normalized_data.csv')

group_negative_ones = df[df['Group'] == -1].shape[0]
print(f"Number of -1 in 'Group' column: {group_negative_ones}")

genre_negative_ones = df[df['Genre 1'] == -1].shape[0]
print(f"Number of -1 in 'Genre 1' column: {genre_negative_ones}")
