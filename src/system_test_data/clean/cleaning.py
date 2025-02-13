import pandas as pd

# Load the dataset
df = pd.read_csv('complete_data2.csv')

# Drop rows with missing values in important columns
df_cleaned = df.dropna(subset=['Group', 'Metascore', 'Directors', 'Stars', 'Genres'])

# Drop the 'Index' column if it exists
df_cleaned = df_cleaned.drop(columns=['Index'], errors='ignore')

# Clean the 'Title' column by removing leading numbers and periods
df_cleaned['Title'] = df_cleaned['Title'].str.replace(r'^\d+\.\s*', '', regex=True)

# Split the 'Stars' column into multiple columns
stars_split = df_cleaned['Stars'].str.split(',', expand=True)
stars_split.columns = [f'Star {i+1}' for i in range(stars_split.shape[1])]

# Split the 'Genres' column into multiple columns
genres_split = df_cleaned['Genres'].str.split(',', expand=True)
genres_split.columns = [f'Genre {i+1}' for i in range(genres_split.shape[1])]

# Concatenate the new columns with the cleaned dataframe
df_cleaned = pd.concat([df_cleaned, stars_split, genres_split], axis=1)

# Drop the original 'Stars' and 'Genres' columns
df_cleaned = df_cleaned.drop(columns=['Stars', 'Genres'])

# Display the first few rows
print(df_cleaned.head())

# Define the output paths
output_path_1 = r'/src/system_test_data\normalize_comparison\cleaned_data2.csv'
output_path_2 = r'/src/system_test_data\encode\cleaned_data2.csv'

# Save the cleaned data to two different CSV files
df_cleaned.to_csv(output_path_1, index=False)
df_cleaned.to_csv(output_path_2, index=False)

print(f"Cleaned data saved to {output_path_1} and {output_path_2}")
