import pandas as pd

# Load your dataset (adjust the file path accordingly)
df = pd.read_csv('IMDB710_Complete.csv')

# Remove rows with missing values in 'Group', 'Metascore', 'Directors', or 'Stars'
df_cleaned = df.dropna(subset=['Group', 'Metascore', 'Directors', 'Stars'])

# Remove the 'Index' column (if it exists)
df_cleaned = df_cleaned.drop(columns=['Index'], errors='ignore')

# Remove numbers and periods from the start of the 'Title' column
df_cleaned['Title'] = df_cleaned['Title'].str.replace(r'^\d+\.\s*', '', regex=True)

# Split the 'Stars' column into multiple columns (if there are commas separating names)
stars_split = df_cleaned['Stars'].str.split(',', expand=True)

# Rename the new columns for better clarity (e.g., 'Star 1', 'Star 2', etc.)
stars_split.columns = [f'Star {i+1}' for i in range(stars_split.shape[1])]

# Concatenate the original dataframe with the new star columns
df_cleaned = pd.concat([df_cleaned, stars_split], axis=1)

# Drop the original 'Stars' column
df_cleaned = df_cleaned.drop(columns=['Stars'])

# Output the cleaned data (this is just for display, you can save it to a new file if needed)
print(df_cleaned.head())  # Display the first few rows of the cleaned dataset

# Optionally, save the cleaned data to a new file
df_cleaned.to_csv('IMDB710_Cleaned.csv', index=False)
