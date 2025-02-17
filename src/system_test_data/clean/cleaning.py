import pandas as pd


def load_and_clean_data(file_path):
    """Loads the dataset, removes rows with missing values, cleans the 'Title' column,
    and splits 'Stars' and 'Genres' into separate columns."""
    df = pd.read_csv(file_path)
    df_cleaned = df.dropna(subset=['Group', 'Metascore', 'Directors', 'Stars', 'Genres'])
    df_cleaned = df_cleaned.drop(columns=['Index'], errors='ignore')
    df_cleaned['Title'] = df_cleaned['Title'].str.replace(r'^\d+\.\s*', '', regex=True)

    stars_split = df_cleaned['Stars'].str.split(',', expand=True)
    stars_split.columns = [f'Star {i+1}' for i in range(stars_split.shape[1])]

    genres_split = df_cleaned['Genres'].str.split(',', expand=True)
    genres_split.columns = [f'Genre {i+1}' for i in range(genres_split.shape[1])]

    df_cleaned = pd.concat([df_cleaned, stars_split, genres_split], axis=1)
    df_cleaned = df_cleaned.drop(columns=['Stars', 'Genres'])

    return df_cleaned


def save_cleaned_data(df, output_path_1, output_path_2):
    """Saves the cleaned dataframe to two different CSV files."""
    df.to_csv(output_path_1, index=False)
    df.to_csv(output_path_2, index=False)
    print(f"Cleaned data saved to {output_path_1} and {output_path_2}")


file_path = 'complete_data2.csv'
output_path_1 = r'C:\Users\user\PycharmProjects\GMR\src\system_test_data\normalize_comparison\cleaned_data2.csv'
output_path_2 = r'C:\Users\user\PycharmProjects\GMR\src\system_test_data\encode\cleaned_data2.csv'

df_cleaned = load_and_clean_data(file_path)
print(df_cleaned.head())
save_cleaned_data(df_cleaned, output_path_1, output_path_2)
