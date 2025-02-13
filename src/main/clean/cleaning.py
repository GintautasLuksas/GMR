import pandas as pd


def clean_data():
    """Loads a dataset, performs cleaning operations, and saves the cleaned data to two different CSV files.
    The cleaning includes handling missing values, removing unwanted columns, and processing string columns."""

    df = pd.read_csv('complete_data.csv')

    df_cleaned = df.dropna(subset=['Group', 'Metascore', 'Directors', 'Stars', 'Genres'])
    df_cleaned = df_cleaned.drop(columns=['Index'], errors='ignore')

    df_cleaned['Title'] = df_cleaned['Title'].str.replace(r'^\d+\.\s*', '', regex=True)

    stars_split = df_cleaned['Stars'].str.split(',', expand=True)
    stars_split.columns = [f'Star {i + 1}' for i in range(stars_split.shape[1])]

    genres_split = df_cleaned['Genres'].str.split(',', expand=True)
    genres_split.columns = [f'Genre {i + 1}' for i in range(genres_split.shape[1])]

    df_cleaned = pd.concat([df_cleaned, stars_split, genres_split], axis=1)
    df_cleaned = df_cleaned.drop(columns=['Stars', 'Genres'])

    print(df_cleaned.head())

    output_path_1 = r'C:\Users\user\PycharmProjects\GMR\src\main\normalize_comparison\cleaned_data.csv'
    output_path_2 = r'C:\Users\user\PycharmProjects\GMR\src\main\encode\cleaned_data.csv'

    df_cleaned.to_csv(output_path_1, index=False)
    df_cleaned.to_csv(output_path_2, index=False)

    print(f"Cleaned data saved to {output_path_1} and {output_path_2}")
