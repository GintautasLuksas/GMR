import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random

def load_and_clean_data(input_path):
    """
    Loads the dataset and performs initial cleaning steps:
    - Fills missing 'Metascore' with the mean value.
    - Fills missing 'Group' with a random value from the predefined list.
    - Drops rows with missing 'Directors', 'Star 1', or 'Genre 1'.
    - Drops the 'Index' column if it exists.
    - Cleans the 'Title' column by removing leading numbers and periods.
    """
    data = pd.read_csv(input_path)
    data['Metascore'] = data['Metascore'].fillna(data['Metascore'].mean())
    group_options = ['15', 'PG', '12A', '18', 'U', 'X', '12']
    data['Group'] = data['Group'].fillna(random.choice(group_options))
    data_cleaned = data.dropna(subset=['Directors', 'Star 1', 'Genre 1'])
    data_cleaned = data_cleaned.drop(columns=['Index'], errors='ignore')
    data_cleaned['Title'] = data_cleaned['Title'].str.replace(r'^\d+\.\s*', '', regex=True)
    return data_cleaned

def split_star_and_genre_columns(data_cleaned):
    """
    Splits the 'Stars' and 'Genres' columns into separate columns.
    Ensures no missing values and strips leading/trailing spaces.
    """
    data_cleaned[['Star 1', 'Star 2', 'Star 3']] = data_cleaned[['Star 1', 'Star 2', 'Star 3']].fillna('').apply(
        lambda x: x.str.split(',')).apply(lambda x: pd.Series(x[0] if isinstance(x, list) else x))
    data_cleaned[['Star 1', 'Star 2', 'Star 3']] = data_cleaned[['Star 1', 'Star 2', 'Star 3']].apply(
        lambda x: x.str.strip())
    data_cleaned[['Genre 1', 'Genre 2', 'Genre 3']] = data_cleaned[['Genre 1', 'Genre 2', 'Genre 3']].fillna('').apply(
        lambda x: x.str.split(',')).apply(lambda x: pd.Series(x[0] if isinstance(x, list) else x))
    data_cleaned[['Genre 1', 'Genre 2', 'Genre 3']] = data_cleaned[['Genre 1', 'Genre 2', 'Genre 3']].apply(
        lambda x: x.str.strip())
    return data_cleaned

def normalize_data(data_cleaned):
    """
    Normalizes selected columns in the dataframe using MinMaxScaler, excluding 'Group'.
    """
    columns_to_normalize = ['Year', 'Length (mins)', 'Rating Amount', 'Metascore', 'Rating']
    for col in columns_to_normalize:
        data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')
    data_cleaned = data_cleaned.dropna(subset=columns_to_normalize)
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(data_cleaned[columns_to_normalize])
    normalized_data = pd.DataFrame(normalized_values, columns=columns_to_normalize)
    data_cleaned.loc[:, columns_to_normalize] = normalized_data
    return data_cleaned

def save_data(data_cleaned, output_paths):
    """
    Saves the cleaned and normalized dataframe to multiple CSV files.
    """
    for output_path in output_paths:
        data_cleaned.to_csv(output_path, index=False)
        print(f"Normalized data saved to {output_path}")

def process_data(input_path, output_paths):
    """
    Processes the movie dataset by cleaning, splitting columns, and normalizing the data.
    """
    data_cleaned = load_and_clean_data(input_path)
    data_cleaned = split_star_and_genre_columns(data_cleaned)
    data_cleaned = normalize_data(data_cleaned)
    save_data(data_cleaned, output_paths)

input_path = r"C:\Users\user\PycharmProjects\GMR\src\main\normalize_comparison\cleaned_data.csv"
output_paths = [
    r"C:\Users\user\PycharmProjects\GMR\src\main\normalize_comparison\normalized_data.csv",
    r"C:\Users\user\PycharmProjects\GMR\src\main\encode\normalized_data.csv"
]
process_data(input_path, output_paths)
