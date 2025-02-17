import pandas as pd

def merge_csv_files():
    """Merge two CSV files: 'imdb_movies.csv' and 'additional_data.csv'.
    The additional data contains 'Directors', 'Stars', and 'Genres'.
    The 'Group' and 'Metascore' columns are swapped before saving the merged data.
    """
    imdb_main = pd.read_csv('imdb_movies.csv')
    imdb_additional = pd.read_csv('additional_data.csv', names=['Directors', 'Stars', 'Genres'])
    imdb_complete = pd.concat([imdb_main, imdb_additional], axis=1)
    imdb_complete = imdb_complete.rename(columns={'Metascore': 'Group', 'Group': 'Metascore'})
    imdb_complete.to_csv(r'C:\Users\user\PycharmProjects\GMR\src\main\clean\complete_data.csv', index=False)
    print("Data successfully merged, and saved as complete_data.csv")

merge_csv_files()
