import pandas as pd

def merge_csv_files():
    imdb_main = pd.read_csv('imdb_movies2.csv')

    imdb_additional = pd.read_csv('additional_data2.csv', names=['Directors', 'Stars', 'Genres'])

    imdb_complete = pd.concat([imdb_main, imdb_additional], axis=1)

    imdb_complete = imdb_complete.rename(columns={'Metascore': 'Group', 'Group': 'Metascore'})

    imdb_complete.to_csv(r'C:\Users\user\PycharmProjects\GMR\src\system_test_data\clean\complete_data2.csv', index=False)
    print("Data successfully merged, and saved as complete_data2.csv")

merge_csv_files()
