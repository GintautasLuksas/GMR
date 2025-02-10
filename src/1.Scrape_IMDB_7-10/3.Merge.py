import pandas as pd

def merge_csv_files():
    imdb_main = pd.read_csv('IMDB710.csv')

    imdb_additional = pd.read_csv('IMDB710_Additional.csv', names=['Directors', 'Stars'])

    imdb_complete = pd.concat([imdb_main, imdb_additional], axis=1)

    imdb_complete.to_csv(r'C:\Users\user\PycharmProjects\GMR\src\2.Cleaning\IMDB710_Complete.csv', index=False)
    print("Data successfully merged and saved as IMDB710_Complete.csv")

merge_csv_files()
