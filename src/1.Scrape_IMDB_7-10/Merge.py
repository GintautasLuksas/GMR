import pandas as pd

def merge_csv_files():
    # Load the primary data (IMDB710.csv)
    imdb_main = pd.read_csv('../2.Cleaning/IMDB710.csv')

    # Load the additional data (IMDB710_Additional.csv)
    imdb_additional = pd.read_csv('../2.Cleaning/IMDB710_Additional.csv', names=['Directors', 'Stars'])

    # Merge the two dataframes based on the row index
    imdb_complete = pd.concat([imdb_main, imdb_additional], axis=1)

    # Save the merged data to a new CSV file
    imdb_complete.to_csv('IMDB710_Complete.csv', index=False)
    print("Data successfully merged and saved as IMDB710_Complete.csv")

if __name__ == "__main__":
    merge_csv_files()
