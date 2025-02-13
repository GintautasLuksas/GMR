import pandas as pd

def merge_csv_files():
    # Load the main IMDb data
    imdb_main = pd.read_csv('imdb_movies.csv')

    # Load the additional data with column names
    imdb_additional = pd.read_csv('additional_data.csv', names=['Directors', 'Stars', 'Genres'])

    # Merge the two DataFrames along the columns (axis=1)
    imdb_complete = pd.concat([imdb_main, imdb_additional], axis=1)

    # Swap the 'Group' and 'Metascore' columns
    imdb_complete = imdb_complete.rename(columns={'Metascore': 'Group', 'Group': 'Metascore'})

    # Save the merged DataFrame as a new CSV file
    imdb_complete.to_csv(r'C:\Users\user\PycharmProjects\GMR\src\main\clean\complete_data.csv', index=False)
    print("Data successfully merged, and saved as complete_data.csv")

# Run the function
merge_csv_files()
