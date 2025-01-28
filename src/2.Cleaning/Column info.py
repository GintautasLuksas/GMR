import pandas as pd


df = pd.read_csv('IMDB710.csv')


filled_count_per_column = df.count()


print(filled_count_per_column)

df = pd.read_csv('IMDB710_Additional.csv')


filled_count_per_column = df.count()

# Print the result
print(filled_count_per_column)

df = pd.read_csv('IMDB710_Complete.csv')

# Get the count of non-NaN values in each column (i.e., how many rows are filled with something)
filled_count_per_column = df.count()

# Print the result
print(filled_count_per_column)

df = pd.read_csv('IMDB710_Cleaned.csv')

# Get the count of non-NaN values in each column (i.e., how many rows are filled with something)
filled_count_per_column = df.count()

# Print the result
print(filled_count_per_column)



