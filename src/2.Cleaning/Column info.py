import pandas as pd


df = pd.read_csv('../1.Scrape_IMDB_7-10/IMDB710.csv')


filled_count_per_column = df.count()


print(filled_count_per_column)

df = pd.read_csv('../1.Scrape_IMDB_7-10/IMDB710_Additional.csv')


filled_count_per_column = df.count()


print(filled_count_per_column)

df = pd.read_csv('IMDB710_Complete.csv')

filled_count_per_column = df.count()


print(filled_count_per_column)

df = pd.read_csv('IMDB710_Cleaned.csv')

filled_count_per_column = df.count()

print(filled_count_per_column)



