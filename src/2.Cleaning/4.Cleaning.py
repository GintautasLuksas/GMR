import pandas as pd


df = pd.read_csv('src/1.Scrape_IMDB_7-10/IMDB710.csv')

df_cleaned = df.dropna(subset=['Group', 'Metascore', 'Directors', 'Stars'])

df_cleaned = df_cleaned.drop(columns=['Index'], errors='ignore')

df_cleaned['Title'] = df_cleaned['Title'].str.replace(r'^\d+\.\s*', '', regex=True)

stars_split = df_cleaned['Stars'].str.split(',', expand=True)

stars_split.columns = [f'Star {i+1}' for i in range(stars_split.shape[1])]

df_cleaned = pd.concat([df_cleaned, stars_split], axis=1)

df_cleaned = df_cleaned.drop(columns=['Stars'])

print(df_cleaned.head())

df_cleaned.to_csv('src/3. Normalize_Comparison/IMDB710_Cleaned.csv', index=False)
