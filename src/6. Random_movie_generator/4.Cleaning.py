import pandas as pd

df = pd.read_csv('C:\\Users\\user\\PycharmProjects\\GMR\\src\\6. Recommendation\\movie_data.csv')


print("Columns in the dataset:")
print(df.columns)

print("Missing values in each column:")
print(df.isnull().sum())

try:
    df_cleaned = df.dropna(subset=['Group', 'Metascore', 'Directors', 'Stars'])

    df_cleaned = df_cleaned.drop(columns=['Index'], errors='ignore')

    df_cleaned['Title'] = df_cleaned['Title'].str.replace(r'^\d+\.\s*', '', regex=True)

    stars_split = df_cleaned['Stars'].str.split(',', expand=True)

    stars_split.columns = [f'Star {i+1}' for i in range(stars_split.shape[1])]

    df_cleaned = pd.concat([df_cleaned, stars_split], axis=1)

    df_cleaned = df_cleaned.drop(columns=['Stars'])

    if 'Star 2' not in df_cleaned.columns:
        df_cleaned['Star 2'] = ''
    if 'Star 3' not in df_cleaned.columns:
        df_cleaned['Star 3'] = ''

    print("\nCleaned data (first 5 rows):")
    print(df_cleaned.head())

    df_cleaned.to_csv('C:/Users/user/PycharmProjects/GMR/test/Cleaned_data.csv', index=False)

    print("\nData has been cleaned and saved successfully.")

except Exception as e:
    print(f"Error occurred: {e}")
