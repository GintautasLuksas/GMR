import pandas as pd

df = pd.read_csv(r'/src/normalize_comparison\cleaned_data.csv')

print(df.count())

print("\nColumns in the dataset:")
print(df.columns)
