import pandas as pd

df = pd.read_csv('complete_data.csv')

missing_values = df.isna().sum()

print(missing_values)

missing_percentage = (df.isna().sum() / len(df)) * 100
print("\nPercentage of missing values per column:")
print(missing_percentage)
