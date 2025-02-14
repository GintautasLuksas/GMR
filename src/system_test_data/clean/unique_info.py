import pandas as pd

df = pd.read_csv('complete_data.csv')

print("Data types of 'Group' and 'Metascore' columns:")
print(df[['Group', 'Metascore']].dtypes)

unique_groups = df['Group'].unique()

print("\nUnique values in 'Group' column:")
print(unique_groups)

group_counts = df['Group'].value_counts()

print("\nCount of each unique value in 'Group' column:")
print(group_counts)
