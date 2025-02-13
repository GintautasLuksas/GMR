import pandas as pd

# Load the dataset
df = pd.read_csv('complete_data.csv')

# Check the data types of 'Group' and 'Metascore' columns
print("Data types of 'Group' and 'Metascore' columns:")
print(df[['Group', 'Metascore']].dtypes)

# Check the unique values in the 'Group' column
unique_groups = df['Group'].unique()

# Display the unique values of the 'Group' column
print("\nUnique values in 'Group' column:")
print(unique_groups)

# Count how many entries each unique value in 'Group' has
group_counts = df['Group'].value_counts()

# Display the count of each unique value in 'Group' column
print("\nCount of each unique value in 'Group' column:")
print(group_counts)
