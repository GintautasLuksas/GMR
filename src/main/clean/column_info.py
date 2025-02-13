import pandas as pd

# Load the dataset
df = pd.read_csv('complete_data.csv')

# Check for missing values in each column
missing_values = df.isna().sum()

# Display the number of missing values for each column
print(missing_values)

# Optionally, you can also display the percentage of missing values for each column
missing_percentage = (df.isna().sum() / len(df)) * 100
print("\nPercentage of missing values per column:")
print(missing_percentage)
