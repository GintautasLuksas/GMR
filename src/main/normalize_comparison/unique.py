import pandas as pd

# Load the cleaned dataset
df = pd.read_csv(r'/src/normalize_comparison\cleaned_data.csv')

# Display the number of rows in each column
print(df.count())

# Display column names to confirm the structure
print("\nColumns in the dataset:")
print(df.columns)
