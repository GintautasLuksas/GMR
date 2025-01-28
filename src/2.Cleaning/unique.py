import pandas as pd

# Load your dataset (adjust the file path accordingly)
df = pd.read_csv('IMDB710_Complete.csv')

# Get the count of unique values in each column
unique_counts = df.nunique()

# Print the result
print(unique_counts)
