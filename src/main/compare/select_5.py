import pandas as pd

# Load the CSV file
df = pd.read_csv('nn_cluster.csv')

# Select 5 random rows
random_rows = df.sample(n=5, random_state=42)  # random_state for reproducibility

# Save the random rows to a new CSV
random_rows.to_csv('random_5.csv', index=False)
