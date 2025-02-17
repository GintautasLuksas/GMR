import pandas as pd

# Load dataset
data = pd.read_csv('nn_with_cluster.csv')

# Display unique values in the 'Group' column
group_unique_values = data['Group'].unique()

print(group_unique_values)
