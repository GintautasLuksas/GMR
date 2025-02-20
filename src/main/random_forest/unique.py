import pandas as pd

data = pd.read_csv('nn_with_cluster.csv')

group_unique_values = data['Group'].unique()

print(group_unique_values)
