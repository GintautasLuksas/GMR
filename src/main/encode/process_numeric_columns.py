import pandas as pd

# Load the data from normalized_data.csv
data = pd.read_csv('normalized_data.csv')

# Select only the specified columns
selected_columns = ['Rating', 'Length (mins)', 'Rating Amount', 'Metascore']
numeric_data = data[selected_columns]

# Save the selected data to a new CSV file
numeric_data.to_csv('numeric.csv', index=False)

print("Selected columns have been saved to 'numeric.csv'")
