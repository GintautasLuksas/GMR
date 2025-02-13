import pandas as pd

data = pd.read_csv('normalized_data.csv')

selected_columns = ['Rating', 'Length (mins)', 'Rating Amount', 'Metascore']
numeric_data = data[selected_columns]

numeric_data.to_csv('numeric.csv', index=False)

print("Selected columns have been saved to 'numeric.csv'")
