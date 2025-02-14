import pandas as pd

data = pd.read_csv('normalized_data2.csv')

selected_columns = ['Rating', 'Length (mins)', 'Rating Amount', 'Metascore']
numeric_data = data[selected_columns]

numeric_data.to_csv('numeric2.csv', index=False)

print("Selected columns have been saved to 'numeric.csv'")
