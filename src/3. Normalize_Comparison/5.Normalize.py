import pandas as pd
from sklearn.preprocessing import MinMaxScaler

input_path = r"C:\Users\user\PycharmProjects\GMR\src\3. Normalize_Comparison\IMDB710_Cleaned.csv"
output_path = r"C:\Users\user\PycharmProjects\GMR\src\3. Normalize_Comparison\Normalized_Data.csv"

data = pd.read_csv(input_path)

max_rating_amount = data['Rating Amount'].max()
data['Weighted Rating'] = (data['Rating'] * data['Rating Amount']) / max_rating_amount

columns_to_normalize = ['Year', 'Length (minutes)', 'Rating Amount', 'Metascore', 'Rating', 'Weighted Rating']

scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(data[columns_to_normalize])

normalized_data = pd.DataFrame(normalized_values, columns=columns_to_normalize)

normalized_data.to_csv(output_path, index=False)

print(f"Normalized data (including Weighted Rating and Rating) saved to {output_path}")
