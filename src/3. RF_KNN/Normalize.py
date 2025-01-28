import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# File paths
input_path = r"C:\Users\BossJore\PycharmProjects\python_SQL\GMR\src\3. RF_KNN\IMDB710_Cleaned.csv"
output_path = r"C:\Users\BossJore\PycharmProjects\python_SQL\GMR\src\3. RF_KNN\Normalized_Data.csv"

# Load data
data = pd.read_csv(input_path)

# Combine Rating and Rating Amount into a new feature "Weighted Rating"
max_rating_amount = data['Rating Amount'].max()
data['Weighted Rating'] = (data['Rating'] * data['Rating Amount']) / max_rating_amount

# Columns to normalize (including 'Weighted Rating')
columns_to_normalize = ['Year', 'Length (minutes)', 'Rating Amount', 'Metascore', 'Rating', 'Weighted Rating']

# Normalize the selected columns
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(data[columns_to_normalize])

# Create DataFrame for normalized data
normalized_data = pd.DataFrame(normalized_values, columns=columns_to_normalize)

# Save the normalized data to CSV
normalized_data.to_csv(output_path, index=False)

# Output message
print(f"Normalized data (including Weighted Rating and Rating) saved to {output_path}")
