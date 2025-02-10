import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Corrected input and output paths
input_path = 'C:\\Users\\user\\PycharmProjects\\GMR\\src\\6. Recommendation\\Cleaned_data.csv'
output_path = 'C:\\Users\\user\\PycharmProjects\\GMR\\src\\6. Recommendation\\Normalized_data.csv'

# Read the input data
data = pd.read_csv(input_path)

# Calculate the weighted rating
max_rating_amount = data['Rating Amount'].max()
data['Weighted Rating'] = (data['Rating'] * data['Rating Amount']) / max_rating_amount

# List of columns to normalize
columns_to_normalize = ['Year', 'Length (minutes)', 'Rating Amount', 'Metascore', 'Rating', 'Weighted Rating']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the selected columns
normalized_values = scaler.fit_transform(data[columns_to_normalize])

# Create a DataFrame for the normalized values
normalized_data = pd.DataFrame(normalized_values, columns=columns_to_normalize)

# Save the normalized data to the output file
normalized_data.to_csv(output_path, index=False)

# Output message
print(f"Normalized data (including Weighted Rating and Rating) saved to {output_path}")
