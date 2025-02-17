import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_data(input_path, output_paths):
    """
    Normalizes selected columns in the dataframe using MinMaxScaler, excluding 'Group'.
    """
    data = pd.read_csv(input_path)

    columns_to_normalize = ['Year', 'Length (mins)', 'Rating Amount', 'Metascore', 'Rating']

    for col in columns_to_normalize:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna(subset=columns_to_normalize)

    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(data[columns_to_normalize])

    normalized_data = pd.DataFrame(normalized_values, columns=columns_to_normalize)

    data.loc[:, columns_to_normalize] = normalized_data

    for output_path in output_paths:
        data.to_csv(output_path, index=False)
        print(f"Normalized data saved to {output_path}")


input_path = r"C:\Users\user\PycharmProjects\GMR\src\system_test_data\normalize_comparison\cleaned_data2.csv"
output_paths = [
    r"C:\Users\user\PycharmProjects\GMR\src\system_test_data\normalize_comparison\normalized_data2.csv",
    r"C:\Users\user\PycharmProjects\GMR\src\system_test_data\encode\normalized_data2.csv"
]
normalize_data(input_path, output_paths)
