import pandas as pd

# Use the absolute path to the file
df = pd.read_csv('C:\\Users\\user\\PycharmProjects\\GMR\\src\\6. Recommendation\\movie_data.csv')


# Print the columns and check for any missing values
print("Columns in the dataset:")
print(df.columns)

print("Missing values in each column:")
print(df.isnull().sum())

try:
    # Drop rows with missing values in specific columns
    df_cleaned = df.dropna(subset=['Group', 'Metascore', 'Directors', 'Stars'])

    # Drop 'Index' column if it exists (ignores errors if the column is not present)
    df_cleaned = df_cleaned.drop(columns=['Index'], errors='ignore')

    # Remove leading numbers and periods from 'Title' column
    df_cleaned['Title'] = df_cleaned['Title'].str.replace(r'^\d+\.\s*', '', regex=True)

    # Split the 'Stars' column into separate columns
    stars_split = df_cleaned['Stars'].str.split(',', expand=True)

    # Rename the columns for the new 'Star' columns
    stars_split.columns = [f'Star {i+1}' for i in range(stars_split.shape[1])]

    # Concatenate the new 'Star' columns with the cleaned dataframe
    df_cleaned = pd.concat([df_cleaned, stars_split], axis=1)

    # Drop the original 'Stars' column
    df_cleaned = df_cleaned.drop(columns=['Stars'])

    # Add empty 'Star 2' and 'Star 3' columns if they don't exist in the split data
    if 'Star 2' not in df_cleaned.columns:
        df_cleaned['Star 2'] = ''
    if 'Star 3' not in df_cleaned.columns:
        df_cleaned['Star 3'] = ''

    # Print the first few rows of the cleaned DataFrame
    print("\nCleaned data (first 5 rows):")
    print(df_cleaned.head())

    # Save the cleaned data to a CSV file
    df_cleaned.to_csv('C:/Users/user/PycharmProjects/GMR/test/Cleaned_data.csv', index=False)

    print("\nData has been cleaned and saved successfully.")

except Exception as e:
    print(f"Error occurred: {e}")
