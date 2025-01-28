import pandas as pd

# File path to your dataset
data_path = r"C:\Users\BossJore\PycharmProjects\python_SQL\GMR\src\3. RF_KNN\IMDB710_Cleaned.csv"

# Load the data
df = pd.read_csv(data_path)

# Check minimum and maximum values of the 'Rating' column
min_rating = df['Rating'].min()
max_rating = df['Rating'].max()

print(f"Minimum Rating: {min_rating}")
print(f"Maximum Rating: {max_rating}")
