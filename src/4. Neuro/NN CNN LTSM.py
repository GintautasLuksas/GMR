import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity

# 1. Data Loading and Preprocessing
data = pd.read_csv('IMDB710_Cleaned.csv')

# Process numerical features
numerical_columns = ['Rating', 'Year', 'Length (minutes)', 'Rating Amount', 'Metascore']
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Process categorical features (Group) using LabelEncoder
encoder = LabelEncoder()
data['Group'] = encoder.fit_transform(data['Group'])

# Process textual features (Directors, Stars) using one-hot encoding
# Ensure that the columns exist before applying pd.get_dummies
star_columns = ['Star 1', 'Star 2', 'Star 3']
directors_column = ['Directors']

# Apply one-hot encoding
data = pd.get_dummies(data, columns=directors_column + star_columns, drop_first=True)

# Fill missing values if necessary
data.fillna(0, inplace=True)

# 2. Process Short Description using TF-IDF
# Ensure the Short Description column exists
if 'Short Description' in data.columns:
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    description_features = tfidf_vectorizer.fit_transform(data['Short Description']).toarray()
else:
    description_features = np.zeros((data.shape[0], 1000))  # Fallback in case column is missing

# 3. Combine all features (numerical, categorical, text)
numerical_data = data.drop(columns=['Title', 'Short Description'])  # Drop non-numerical features
X_combined = np.hstack((numerical_data.values, description_features))

# 4. Prepare data for the autoencoder
X_train, X_test = train_test_split(X_combined, test_size=0.2, random_state=42)

# 5. Build the Autoencoder Model

# Define input layer
input_layer = Input(shape=(X_train.shape[1],))

# Encoder layers
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# Latent space (bottleneck)
latent_space = Dense(16, activation='relu')(encoded)

# Decoder layers
decoded = Dense(32, activation='relu')(latent_space)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)

# Output layer
output_layer = Dense(X_train.shape[1], activation='sigmoid')(decoded)

# Build the model
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

# 6. Train the Autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_test, X_test))

# 7. Extract the learned latent space (compressed representation)
encoder_model = Model(inputs=input_layer, outputs=latent_space)
latent_representations = encoder_model.predict(X_combined)

# 8. Movie Recommendation Logic

# Calculate cosine similarity between movies based on their latent space representation
similarity_matrix = cosine_similarity(latent_representations)

# Function to recommend movies based on the input movie
def recommend_movie(movie_index, top_n=5):
    movie_similarities = similarity_matrix[movie_index]
    similar_movie_indices = movie_similarities.argsort()[-top_n-1:-1][::-1]  # Get top N similar movies
    return data['Title'].iloc[similar_movie_indices]

# Test the recommendation system (for the first movie in the dataset)
recommended_movies = recommend_movie(movie_index=0, top_n=5)
print("Recommended Movies:")
print(recommended_movies)
