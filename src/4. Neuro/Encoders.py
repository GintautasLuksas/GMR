import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity



def load_and_preprocess_data():
    """
    Loads and preprocesses the movie dataset. It scales numerical features, encodes categorical features,
    and applies one-hot encoding to text features. Missing values are filled with 0.
    """
    data = pd.read_csv('IMDB710_Cleaned.csv')

    numerical_columns = ['Rating', 'Year', 'Length (minutes)', 'Rating Amount', 'Metascore']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    encoder = LabelEncoder()
    data['Group'] = encoder.fit_transform(data['Group'])

    star_columns = ['Star 1', 'Star 2', 'Star 3']
    directors_column = ['Directors']
    data = pd.get_dummies(data, columns=directors_column + star_columns, drop_first=True)

    data.fillna(0, inplace=True)

    if 'Short Description' in data.columns:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        description_features = tfidf_vectorizer.fit_transform(data['Short Description']).toarray()
    else:
        description_features = np.zeros((data.shape[0], 1000))

    numerical_data = data.drop(columns=['Title', 'Short Description'])
    X_combined = np.hstack((numerical_data.values, description_features))

    return data, X_combined


class Autoencoder(nn.Module):
    """
    Defines the Autoencoder neural network model using PyTorch.
    """

    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Sigmoid activation to match original data range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(model, X_train, X_test, epochs=50, batch_size=256):
    """
    Trains the autoencoder model using the given training data.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Convert numpy arrays to PyTorch tensors and ensure they are of the correct dtype
    X_train_tensor = torch.tensor(X_train.astype(np.float32), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.astype(np.float32), dtype=torch.float32)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(X_train_tensor)

        # Compute loss
        loss = criterion(output, X_train_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    return model


def extract_latent_representations(model, X_combined):
    """
    Extracts the learned latent space (compressed representation) from the trained autoencoder.
    """
    model.eval()

    # Ensure X_combined is of the correct dtype (float32)
    X_combined = X_combined.astype(np.float32)

    # Convert to PyTorch tensor
    X_combined_tensor = torch.tensor(X_combined, dtype=torch.float32)

    with torch.no_grad():
        latent_representations = model.encoder(X_combined_tensor)

    return latent_representations.numpy()


def recommend_movie(movie_name, similarity_matrix, data, top_n=5):
    """
    Recommends top N similar movies based on the cosine similarity of their latent space representations.
    """
    movie_index = data[data['Title'] == movie_name].index[0]
    movie_similarities = similarity_matrix[movie_index]
    similar_movie_indices = movie_similarities.argsort()[-top_n - 1:-1][::-1]
    return data['Title'].iloc[similar_movie_indices]


def evaluate_model(X_test, X_pred):
    """
    Evaluate the autoencoder performance using reconstruction error and metrics.
    """
    mse = np.mean((X_test - X_pred) ** 2)
    print(f'Reconstruction MSE: {mse:.4f}')

    # For classification tasks, we can use accuracy, F1, precision, and recall
    # but since this is an autoencoder, these metrics are not applicable here.
    return mse


# Main process
data, X_combined = load_and_preprocess_data()

X_train, X_test = train_test_split(X_combined, test_size=0.2, random_state=42)

# Build the model
autoencoder = Autoencoder(X_train.shape[1])

# Train the autoencoder
autoencoder = train_autoencoder(autoencoder, X_train, X_test)

# Extract the latent representations
latent_representations = extract_latent_representations(autoencoder, X_combined)

# Calculate the similarity matrix
similarity_matrix = cosine_similarity(latent_representations)

# Get recommended movies based on user input
movie_name = input("Enter a movie name: ")
recommended_movies = recommend_movie(movie_name, similarity_matrix, data)

print("Recommended Movies:")
print(recommended_movies)

# Evaluate the model (reconstruction error)
X_test_tensor = torch.tensor(X_test.astype(np.float32), dtype=torch.float32)
with torch.no_grad():
    X_pred = autoencoder(X_test_tensor).numpy()

# Evaluate model performance
evaluate_model(X_test, X_pred)
