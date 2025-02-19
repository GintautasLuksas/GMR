import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Input, Dense
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import warnings
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('stopwords')

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings('ignore', category=UserWarning, message=".*Could not find the number of physical cores*")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """Processes text by converting to lowercase, removing punctuation and digits,
    tokenizing, removing stopwords, and lemmatizing."""
    if isinstance(text, str):
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        text = ''.join([char for char in text if not char.isdigit()])
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)
    return ""


data = pd.read_csv('cleaned_data2.csv')
numerical_data = pd.read_csv('numeric2.csv')

data['Cleaned_Title'] = data['Title'].apply(preprocess_text)
data['Cleaned_Short_Description'] = data['Short Description'].apply(preprocess_text)
data['Cleaned_Directors'] = data['Directors'].apply(preprocess_text)
data['Cleaned_Stars'] = data[['Star 1', 'Star 2', 'Star 3']].fillna('').agg(' '.join, axis=1).apply(preprocess_text)
data['Cleaned_Group'] = data['Group'].astype(str).apply(preprocess_text)
data['Cleaned_Genre'] = data[['Genre 1', 'Genre 2', 'Genre 3']].fillna('').agg(' '.join, axis=1).apply(preprocess_text)

label_encoder = LabelEncoder()
data['Encoded_Group'] = label_encoder.fit_transform(data['Cleaned_Group'])

model = SentenceTransformer('all-MiniLM-L6-v2')

title_embeddings = model.encode(data['Cleaned_Title'].tolist())
description_embeddings = model.encode(data['Cleaned_Short_Description'].tolist())
directors_embeddings = model.encode(data['Cleaned_Directors'].tolist())
stars_embeddings = model.encode(data['Cleaned_Stars'].tolist())
genre_embeddings = model.encode(data['Cleaned_Genre'].tolist())
group_embeddings = model.encode(data['Cleaned_Group'].tolist())

final_features = np.concatenate(
    (numerical_data.values, title_embeddings, description_embeddings, directors_embeddings, stars_embeddings,
     genre_embeddings, group_embeddings, data['Encoded_Group'].values.reshape(-1, 1)), axis=1
)

final_features = pd.DataFrame(final_features).apply(pd.to_numeric, errors='coerce').fillna(0).values

input_layer = Input(shape=(final_features.shape[1],))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(final_features.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(final_features, final_features, epochs=100, batch_size=256, shuffle=True)

encoder = Model(input_layer, encoded)
encoded_features = encoder.predict(final_features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_clusters = kmeans.fit_predict(encoded_features)

agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_clustering.fit_predict(encoded_features)

data['KMeans_Cluster'] = kmeans_clusters
data['Agglomerative_Cluster'] = agg_clusters

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(encoded_features)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['KMeans_Cluster'], cmap='viridis')
plt.title('Movie Clusters (Using K-Means)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')

plt.subplot(1, 2, 2)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['Agglomerative_Cluster'], cmap='viridis')
plt.title('Movie Clusters (Using Agglomerative Clustering)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.show()

sil_score_kmeans = silhouette_score(encoded_features, kmeans_clusters)
print(f"Silhouette Score (K-Means): {sil_score_kmeans:.4f}")

sil_score_agg = silhouette_score(encoded_features, agg_clusters)
print(f"Silhouette Score (Agglomerative Clustering): {sil_score_agg:.4f}")

title_embeddings_df = pd.DataFrame(title_embeddings)
description_embeddings_df = pd.DataFrame(description_embeddings)
directors_embeddings_df = pd.DataFrame(directors_embeddings)
stars_embeddings_df = pd.DataFrame(stars_embeddings)
genre_embeddings_df = pd.DataFrame(genre_embeddings)
group_embeddings_df = pd.DataFrame(group_embeddings)

nn_cluster_labels = data[['Title', 'Year', 'Rating', 'Length (mins)', 'Rating Amount',
                          'Group', 'Metascore', 'Short Description', 'Directors',
                          'Star 1', 'Star 2', 'Star 3', 'Genre 1', 'Genre 2', 'Genre 3',
                          'KMeans_Cluster', 'Agglomerative_Cluster']]

nn_cluster_labels = pd.concat([nn_cluster_labels, title_embeddings_df, description_embeddings_df,
                               directors_embeddings_df, stars_embeddings_df, genre_embeddings_df, group_embeddings_df], axis=1)


nn_cluster_labels.to_csv(r'C:\Users\user\PycharmProjects\GMR\src\main\recommendation_nn\nn_cluster2.csv', index=False)

nn_with_cluster = data[['Title', 'Year', 'Rating', 'Length (mins)', 'Rating Amount',
                        'Group', 'Metascore', 'Short Description', 'Directors',
                        'Star 1', 'Star 2', 'Star 3', 'Genre 1', 'Genre 2', 'Genre 3',
                        'KMeans_Cluster', 'Agglomerative_Cluster']]

nn_with_cluster.to_csv(r'C:\Users\user\PycharmProjects\GMR\src\system_test_data\random_forest\nn_with_cluster2.csv', index=False)
