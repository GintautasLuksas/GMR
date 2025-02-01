import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('IMDB710_Clustered.csv')

# Drop non-numeric columns and store the numeric ones
numeric_columns = ['Year', 'Rating', 'Length (minutes)', 'Rating Amount', 'Group', 'Metascore']
df_numeric = df[numeric_columns]

# Add the cluster columns to the target variables
y_kmeans = df['KMeans_Cluster']
y_agglo = df['Agglomerative_Cluster']

# Split the dataset into training and testing sets
X_train, X_test, y_train_kmeans, y_test_kmeans = train_test_split(df_numeric, y_kmeans, test_size=0.3, random_state=42)
X_train, X_test, y_train_agglo, y_test_agglo = train_test_split(df_numeric, y_agglo, test_size=0.3, random_state=42)

# Scale the data (standardize features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Impute missing values using the mean strategy
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

# Train Random Forest Classifier for KMeans_Cluster
rf_kmeans = RandomForestClassifier(random_state=42)
rf_kmeans.fit(X_train_imputed, y_train_kmeans)

# Predict and evaluate KMeans model
y_pred_kmeans = rf_kmeans.predict(X_test_imputed)
accuracy_kmeans = accuracy_score(y_test_kmeans, y_pred_kmeans)
f1_kmeans = f1_score(y_test_kmeans, y_pred_kmeans, average='weighted')
conf_matrix_kmeans = confusion_matrix(y_test_kmeans, y_pred_kmeans)

# Train Random Forest Classifier for Agglomerative_Cluster
rf_agglo = RandomForestClassifier(random_state=42)
rf_agglo.fit(X_train_imputed, y_train_agglo)

# Predict and evaluate Agglomerative model
y_pred_agglo = rf_agglo.predict(X_test_imputed)
accuracy_agglo = accuracy_score(y_test_agglo, y_pred_agglo)
f1_agglo = f1_score(y_test_agglo, y_pred_agglo, average='weighted')
conf_matrix_agglo = confusion_matrix(y_test_agglo, y_pred_agglo)

# Output the results
print(f"KMeans Model Accuracy: {accuracy_kmeans:.4f}")
print(f"KMeans Model F1 Score: {f1_kmeans:.4f}")
print("KMeans Confusion Matrix:")
print(conf_matrix_kmeans)

print(f"\nAgglomerative Model Accuracy: {accuracy_agglo:.4f}")
print(f"Agglomerative Model F1 Score: {f1_agglo:.4f}")
print("Agglomerative Confusion Matrix:")
print(conf_matrix_agglo)
