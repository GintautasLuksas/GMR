import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# File paths
input_path = r"C:\Users\BossJore\PycharmProjects\python_SQL\GMR\src\3. RF_KNN\IMDB710_Cleaned.csv"
output_path = r"C:\Users\BossJore\PycharmProjects\python_SQL\GMR\src\3. RF_KNN\Normalized_Data.csv"

# Load the data
df = pd.read_csv(input_path)

# Normalize selected columns
features = ['Year', 'Length (minutes)', 'Rating Amount', 'Metascore']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Convert ratings into categories
bins = [7.0, 7.8, 8.5, 9.5]
labels = ['Low', 'Medium', 'High']
df['Rating_Class'] = pd.cut(df['Rating'], bins=bins, labels=labels, include_lowest=True)

# Create new feature 'Weighted Rating'
max_rating_amount = df['Rating Amount'].max()
df['Weighted Rating'] = (df['Rating'] * df['Rating Amount']) / max_rating_amount

# Update features to include 'Weighted Rating'
features.append('Weighted Rating')

# Features and target
X = df[features]
y = df['Rating_Class']

# Train-test split (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the training set further into training and validation sets (80% training, 20% validation)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Balance the training set using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Models
models = {
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Hyperparameter tuning with GridSearchCV for each model
param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'bootstrap': [True]
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    "Naive Bayes": {
        'var_smoothing': [1e-9, 1e-8]
    }
}

# Function to perform GridSearchCV
def perform_grid_search(model_name, model, param_grid):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, verbose=2, n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)  # Use resampled data
    return grid_search.best_estimator_, grid_search.best_params_

# Train and evaluate models
results = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")

    # For Naive Bayes, no grid search for hyperparameters
    if model_name == "Naive Bayes":
        best_model = model
        best_model.fit(X_train_resampled, y_train_resampled)
        best_params = {}
    else:
        best_model, best_params = perform_grid_search(model_name, model, param_grids[model_name])

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    results[model_name] = {
        "Best Model": best_model,
        "Best Params": best_params,
        "Accuracy": accuracy,
        "Classification Report": report
    }

# Display results
for model_name, result in results.items():
    print(f"\n{model_name} Results:")
    print(f"Best Params: {result['Best Params']}")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"Classification Report:\n{result['Classification Report']}")

# Visualize class distribution for the entire dataset
plt.figure(figsize=(6, 4))
df['Rating_Class'].value_counts().plot(kind='bar', color=['green', 'orange', 'red'])
plt.title('Distribution of Rating Class (Original Data)')
plt.xlabel('Rating Class')
plt.ylabel('Frequency')
plt.xticks([0, 1, 2], ['Low', 'Medium', 'High'], rotation=0)
plt.show()

# Visualize class distribution for training, validation, and test sets
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Training set
axes[0].bar(pd.Series(y_train).value_counts().index, pd.Series(y_train).value_counts().values)
axes[0].set_title('Training Set')
axes[0].set_xlabel('Rating Class')
axes[0].set_ylabel('Frequency')
axes[0].set_xticklabels(['Low', 'Medium', 'High'])

# Validation set
axes[1].bar(pd.Series(y_valid).value_counts().index, pd.Series(y_valid).value_counts().values)
axes[1].set_title('Validation Set')
axes[1].set_xlabel('Rating Class')
axes[1].set_ylabel('Frequency')
axes[1].set_xticklabels(['Low', 'Medium', 'High'])

# Test set
axes[2].bar(pd.Series(y_test).value_counts().index, pd.Series(y_test).value_counts().values)
axes[2].set_title('Test Set')
axes[2].set_xlabel('Rating Class')
axes[2].set_ylabel('Frequency')
axes[2].set_xticklabels(['Low', 'Medium', 'High'])

plt.tight_layout()
plt.show()

# Plot model accuracies
model_names = list(results.keys())
accuracies = [result['Accuracy'] for result in results.values()]

plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracies, color=['skyblue', 'orange', 'green'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
