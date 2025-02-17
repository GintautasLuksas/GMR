import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report, roc_curve, auc)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load dataset
data = pd.read_csv('nn_with_cluster.csv')

print("Initial data preview:")
print(data.head())

# Separate numeric and categorical columns
X = data.drop(columns=['KMeans_Cluster', 'Agglomerative_Cluster', 'Title',
                       'Short Description', 'Directors', 'Star 1', 'Star 2', 'Star 3', 'Genre 2', 'Genre 3',
                       'Metascore'])

# Identifying numeric and categorical columns
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Handle missing values for numeric columns using SimpleImputer (filling with the mean)
imputer = SimpleImputer(strategy='mean')
X[numeric_columns] = imputer.fit_transform(X[numeric_columns])

# Handle 'Group' column (categorical) separately by applying the mapping directly
group_mapping = {
    'G': 0, 'PG': 0, 'PG-13': 0, 'Approved': 0, 'Not Rated': 0,
    'U': 0, '12': 0, '12A': 0, 'A': 0, 'AA': 0, 'R': 1, 'X': 1, '18': 1, 'Rejected': 1, 'Passed': 1,
    '15': 0
}

# Check for any unexpected values in the 'Group' column and raise an error if found
unexpected_groups = X['Group'].unique()
valid_groups = set(group_mapping.keys())
invalid_groups = [group for group in unexpected_groups if group not in valid_groups]

if invalid_groups:
    raise ValueError(f"Unexpected values found in 'Group' column: {invalid_groups}")
else:
    # Apply the mapping to the 'Group' column
    X['Group'] = X['Group'].map(group_mapping)

# One-hot encode 'Genre 1' column
X = pd.get_dummies(X, columns=['Genre 1'], drop_first=True)

print("Features shape (after one-hot encoding and group mapping):", X.shape)

# Standardize before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (retain 95% of variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"Features shape (after PCA): {X_pca.shape}")
print(f"Explained variance retained: {sum(pca.explained_variance_ratio_):.2f}")

# Train-test split
Y_kmeans = data['KMeans_Cluster']
X_train, X_test, Y_train_kmeans, Y_test_kmeans = train_test_split(X_pca, Y_kmeans, test_size=0.2, random_state=42)
Y_train_agglomerative, Y_test_agglomerative = Y_train_kmeans.copy(), Y_test_kmeans.copy()

print(f"Training set size (KMeans): {X_train.shape}, Test set size (KMeans): {X_test.shape}")

# Handle class imbalance with SMOTE
min_class_count = min(Y_train_kmeans.value_counts())
k_neighbors = max(1, min(3, min_class_count - 1))
print(f"Using k_neighbors: {k_neighbors}")
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

# RandomForest Classifier
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Hyperparameter tuning grid
param_dist = {
    'rf__n_estimators': randint(5, 20),
    'rf__max_depth': [None, 1, 10, 15, 20, 25],
    'rf__min_samples_split': randint(2, 10),
    'rf__min_samples_leaf': randint(1, 10),
    'rf__bootstrap': [True, False],
    'rf__max_features': ['sqrt', 'log2', None],
    'rf__min_impurity_decrease': [0.0, 0.01, 0.1]
}

# Pipeline: SMOTE + RandomForest
pipeline = Pipeline([('smote', smote), ('rf', rf)])

# Cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def train_and_evaluate(y_train, y_test, model_name):
    """Trains a model with RandomizedSearchCV and evaluates it."""
    try:
        print(f"\nStarting {model_name} Clustering Model Training and Evaluation...")

        # Perform RandomizedSearchCV
        grid_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist,
                                         n_iter=10, cv=cv, verbose=2, random_state=42)
        grid_search.fit(X_train, y_train)

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best score for {model_name}: {grid_search.best_score_:.2f}")

        # Make predictions on the test set
        y_pred = grid_search.best_estimator_.predict(X_test)

        # Calculate probabilities for ROC and AUC
        y_prob = grid_search.best_estimator_.predict_proba(X_test)

        # Evaluate performance
        print(f"\nPerformance on {model_name} Test Set:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
        print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report: \n{classification_report(y_test, y_pred)}")

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc:.2f}")

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic for {model_name}')
        plt.legend(loc="lower right")
        plt.show()

    except Exception as e:
        print(f"Error during {model_name} model training and evaluation: {e}")


# Train and evaluate models
train_and_evaluate(Y_train_kmeans, Y_test_kmeans, 'KMeans')
train_and_evaluate(Y_train_agglomerative, Y_test_agglomerative, 'Agglomerative')
