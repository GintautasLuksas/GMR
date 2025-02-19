import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report, roc_curve, auc)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load dataset
data = pd.read_csv('clustered_data.csv')

# Prepare feature matrix
X = data.drop(columns=['KMeans Cluster', 'Agglomerative Cluster', 'Title',
                       'Short Description', 'Directors', 'Star 1', 'Star 2', 'Star 3', 'Genre 2', 'Genre 3',
                       'Metascore'])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X[X.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(
    X.select_dtypes(include=['float64', 'int64']))

# Encode 'Group' column
group_mapping = {
    'G': 0, 'PG': 0, 'PG-13': 0, 'Approved': 0, 'Not Rated': 0, 'U': 0, '12': 0, '12A': 0, 'A': 0, 'AA': 0,
    'R': 1, 'X': 1, '18': 1, 'Rejected': 1, 'Passed': 1, '15': 0
}
X['Group'] = X['Group'].map(group_mapping)

# One-hot encode 'Genre 1'
X = pd.get_dummies(X, columns=['Genre 1'], drop_first=True)

# Standardize and apply PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)


# Train-test split
def train_model(cluster_column, model_name):
    """Trains and evaluates a model for a given cluster column."""
    Y = data[cluster_column]
    X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=42)

    # Handle class imbalance with SMOTE
    min_class_count = min(Y_train.value_counts())
    k_neighbors = max(1, min(3, min_class_count - 1))
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

    # RandomForest Classifier
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_dist = {
        'rf__n_estimators': randint(5, 20),
        'rf__max_depth': [None, 1, 10, 15, 20, 25],
        'rf__min_samples_split': randint(2, 10),
        'rf__min_samples_leaf': randint(1, 10),
        'rf__bootstrap': [True, False],
        'rf__max_features': ['sqrt', 'log2', None],
        'rf__min_impurity_decrease': [0.0, 0.01, 0.1]
    }

    pipeline = Pipeline([('smote', smote), ('rf', rf)])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    try:
        grid_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist,
                                         n_iter=10, cv=cv, verbose=2, random_state=42)
        grid_search.fit(X_train, Y_train)

        # Predictions
        Y_pred = grid_search.best_estimator_.predict(X_test)
        Y_prob = grid_search.best_estimator_.predict_proba(X_test)

        # Evaluation Metrics
        print(f"\nPerformance on {model_name} Test Set:")
        print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.2f}")
        print(f"F1 Score: {f1_score(Y_test, Y_pred, average='weighted'):.2f}")
        print(f"Precision: {precision_score(Y_test, Y_pred, average='weighted'):.2f}")
        print(f"Recall: {recall_score(Y_test, Y_pred, average='weighted'):.2f}")
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")

        # Confusion Matrix
        cm = confusion_matrix(Y_test, Y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(Y_test, Y_prob[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic for {model_name}')
        plt.legend(loc='lower right')
        plt.show()

    except Exception as e:
        print(f"Error during {model_name} training: {e}")


# Train and evaluate both models
train_model('KMeans Cluster', 'KMeans')
train_model('Agglomerative Cluster', 'Agglomerative')
