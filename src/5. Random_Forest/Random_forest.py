import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report, roc_curve, auc)
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust based on your system


# Load Data
data = pd.read_csv('NN_cluster_labels_with_embeddings.csv')

# Drop non-numeric columns and handle missing values
X = data.drop(columns=['KMeans_Cluster', 'Agglomerative_Cluster', 'Title', 'Year',
                       'Short Description', 'Directors', 'Star 1', 'Star 2', 'Star 3',
                       'Group', 'Metascore']).fillna(0)  # Fill NaN with 0

Y_kmeans = data['KMeans_Cluster']
Y_agglomerative = data['Agglomerative_Cluster']

# Train-Test Split
X_train, X_test, Y_train_kmeans, Y_test_kmeans = train_test_split(X, Y_kmeans, test_size=0.2, random_state=42)
Y_train_agglomerative, Y_test_agglomerative = Y_train_kmeans.copy(), Y_test_kmeans.copy()

# Handle imbalanced classes with SMOTE
min_class_count = min(Y_train_kmeans.value_counts())
k_neighbors = max(1, min(3, min_class_count - 1))  # Ensure k_neighbors is valid
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

# Random Forest Classifier with Balanced Class Weights
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Hyperparameter Search Space
param_dist = {
    'rf__n_estimators': randint(50, 200),
    'rf__max_depth': [None, 10, 20, 30, 40, 50],
    'rf__min_samples_split': randint(2, 20),
    'rf__min_samples_leaf': randint(1, 20),
    'rf__bootstrap': [True, False],
    'rf__max_features': ['sqrt', 'log2', None],
    'rf__min_impurity_decrease': [0.0, 0.01, 0.1]
}

# Create Pipeline
pipeline = Pipeline([('smote', smote), ('rf', rf)])

# Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def train_and_evaluate(y_train, y_test, model_name):
    """Trains a model with RandomizedSearchCV and evaluates it."""

    # Randomized Search
    grid_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist,
                                     n_iter=10, cv=cv, n_jobs=1, verbose=1, random_state=42)
    grid_search.fit(X_train, y_train)

    # Best Model
    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)

    # Evaluation Metrics
    print(f"\n{model_name} Clustering Results:")
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, Y_pred))
    print("F1 Score:", f1_score(y_test, Y_pred, average='weighted'))
    print("Precision:", precision_score(y_test, Y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, Y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, Y_pred))
    print("Classification Report:\n", classification_report(y_test, Y_pred))

    # ROC Curve (only for binary classification)
    try:
        if len(np.unique(y_test)) == 2:  # Ensure binary classification
            fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            print(f"AUC for {model_name}:", roc_auc)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic - {model_name}')
            plt.legend(loc="lower right")
            plt.show()
    except Exception as e:
        print(f"Error plotting ROC curve for {model_name}: {e}")


# Train and Evaluate for KMeans
train_and_evaluate(Y_train_kmeans, Y_test_kmeans, "KMeans")

# Train and Evaluate for Agglomerative
train_and_evaluate(Y_train_agglomerative, Y_test_agglomerative, "Agglomerative")
 