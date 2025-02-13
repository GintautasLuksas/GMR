import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report, roc_curve, auc)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load dataset
data = pd.read_csv('nn_cluster2.csv')

print("Initial data preview:")
print(data.head())

# Drop non-numeric and target columns
X = data.drop(columns=['KMeans_Cluster', 'Agglomerative_Cluster', 'Title', 'Year',
                       'Short Description', 'Directors', 'Star 1', 'Star 2', 'Star 3', 'Genre 1', 'Genre 2', 'Genre 3',
                       'Group', 'Metascore']).fillna(0)
Y_kmeans = data['KMeans_Cluster']
Y_agglomerative = data['Agglomerative_Cluster']

print("Features shape (before PCA):", X.shape)

# Standardize before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (retain 95% of variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"Features shape (after PCA): {X_pca.shape}")
print(f"Explained variance retained: {sum(pca.explained_variance_ratio_):.2f}")

# Train-test split
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
                                         n_iter=10, cv=cv, n_jobs=1, verbose=1, random_state=42)

        print("Starting RandomizedSearchCV...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        Y_pred = best_model.predict(X_test)

        print(f"Best Hyperparameters for {model_name}: {grid_search.best_params_}")

        # Model evaluation metrics
        print(f"\n{model_name} Clustering Results:")
        print("Accuracy:", accuracy_score(y_test, Y_pred))
        print("F1 Score:", f1_score(y_test, Y_pred, average='weighted'))
        print("Precision:", precision_score(y_test, Y_pred, average='weighted'))
        print("Recall:", recall_score(y_test, Y_pred, average='weighted'))
        print("Confusion Matrix:\n", confusion_matrix(y_test, Y_pred))
        print("Classification Report:\n", classification_report(y_test, Y_pred))

        # Feature importance from the RandomForest model
        if hasattr(best_model.named_steps['rf'], 'feature_importances_'):
            feature_importances = best_model.named_steps['rf'].feature_importances_

            # Get sorted indices (top 10 most important PCA components)
            sorted_indices = np.argsort(feature_importances)[::-1][:10]

            print("\nTop 10 Feature Importances (PCA Components):")
            for idx in sorted_indices:
                print(f"PCA Component {idx}: {feature_importances[idx]:.4f}")

            # Plot feature importances
            plt.figure(figsize=(10, 6))
            plt.barh([f"PCA {idx}" for idx in sorted_indices], feature_importances[sorted_indices])
            plt.xlabel("Feature Importance")
            plt.ylabel("PCA Components")
            plt.title(f"Top 10 Feature Importances - {model_name}")
            plt.gca().invert_yaxis()  # Reverse order for better visualization
            plt.show()

        # ROC Curve and AUC (for multi-class classification using one-vs-rest)
        try:
            # Binarize the output labels for multi-class ROC curve computation
            y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

            # Calculate the ROC curve for each class (one-vs-rest approach)
            fpr, tpr, roc_auc = dict(), dict(), dict()

            for i in range(y_test_binarized.shape[1]):
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], best_model.predict_proba(X_test)[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot ROC curve for each class
            plt.figure()
            colors = ['blue', 'green', 'red']  # Adjust for the number of classes
            for i in range(y_test_binarized.shape[1]):
                plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"Receiver Operating Characteristic (AUC for Multi-Class)")
            plt.legend(loc="lower right")
            plt.show()

            # Print average AUC for all classes (macro-average AUC)
            avg_auc = np.mean(list(roc_auc.values()))
            print(f"Macro-average AUC: {avg_auc:.2f}")

        except Exception as e:
            print(f"Error plotting ROC curve for {model_name}: {e}")

    except Exception as e:
        print(f"Error during model training or evaluation for {model_name}: {e}")


# Train and evaluate models
train_and_evaluate(Y_train_kmeans, Y_test_kmeans, "KMeans")
train_and_evaluate(Y_train_agglomerative, Y_test_agglomerative, "Agglomerative")
