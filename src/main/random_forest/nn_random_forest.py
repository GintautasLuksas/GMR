import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

data = pd.read_csv('nn_with_cluster.csv')

X = data.drop(columns=['KMeans_Cluster', 'Agglomerative_Cluster', 'Title', 'Short Description', 'Directors', 'Star 1', 'Star 2',
                       'Star 3', 'Genre 2', 'Genre 3', 'Metascore'])

numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
X[numeric_columns] = imputer.fit_transform(X[numeric_columns])

group_mapping = {'G': 0, 'PG': 0, 'PG-13': 0, 'Approved': 0, 'Not Rated': 0, 'U': 0, '12': 0, '12A': 0, 'A': 0, 'AA': 0,
                 'R': 1, 'X': 1, '18': 1, 'Rejected': 1, 'Passed': 1, '15': 0}
X['Group'] = X['Group'].map(group_mapping)

X = pd.get_dummies(X, columns=['Genre 1'], drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)


def train_and_evaluate(X, Y, model_name):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    print(f"Training set size ({model_name}):", X_train.shape)
    print(f"Cluster Distribution ({model_name}):", Counter(Y_train))

    smote = SMOTE(random_state=42, k_neighbors=min(3, min(Counter(Y_train).values()) - 1))
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_dist = {
        'rf__n_estimators': randint(10, 100),  # Increased range
        'rf__max_depth': [None, 10, 20, 30, 40, 50],  # Added more options
        'rf__min_samples_split': randint(2, 10),
        'rf__min_samples_leaf': randint(1, 10),
        'rf__bootstrap': [True, False],
        'rf__max_features': ['sqrt', 'log2', None]
    }

    pipeline = Pipeline([('smote', smote), ('rf', rf)])
    grid_search = RandomizedSearchCV(pipeline, param_dist, n_iter=20, cv=3, verbose=1, random_state=42)  # Increased n_iter
    grid_search.fit(X_train, Y_train)

    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    Y_pred = grid_search.best_estimator_.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='weighted')
    precision = precision_score(Y_test, Y_pred, average='weighted')
    recall = recall_score(Y_test, Y_pred, average='weighted')

    print(f"Accuracy ({model_name}): {accuracy:.2f}")
    print(f"F1 Score ({model_name}): {f1:.2f}")
    print(f"Precision ({model_name}): {precision:.2f}")
    print(f"Recall ({model_name}): {recall:.2f}")
    print(classification_report(Y_test, Y_pred))

    cm = confusion_matrix(Y_test, Y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({model_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


train_and_evaluate(X_pca, data['KMeans_Cluster'], 'KMeans')
train_and_evaluate(X_pca, data['Agglomerative_Cluster'], 'Agglomerative')
