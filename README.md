# Gintautas Movie Recommendation (VCS GMR)

## Description
Gintautas Movie Recommendation (VCS GMR) is a project that scrapes, cleans, and clusters movie data from IMDB to build a recommendation system. The project applies machine learning techniques such as clustering and natural language processing (NLP) to recommend movies based on various features like rating, length, and user reviews.

## Installation
1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd VCS_GMR
   ```

2. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**
   - Create a `.env` file in the project root directory.
   - Add the following:
     ```env
     DB_HOST=<your_database_host>
     DB_USER=<your_database_user>
     DB_PASSWORD=<your_database_password>
     DB_NAME=<your_database_name>
     ```

4. **Run the Scraping Script**
   ```bash
   python main_scrape.py
   ```

5. **Merge and Clean Data**
   ```bash
   python merge.py
   python cleaning.py
   ```

6. **Normalize Data**
   ```bash
   python normalize.py
   ```

7. **Run Clustering and Recommendation System**
   ```bash
   python comparison.py
   python encode_comparison.py
   ```

## Project Structure
```
VCS_GMR/
│── src/
│   ├── main/
│   │   ├── clean/             # Data cleaning scripts
│   │   │   ├── cleaning.py
│   │   │   ├── complete_data.csv
│   │   ├── compare/           # Clustering and recommendation analysis
│   │   │   ├── clustered_data2.csv
│   │   │   ├── cosing_euclidean.py
│   │   │   ├── recommendation_results_combined.csv
│   │   ├── compare_nn/        # Neural network comparison scripts
│   │   │   ├── nn_cluster.csv
│   │   │   ├── nn_cosing_euclidean.py
│   │   ├── encode/            # Encoding and processing numerical columns
│   │   │   ├── cleaned_data.csv
│   │   │   ├── encode_comparison.py
│   │   │   ├── normalized_data.csv
│   │   ├── normalize_comparison/ # Normalization and comparison analysis
│   │   │   ├── cleaned_data.csv
│   │   │   ├── comparison.py
│   │   │   ├── normalize.py
│   │   ├── random_forest/     # Random forest model implementations
│   │   │   ├── clustered_data.csv
│   │   │   ├── nn_random_forest.py
│   │   │   ├── random_forest.py
│   │   ├── scrape/            # Web scraping and data collection
│   │   │   ├── additional_scrape.py
│   │   │   ├── imdb_movies.csv
│   │   │   ├── main_scrape.py
│   │   │   ├── merge.py
│   │   ├── system_test_data/  # Testing dataset handling
│   │   │   ├── cleaning.py
│   │   │   ├── complete_data2.csv
│── .gitignore
│── README.md
│── requirements.txt
```

## Class Diagram
```
+----------------------+
| MovieScraper        |
+----------------------+
| - scrape_movies()   |
| - save_to_csv()     |
+----------------------+
       |
       v
+----------------------+
| DataCleaner         |
+----------------------+
| - clean_data()      |
| - split_values()    |
+----------------------+
       |
       v
+----------------------+
| Normalizer          |
+----------------------+
| - normalize_data()  |
+----------------------+
       |
       v
+----------------------+
| ClusterComparison   |
+----------------------+
| - kmeans_clustering() |
| - agglomerative_clustering() |
+----------------------+
       |
       v
+----------------------+
| Recommendation      |
+----------------------+
| - recommend_movies() |
+----------------------+
```
**Description:**
- `MovieScraper`: Handles movie data extraction from IMDB.
- `DataCleaner`: Cleans and preprocesses data.
- `Normalizer`: Normalizes movie features using MinMaxScaler.
- `ClusterComparison`: Applies K-Means and Agglomerative clustering.
- `Recommendation`: Generates movie recommendations.

## Database Diagram
```
+--------------------+
| Movies            |
+--------------------+
| id (PK)           |
| title             |
| year              |
| rating            |
| length (mins)     |
| rating_amount     |
| metascore         |
| group             |
| description       |
+--------------------+
       |
       v
+--------------------+
| Additional Data   |
+--------------------+
| id (PK, FK)       |
| directors         |
| stars            |
| genres           |
+--------------------+
```
**Description:**
- `Movies`: Stores main movie information.
- `Additional Data`: Stores director, star, and genre details linked to movies.

## Project Status
### **Completed Features:**
✔ Movie data scraping from IMDB
✔ Data cleaning and preprocessing
✔ Feature normalization
✔ Clustering using K-Means and Agglomerative Clustering
✔ Recommendation system based on clustering
✔ Evaluation with Silhouette Score and Davies-Bouldin Index

### **Future Improvements:**
- Improve clustering accuracy by tuning hyperparameters
- Enhance recommendation logic with deep learning models
- Implement a web-based UI for user interaction
- Add real-time movie data updates

---
**Developed by Gintautas | VCS GMR**

