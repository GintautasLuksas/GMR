# Gintautas Movie Recommendation (VCS GMR)

## Description
Gintautas Movie Recommendation (GMR) is a project that scrapes, cleans, and clusters movie data from IMDB to build and compare a recommendation system. The project applies machine learning techniques such as clustering and natural language processing (NLP) to recommend movies based on various features like rating, length, metascore, and user reviews.

## Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/GintautasLuksas/GMR.git
   cd VCS_GMR
   ```

2. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Scraping Script**
   ```bash
   python main_scrape.py
   python additional_scrape.py
   ```

4. **Merge and Clean Data**
   ```bash
   python merge.py
   python cleaning.py
   ```

5. **Normalize Data**
   ```bash
   python normalize.py
   ```

6. **Run Regular and NN Clustering**
   ```bash
   python comparison.py
   python encode_comparison.py
   ```

7. **Run Regular and NN Clustered Data with Random Forest**
   ```bash
   python random_forest.py
   python nn_random_forest.py
   ```

8. **Run Regular and NN Clustering**
   ```bash
   python cosing_euclidean.py
   python nn_cosing_euclidean.py
   ```

## Project Structure
```
VCS_GMR/
│── src/
│   ├── main/
│   │   ├── clean/             # Data cleaning scripts
│   │   ├── recommendation/    # Clustering and recommendation analysis
│   │   ├── recommendation_nn/ # Neural network comparison scripts
│   │   ├── encode/            # Encoding and processing numerical columns
│   │   ├── normalize_comparison/ # Normalization and comparison analysis
│   │   ├── random_forest/     # Random forest model implementations
│   │   ├── scrape/            # Web scraping and data collection
│   │   ├── system_test_data/  # Testing dataset handling
│── .gitignore
│── README.md
│── requirements.txt
```

## Project Evaluation

### Scraping
- **`main_scrape.py`** scrapes IMDB with filters set for movies rated 9.9 - 7.0, with at least 10,000 user ratings.
- **`additional_scrape.py`** collects additional data by pressing the 'info' button, including Directors, Stars, and Genres.
- 2400 complete movies were scraped.
- **`merge.py`** merges `imdb_movies.csv` and `additional_data.csv` to create `complete_data.csv`.

### Cleaning
- **`cleaning.py`** loads the dataset, performs cleaning operations, and saves the cleaned data.
- Stars and Genres are distributed into separate columns (up to 3 per movie).
- After cleaning, 1630 full records remain.

### Normalization
- **`normalize.py`** uses MinMaxScaler to normalize numeric columns: Year, Length (mins), Rating Amount, Rating, and Metascore.

### Regular Clustering
- DBScan, K-Means, and Agglomerative Clustering were tested. DBScan failed and was excluded.
- **`comparison.py`** applies:
  - Elbow method and dendrogram to determine optimal clusters.
  - PCA for a 2D view.
  - Silhouette Score and Davies-Bouldin Score for evaluation.
  - **Chosen Clusters:** 3

  **Elbow Method Visualization:**
  _[Insert Elbow Image Here]_  
  **Dendrogram:**
  _[Insert Dendrogram Image Here]_

- **Silhouette Scores:**
  - K-Means: **0.313**
  - Agglomerative: **0.281**

  **Silhouette Score Comparison:**
  _[Insert Silhouette Score Image Here]_

  **Cluster Visualization:**
  _[Insert Clustering Image Here]_

- **Davies-Bouldin Score for KMeans (3 clusters):** **1.076**

### Clustering with Neural Network
- Text columns (Title, Short Description, Directors, Stars, Group, Genre) are preprocessed.
- A pre-trained Sentence Transformer model (`all-MiniLM-L6-v2`) generates embeddings.
- Autoencoder neural network is built and trained.

  **Elbow Method & Dendrogram:**
  _[Insert NN Elbow & Dendrogram Images Here]_

- **Silhouette Scores:**
  - K-Means: **0.4046**
  - Agglomerative: **0.3066**

  **Cluster Visualization:**
  _[Insert Cluster Image Here]_

### Random Forest Classification
#### Regular Clustering
- **K-Means Performance:**
  - Accuracy: **0.60**
  - F1 Score: **0.60**
  - Precision: **0.62**
  - Recall: **0.60**
  _[Insert Heatmap K-Means Image Here]_

- **Agglomerative Performance:**
  - Accuracy: **0.73**
  - F1 Score: **0.74**
  - Precision: **0.75**
  - Recall: **0.73**
  _[Insert Heatmap Agglomerative Image Here]_

#### Neural Network Random Forest
- **K-Means Performance:**
  - Accuracy: **0.49**
  - F1 Score: **0.49**
  - Precision: **0.50**
  - Recall: **0.49**
  _[Insert Heatmap K-Means Image Here]_

- **Agglomerative Performance:**
  - Accuracy: **0.55**
  - F1 Score: **0.55**
  - Precision: **0.55**
  - Recall: **0.55**
  _[Insert Heatmap Agglomerative Image Here]_

**Best Performing Clustering:** **Regular Agglomerative**

## Recommendation System
### Regular Recommendation
- Uses Agglomerative clusters as the primary similarity metric.
- Filters based on rating, Metascore, Genre, and numeric features.

### Neural Network Recommendation
- Uses embedded data for recommendations.

## Evaluation Criteria for Movie Recommendations
- **Genre Match**: Verified via IMDb & Rotten Tomatoes.
- **Rating Proximity**: Compared across IMDb and Rotten Tomatoes.
- **Length Similarity**: Thresholds of ±5 and ±10 minutes.
- **Director Differences**: Style comparison.
- **Thematic Alignment**: Based on reviews and summaries.
- **ChatGPT Insights**: AI-based movie context analysis.

---

## Project Status
### **Completed Features:**
✔ Movie data scraping from IMDB  
✔ Data cleaning and preprocessing  
✔ Feature normalization  
✔ Clustering using K-Means and Agglomerative Clustering  
✔ Recommendation system based on clustering  
✔ Evaluation with Silhouette Score and Davies-Bouldin Index  

### **Future Improvements:**
- Expand dataset with more relevant columns (e.g., user & critic reviews).
- Enhance recommendation logic with deep learning models.
- Implement a web-based UI for user interaction.

---
**Developed by Gintautas | GMR**

