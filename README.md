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
│   │   ├── encode/            
│   │   │   ├── cleaned_data2.csv
│   │   │   ├── encode_comparison.py
│   │   │   ├── normalized_data2.csv
│   │   │   ├── numeric2.csv
│   │   ├── normalize_comparison/ 
│   │   │   ├── cleaned_data.csv
│   │   │   ├── clustered_data2.csv
│   │   │   ├── comparison.py
│   │   │   ├── normalize.py
│   │   │   ├── normalize_data2.csv
│   │   ├── random_forest/     
│   │   │   ├── clustered_data.csv
│   │   │   ├── nn_random_forest.py
│   │   │   ├── nn_random_forest2.csv
│   │   │   ├── random_forest.py
│   │   ├── scrape2/           
│   │   │   ├── additional_scrape2.csv
│   │   │   ├── additional_scrape.py
│   │   │   ├── imdb_movies.csv
│   │   │   ├── main_scrape.py
│   │   │   ├── merge.py
│── .gitignore
│── README.md
│── requirements.txt
```

## Project Status
### **Completed Features:**
✔ Movie data scraping from IMDB
✔ Data cleaning and preprocessing
✔ Feature normalization
✔ Clustering using K-Means and Agglomerative Clustering
✔ Recommendation system based on clustering
✔ Evaluation with Silhouette Score and Davies-Bouldin Index

### **Future Improvements:**
- Wider dataset with more entries and more relevant columns such as reviews by both viewers and critics.
- Enhance recommendation logic with deep learning models.
- Improve clustering accuracy by tuning hyperparameters.
- Implement a web-based UI for user interaction.

---
**Developed by Gintautas | GMR**

