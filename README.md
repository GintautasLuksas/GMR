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
│   │   ├── recommendation/           # Clustering and recommendation analysis
│   │   │   ├── clustered_data2.csv
│   │   │   ├── cosing_euclidean.py
│   │   │   ├── recommendation_results_combined.csv
│   │   ├── recommendation_nn/        # Neural network comparison scripts
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

## Project Evaluation

*Scraping*
main_scrape.py from IMDB site where filters were set rating 9.9 - 7.00, user rated movie atleast 10000 times.
it scrapes Index, Title,Year, Rating,Length (mins), Rating Amount, Metascore, Group, Short Description.

additional.scrape.py same filter set. By pressing 'info' button collecting additional data: Directors, Stars,Genres.

2400 complete movies were scraped in this instance.
merge.py merges mimd_movies.csv and additional_data.csv creating complete_data.csv

*Cleaning*

cleaning.py loads a dataset, performs cleaning operations, and saves the cleaned data to two different CSV files.
The cleaning includes handling missing values, removing unwanted columns, and processing string columns.
Stars and Genres distributed to different columns up to 3 stars and Genres per movie.

Data saved to cleaned_data.csv
After cleaning 1630 full lines left.

*Normalize*

normalize.py using MinMaxScaler normalize numeric columns:
Year, Length (mins), Rating Amount, Rating, Metascore
saves data to normalized_data.csv

*Regular Clustering*

DBscan, Kmeans and Agglomerative methods were tested. Dbscan failed and was not used for comparison.
comparison.py 
elbow method and dendogram used to find the best cluster amount.
(elbow image)
(Dendogram image)

PCA used for 2 dimetinal view
Other metrics Silhouette Score and Davies-Bouldin Score for Kmeans are used
Silhouette Score used for Agglomerative.
3 clusters are chosen. 

Silhouette Score for K-Means: 0.313
Silhouette Score for Agglomerative: 0.281

(Silhouette Score Comparison Image)

(Clustering image)

Davies-Bouldin Score for KMeans with 3 clusters: 1.076



*Clustering with Neuro Network*

Text columns: Title, Short Description, Directors, Stars, Group, Genre  preprocessed.
Numeric normalized columns loaded
A pre-trained Sentence Transformer model (all-MiniLM-L6-v2) is used to generate vector embeddings for each text column (e.g., Title, Description, Directors, etc.). 
These embeddings convert the text into dense numerical vectors that capture the semantic meaning of the content.

numeric data columns are combined together with generated embedding form feature set called final_features, which will be used for clustering and other tasks.
autoencoder neural network model is built and trained.

elbow method and dendogram used.
(nn_elbow)
(nn_dendogram)
3 clusters used to be compared with regular clustering.


Silhouette Score (K-Means): 0.4046
Silhouette Score (Agglomerative Clustering): 0.3066

(Cluster Image)
Silhouette Score is better compared to regular clustering, but clusters are very far from each other, indicating that they are deiverse.

Random Forest

Cluster classification validation:

Performance on KMeans Test Set:
Accuracy: 0.60
F1 Score: 0.60
Precision: 0.62
Recall: 0.60
(Heatmap K-means)

Performance on Agglomerative Test Set:
Accuracy: 0.73
F1 Score: 0.74
Precision: 0.75
Recall: 0.73
(Heatmap Agglomerative)

Neuro Network Random Forest

Performance on KMeans Test Set:
Accuracy: 0.49
F1 Score: 0.49
Precision: 0.50
Recall: 0.49
(Heatmap K-means)

Performance on Agglomerative Test Set:
Accuracy (Agglomerative): 0.55
F1 Score (Agglomerative): 0.55
Precision (Agglomerative): 0.55
Recall (Agglomerative): 0.55
(Heatmap Agglomerative)


Best Performing clustering is by regular Agglomerative.

*Recomendation*
Using agglomerative cluster as must match value between set movie and entered movie
After filtering similarity score between rating and Metascore columns
After finding simmilar Genre movie.
After other numeric metrics are taken to account.


*Recommendation with Neuro Network*
Embeded data is used.



About tables. My system finds most simmilar movie from dataset. After set of simmilar movies together with watched movies is added to chatgpt for comparison.

Evaluation Criteria for Movie Recommendations
The evaluation of movie recommendations was conducted using the following criteria:

Genre Match
The primary genres of both watched and recommended movies were reviewed using credible movie databases and reviews (such as IMDb and Rotten Tomatoes) to determine exact or partial genre matches.

Rating Proximity
User ratings from platforms like IMDb and Rotten Tomatoes were examined to evaluate how closely the ratings align between the watched and recommended movies.

Length Similarity
The runtime of each movie was researched to compare their lengths, considering a threshold of 5 and 10 minutes.

Director Differences
The filmographies and styles of the directors were explored to assess whether they have a similar directorial style or produce similar types of films.

Thematic Alignment
Synopses, themes, and critical reviews of the movies were analyzed to identify strong thematic connections or overlaps.

ChatGPT Insights
Broader knowledge about movie trends, themes, and audience perceptions was utilized, based on discussions, articles, and analyses from reputable sources, to inform the ChatGPT column. This evaluated the overall similarity based on cultural context and critical reception.


By combining these aspects, a comprehensive evaluation framework was established to assess the quality of the model's movie recommendations.







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
- Implement a web-based UI for user interaction.

---
**Developed by Gintautas | GMR**

