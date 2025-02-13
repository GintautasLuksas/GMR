# Gintautas Movie Recommendation System

## Overview
This project is a movie recommendation system that uses clustering and deep learning techniques to suggest movies based on various parameters. The system primarily considers **user ratings (Rating)** and **critic scores (Metascore)** as the most significant factors when comparing movies.

## Workflow
The project consists of several scripts, each responsible for different stages of data collection, cleaning, clustering, and evaluation:

### 1. **Data Scraping**
- **`Main_scrape.py`**: Scrapes essential movie details from IMDb, including Title, Metascore, Group, Year, Length, Rating, Rate Amount, and Short Description. It filters movies with a **minimum rating of 7.0** and at least **10,000 user ratings**.
- **`Additional_scrape.py`**: Extracts additional data such as Directors, Stars, and Genres.

### 2. **Data Processing**
- **`Merge_scraped_data.py`**: Merges the collected data into a structured dataset.
- **`cleaning.py`**: Cleans missing data, separates Stars and Genre into different columns, and removes numbering from movie titles.
- **`normalize.py`**: Normalizes numerical features using **MinMaxScaler**, including Length (mins), Rating Amount, Metascore, and Rating.

### 3. **Clustering & Feature Importance Analysis**
- **`comparison.py`**: Performs **KMeans** and **Agglomerative clustering** and evaluates models using **Silhouette Score**.
  - **KMeans Score**: **0.313**
  - **Agglomerative Score**: **0.281**

  **Feature Importance in KMeans Clusters:**
  - **Cluster 0**: High **Rating (0.476)**, High **Metascore (0.755)**
  - **Cluster 1**: Moderate **Metascore (0.784)**, Low **Rating Amount**
  - **Cluster 2**: Lower **Metascore (0.512)**, Balanced Rating

### 4. **Deep Learning & Encoding**
- **`Encode_comparison.py`**: Utilizes **SentenceTransformer** to encode textual data and runs it through an **autoencoder neural network**.
- Applies **KMeans** and **Agglomerative clustering** on embedded data.
- **Silhouette Scores for Encoded Data:**
  - **KMeans**: **0.4308**
  - **Agglomerative**: **0.3613**

### 5. **Model Evaluation**
- **`Random_forest.py`**: Uses **Random Forest** on encoded data to validate cluster assignments.
  - **KMeans Clustering Results:**
    - Accuracy: **53.37%**
    - Macro AUC: **0.71**
  - **Agglomerative Clustering Results:**
    - Accuracy: **53.37%**
    - Macro AUC: **0.71**

### 6. **System Testing with Lower Rated Movies**
- **`system_test_data`**: Contains a test dataset with **50 movies** (Rating **5-7**) from **2023-2025** to evaluate the model.

### 7. **Similarity Analysis**
- **`Cosing_euclidean.py`**: Measures model effectiveness by comparing movies using a combination of **Cosine and Euclidean** similarity metrics.
- Recommendations are cluster-bound, meaning a movie is first assigned a cluster before comparing additional metrics.

## Installation & Usage
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the scraping scripts to collect data:
   ```bash
   python Main_scrape.py
   python Additional_scrape.py
   ```
3. Merge and clean the data:
   ```bash
   python Merge_scraped_data.py
   python cleaning.py
   ```
4. Normalize and encode data:
   ```bash
   python normalize.py
   python Encode_comparison.py
   ```
5. Run clustering analysis:
   ```bash
   python comparison.py
   ```
6. Evaluate the model:
   ```bash
   python Random_forest.py
   ```
7. Test recommendation system:
   ```bash
   python Cosing_euclidean.py
   ```

## Future Improvements
- Improve clustering accuracy by using additional feature engineering techniques.
- Experiment with different deep learning models for better encoding.
- Implement a user-friendly web interface for movie recommendations.

## Author
Gintautas

---
This project is designed to provide intelligent movie recommendations based on IMDb data, leveraging clustering techniques and deep learning embeddings for improved results.

