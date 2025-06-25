# STEP 1 - IMPORTING LIBRARIES
import time
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import os
import requests
import zipfile
from pathlib import Path
from recommender import LocalMovieRecommender

# STEP 2 - FETCH THE DATASET
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Download MovieLens dataset if not already present
dataset_url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
zip_path = os.path.join(data_dir, "ml-latest-small.zip")
dataset_dir = os.path.join(data_dir, "ml-latest-small")

if not os.path.exists(dataset_dir):
    print("Downloading MovieLens dataset...")
    response = requests.get(dataset_url)
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Remove the zip file
    os.remove(zip_path)

print("Dataset files:", os.listdir(dataset_dir))

# STEP 3 - LOAD AND PREPARE THE DATA
print("Loading and preparing data...")
ratings_df = pd.read_csv(os.path.join(dataset_dir, "ratings.csv"))
movies_df = pd.read_csv(os.path.join(dataset_dir, "movies.csv"))

print("Ratings dataset info:")
print(ratings_df.info())
print("\nFirst few ratings:")
print(ratings_df.head())

print("\nMovies dataset info:")
print(movies_df.info())
print("\nFirst few movies:")
print(movies_df.head())

# STEP 4 - CREATE USER-ITEM MATRIX
print("\nCreating user-item matrix...")
# Create a pivot table with users as rows and movies as columns
user_item_matrix = ratings_df.pivot_table(
    index="userId", columns="movieId", values="rating", fill_value=0
)

print(f"User-item matrix shape: {user_item_matrix.shape}")
print(f"Number of users: {user_item_matrix.shape[0]}")
print(f"Number of movies: {user_item_matrix.shape[1]}")

# STEP 5 - TRAIN THE MODEL
print("\nTraining the recommendation model...")
recommender = LocalMovieRecommender(user_item_matrix, movies_df, n_components=50)
recommender.fit()

# STEP 6 - TEST RECOMMENDATIONS
print("\n" + "=" * 50)
print("TESTING RECOMMENDATIONS")
print("=" * 50)

# Get a random user for testing
test_user = user_item_matrix.index[np.random.randint(0, len(user_item_matrix.index))]
print(f"\nGetting recommendations for user {test_user}:")

recommendations = recommender.get_recommendations(test_user, n_recommendations=10)

print(f"\nTop 10 movie recommendations for user {test_user}:")
for i, rec in enumerate(recommendations, 1):
    print(
        f"{i}. {rec['title']} (Genres: {rec['genres']}) - Score: {rec['similarity_score']:.4f}"
    )

# Test similar movies
print(f"\n" + "=" * 50)
print("TESTING SIMILAR MOVIES")
print("=" * 50)

# Get a random movie for testing
test_movie_id = movies_df["movieId"].sample(1).iloc[0]
test_movie_title = movies_df[movies_df["movieId"] == test_movie_id]["title"].iloc[0]

print(f"\nFinding movies similar to '{test_movie_title}' (ID: {test_movie_id}):")

similar_movies = recommender.get_similar_movies(test_movie_id, n_similar=10)

print(f"\nTop 10 similar movies to '{test_movie_title}':")
for i, movie in enumerate(similar_movies, 1):
    print(
        f"{i}. {movie['title']} (Genres: {movie['genres']}) - Score: {movie['similarity_score']:.4f}"
    )

# STEP 7 - SAVE THE MODEL
print(f"\n" + "=" * 50)
print("SAVING MODEL")
print("=" * 50)

import pickle

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the recommender model
model_path = "models/recommender.pkl"
with open(model_path, "wb") as f:
    pickle.dump(recommender, f)

print(f"Model saved to {model_path}")

# STEP 8 - EVALUATION METRICS
print(f"\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)


def calculate_coverage(recommender, user_item_matrix, n_recommendations=10):
    """Calculate recommendation coverage"""
    all_recommended_items = set()
    total_users = min(
        100, len(user_item_matrix.index)
    )  # Sample 100 users for efficiency

    for user_id in user_item_matrix.index[:total_users]:
        recommendations = recommender.get_recommendations(user_id, n_recommendations)
        for rec in recommendations:
            all_recommended_items.add(rec["movieId"])

    coverage = len(all_recommended_items) / len(user_item_matrix.columns)
    return coverage


# Calculate coverage
coverage = calculate_coverage(recommender, user_item_matrix)
print(f"Recommendation Coverage: {coverage:.4f} ({coverage*100:.2f}%)")

# Calculate sparsity
total_ratings = (user_item_matrix != 0).sum().sum()
total_possible_ratings = user_item_matrix.shape[0] * user_item_matrix.shape[1]
sparsity = 1 - (total_ratings / total_possible_ratings)
print(f"Data Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")

print(f"\n" + "=" * 50)
print("LOCAL MOVIE RECOMMENDER SYSTEM READY!")
print("=" * 50)
print("The system has been trained and is ready to provide recommendations.")
print("Model saved to: models/recommender.pkl")
print("You can now use the recommender object to get recommendations for any user.")
