"""
Unit tests for the Movie Recommender System
"""

import unittest
import tempfile
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import time

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.recommender import MovieRecommenderAPI, LocalMovieRecommender
from database import MovieDatabase


class TestLocalMovieRecommender(unittest.TestCase):
    """Test the LocalMovieRecommender class"""

    def setUp(self):
        """Set up test data"""
        # Create sample data
        self.user_item_matrix = pd.DataFrame(
            {
                1: [5, 3, 0, 4, 0],
                2: [0, 4, 5, 0, 3],
                3: [3, 0, 4, 5, 0],
                4: [0, 5, 0, 3, 4],
                5: [4, 0, 3, 0, 5],
            },
            index=[1, 2, 3, 4, 5],
        )

        self.movies_df = pd.DataFrame(
            {
                "movieId": [1, 2, 3, 4, 5],
                "title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
                "genres": ["Action", "Comedy", "Drama", "Thriller", "Romance"],
            }
        )

        self.recommender = LocalMovieRecommender(
            self.user_item_matrix, self.movies_df, n_components=3
        )

    def test_initialization(self):
        """Test recommender initialization"""
        self.assertEqual(self.recommender.n_components, 3)
        self.assertEqual(len(self.recommender.user_mapping), 5)
        self.assertEqual(len(self.recommender.item_mapping), 5)

    def test_fit(self):
        """Test model fitting"""
        result = self.recommender.fit()
        self.assertIsNotNone(self.recommender.svd)
        self.assertIsNotNone(self.recommender.user_factors)
        self.assertIsNotNone(self.recommender.item_factors)
        self.assertEqual(result, self.recommender)

    def test_get_recommendations(self):
        """Test getting recommendations"""
        self.recommender.fit()
        recommendations = self.recommender.get_recommendations(1, 3)

        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 3)

        if recommendations:
            self.assertIn("movieId", recommendations[0])
            self.assertIn("title", recommendations[0])
            self.assertIn("genres", recommendations[0])
            self.assertIn("similarity_score", recommendations[0])

    def test_get_similar_movies(self):
        """Test getting similar movies"""
        self.recommender.fit()
        similar_movies = self.recommender.get_similar_movies(1, 3)

        self.assertIsInstance(similar_movies, list)
        self.assertLessEqual(len(similar_movies), 3)

        if similar_movies:
            self.assertIn("movieId", similar_movies[0])
            self.assertIn("title", similar_movies[0])
            self.assertIn("genres", similar_movies[0])
            self.assertIn("similarity_score", similar_movies[0])

    def test_invalid_user_id(self):
        """Test handling of invalid user ID"""
        self.recommender.fit()
        recommendations = self.recommender.get_recommendations(999, 3)
        self.assertEqual(recommendations, [])

    def test_invalid_movie_id(self):
        """Test handling of invalid movie ID"""
        self.recommender.fit()
        similar_movies = self.recommender.get_similar_movies(999, 3)
        self.assertEqual(similar_movies, [])


class TestMovieRecommenderAPI(unittest.TestCase):
    """Test the MovieRecommenderAPI class"""

    def setUp(self):
        """Set up test data"""
        # Create a temporary model file
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")

        # Create a mock recommender
        self.mock_recommender = MagicMock()
        self.mock_recommender.movies_df = pd.DataFrame(
            {
                "movieId": [1, 2, 3],
                "title": ["Test Movie 1", "Test Movie 2", "Test Movie 3"],
                "genres": ["Action", "Comedy", "Drama"],
            }
        )
        self.mock_recommender.user_mapping = {1: 0, 2: 1}

        # Mock the get_recommendations method
        self.mock_recommender.get_recommendations.return_value = [
            {
                "movieId": 1,
                "title": "Test Movie 1",
                "genres": "Action",
                "similarity_score": 0.8,
            }
        ]

        # Mock the get_similar_movies method
        self.mock_recommender.get_similar_movies.return_value = [
            {
                "movieId": 2,
                "title": "Test Movie 2",
                "genres": "Comedy",
                "similarity_score": 0.7,
            }
        ]

    def tearDown(self):
        """Clean up test files"""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("pickle.dump")
    @patch("builtins.open", create=True)
    def test_load_model_success(self, mock_open, mock_pickle_dump):
        """Test successful model loading"""
        # Mock file existence and pickle load
        with patch("os.path.exists", return_value=True):
            with patch("pickle.load", return_value=self.mock_recommender):
                api = MovieRecommenderAPI(self.model_path)
                self.assertIsNotNone(api.recommender)

    def test_load_model_not_found(self):
        """Test model loading when file doesn't exist"""
        with patch("os.path.exists", return_value=False):
            api = MovieRecommenderAPI(self.model_path)
            self.assertIsNone(api.recommender)

    def test_get_user_recommendations_success(self):
        """Test successful user recommendations"""
        api = MovieRecommenderAPI()
        api.recommender = self.mock_recommender

        result = api.get_user_recommendations(1, 5)

        self.assertNotIn("error", result)
        self.assertEqual(result["user_id"], 1)
        self.assertEqual(result["count"], 1)
        self.assertIn("recommendations", result)

    def test_get_user_recommendations_no_model(self):
        """Test user recommendations when model is not loaded"""
        api = MovieRecommenderAPI()
        api.recommender = None

        result = api.get_user_recommendations(1, 5)

        self.assertIn("error", result)
        self.assertIn("Model not loaded", result["error"])

    def test_get_similar_movies_success(self):
        """Test successful similar movies"""
        api = MovieRecommenderAPI()
        api.recommender = self.mock_recommender

        result = api.get_similar_movies(1, 5)

        self.assertNotIn("error", result)
        self.assertIn("similar_movies", result)

    def test_get_movie_info_success(self):
        """Test successful movie info retrieval"""
        api = MovieRecommenderAPI()
        api.recommender = self.mock_recommender

        result = api.get_movie_info(1)

        self.assertNotIn("error", result)
        self.assertEqual(result["movie_id"], 1)
        self.assertEqual(result["title"], "Test Movie 1")
        self.assertEqual(result["genres"], "Action")

    def test_get_movie_info_not_found(self):
        """Test movie info when movie doesn't exist"""
        api = MovieRecommenderAPI()
        api.recommender = self.mock_recommender

        result = api.get_movie_info(999)

        self.assertIn("error", result)
        self.assertIn("not found", result["error"])

    def test_search_movies_success(self):
        """Test successful movie search"""
        api = MovieRecommenderAPI()
        api.recommender = self.mock_recommender

        result = api.search_movies("Test", 5)

        self.assertNotIn("error", result)
        self.assertEqual(result["query"], "test")
        self.assertIn("movies", result)
        self.assertIn("count", result)

    def test_get_available_users(self):
        """Test getting available users"""
        api = MovieRecommenderAPI()
        api.recommender = self.mock_recommender

        result = api.get_available_users(5)

        self.assertNotIn("error", result)
        self.assertIn("users", result)
        self.assertIn("count", result)
        self.assertIn("total_users", result)

    def test_get_available_movies(self):
        """Test getting available movies"""
        api = MovieRecommenderAPI()
        api.recommender = self.mock_recommender

        result = api.get_available_movies(5)

        self.assertNotIn("error", result)
        self.assertIn("movies", result)
        self.assertIn("count", result)
        self.assertIn("total_movies", result)

    @patch("ollama.chat")
    def test_ask_llm_success(self, mock_ollama_chat):
        """Test successful LLM query"""
        # Mock Ollama response
        mock_ollama_chat.return_value = {
            "message": {"content": "1. Die Hard\n2. The Matrix\n3. Mission Impossible"}
        }

        api = MovieRecommenderAPI()
        api.recommender = self.mock_recommender

        result = api.ask_llm("action movies", model="phi", n_recommendations=5)

        self.assertNotIn("error", result)
        self.assertEqual(result["query"], "action movies")
        self.assertEqual(result["model_used"], "phi")
        self.assertIn("recommended_movies", result)
        self.assertIn("llm_response", result)

    def test_ask_llm_invalid_model(self):
        """Test LLM query with invalid model"""
        api = MovieRecommenderAPI()
        api.recommender = self.mock_recommender

        result = api.ask_llm(
            "action movies", model="invalid-model", n_recommendations=5
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["model_used"], "phi")  # Should default to phi


class TestMovieDatabase(unittest.TestCase):
    """Test the MovieDatabase class"""

    def setUp(self):
        """Set up test database"""
        self.temp_db_path = tempfile.mktemp(suffix=".db")
        self.db = MovieDatabase(self.temp_db_path)

    def tearDown(self):
        """Clean up test database"""
        # No need to close connection since we create new ones per operation
        # Just try to remove the file with error handling
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except PermissionError:
                # On Windows, sometimes the file is still in use
                # We'll just leave it and let the OS clean it up later
                pass

    def test_create_user_success(self):
        """Test successful user creation"""
        result = self.db.create_user("testuser", "test@example.com", "hashed_password")
        self.assertTrue(result)

    def test_create_user_duplicate(self):
        """Test user creation with duplicate username"""
        # Create first user
        self.db.create_user("testuser", "test@example.com", "hashed_password")

        # Try to create duplicate
        result = self.db.create_user(
            "testuser", "test2@example.com", "hashed_password2"
        )
        self.assertFalse(result)

    def test_get_user_by_username_success(self):
        """Test successful user retrieval"""
        self.db.create_user("testuser", "test@example.com", "hashed_password")

        user = self.db.get_user_by_username("testuser")

        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "testuser")
        self.assertEqual(user["email"], "test@example.com")
        self.assertEqual(user["password_hash"], "hashed_password")

    def test_get_user_by_username_not_found(self):
        """Test user retrieval when user doesn't exist"""
        user = self.db.get_user_by_username("nonexistent")
        self.assertIsNone(user)

    def test_add_rating_success(self):
        """Test successful rating addition"""
        self.db.create_user("testuser", "test@example.com", "hashed_password")
        user = self.db.get_user_by_username("testuser")

        result = self.db.add_rating(user["id"], 1, 4.5)
        self.assertTrue(result)

    def test_get_user_ratings(self):
        """Test getting user ratings"""
        self.db.create_user("testuser", "test@example.com", "hashed_password")
        user = self.db.get_user_by_username("testuser")

        # Add some ratings
        self.db.add_rating(user["id"], 1, 4.5)
        self.db.add_rating(user["id"], 2, 3.5)

        ratings = self.db.get_user_ratings(user["id"])

        self.assertEqual(len(ratings), 2)
        self.assertEqual(ratings[0]["movie_id"], 1)
        self.assertEqual(ratings[0]["rating"], 4.5)

    def test_add_to_watchlist_success(self):
        """Test successful watchlist addition"""
        self.db.create_user("testuser", "test@example.com", "hashed_password")
        user = self.db.get_user_by_username("testuser")

        result = self.db.add_to_watchlist(user["id"], 1)
        self.assertTrue(result)

    def test_get_watchlist(self):
        """Test getting user watchlist"""
        self.db.create_user("testuser", "test@example.com", "hashed_password")
        user = self.db.get_user_by_username("testuser")

        # Add some movies to watchlist
        self.db.add_to_watchlist(user["id"], 1)
        self.db.add_to_watchlist(user["id"], 2)

        watchlist = self.db.get_watchlist(user["id"])

        self.assertEqual(len(watchlist), 2)
        self.assertIn(1, watchlist)
        self.assertIn(2, watchlist)

    def test_save_and_get_user_preferences(self):
        """Test saving and retrieving user preferences"""
        self.db.create_user("testuser", "test@example.com", "hashed_password")
        user = self.db.get_user_by_username("testuser")

        # Save preferences
        genres = ["Action", "Comedy"]
        decades = ["1990s", "2000s"]
        max_rating = 4.5

        result = self.db.save_user_preferences(user["id"], genres, decades, max_rating)
        self.assertTrue(result)

        # Get preferences
        prefs = self.db.get_user_preferences(user["id"])

        self.assertIsNotNone(prefs)
        self.assertEqual(prefs["favorite_genres"], genres)
        self.assertEqual(prefs["preferred_decades"], decades)
        self.assertEqual(prefs["max_rating"], max_rating)


if __name__ == "__main__":
    unittest.main()
