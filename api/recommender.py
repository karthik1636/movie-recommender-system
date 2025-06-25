import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import ollama
import re
import time
from utils.monitoring import get_performance_monitor


class LocalMovieRecommender:
    def __init__(self, user_item_matrix, movies_df, n_components=50):
        self.user_item_matrix = user_item_matrix
        self.movies_df = movies_df
        self.n_components = n_components
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = {
            user_id: idx for idx, user_id in enumerate(user_item_matrix.index)
        }
        self.item_mapping = {
            item_id: idx for idx, item_id in enumerate(user_item_matrix.columns)
        }
        self.reverse_item_mapping = {
            idx: item_id for item_id, idx in self.item_mapping.items()
        }

    def fit(self):
        """Train the SVD model"""
        print("Training SVD model...")
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_factors = self.svd.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd.components_.T

        print(
            f"Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}"
        )
        return self

    def get_recommendations(self, user_id, n_recommendations=10):
        """Get movie recommendations for a user"""
        if user_id not in self.user_mapping:
            print(f"User {user_id} not found in training data")
            return []

        user_idx = self.user_mapping[user_id]
        user_vector = self.user_factors[user_idx].reshape(1, -1)

        # Calculate similarity with all items
        similarities = cosine_similarity(user_vector, self.item_factors)[0]

        # Get indices of top similar items
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]

        recommendations = []
        for idx in top_indices:
            movie_id = self.reverse_item_mapping[idx]
            movie_info = self.movies_df[self.movies_df["movieId"] == movie_id]
            if not movie_info.empty:
                title = movie_info.iloc[0]["title"]
                genres = movie_info.iloc[0]["genres"]
                similarity_score = similarities[idx]
                recommendations.append(
                    {
                        "movieId": movie_id,
                        "title": title,
                        "genres": genres,
                        "similarity_score": similarity_score,
                    }
                )

        return recommendations

    def get_similar_movies(self, movie_id, n_similar=10):
        """Find similar movies based on item-item similarity"""
        if movie_id not in self.item_mapping:
            print(f"Movie {movie_id} not found in training data")
            return []

        movie_idx = self.item_mapping[movie_id]
        movie_vector = self.item_factors[movie_idx].reshape(1, -1)

        # Calculate similarity with all other movies
        similarities = cosine_similarity(movie_vector, self.item_factors)[0]

        # Get indices of top similar movies (excluding itself)
        top_indices = np.argsort(similarities)[::-1][1 : n_similar + 1]

        similar_movies = []
        for idx in top_indices:
            similar_movie_id = self.reverse_item_mapping[idx]
            movie_info = self.movies_df[self.movies_df["movieId"] == similar_movie_id]
            if not movie_info.empty:
                title = movie_info.iloc[0]["title"]
                genres = movie_info.iloc[0]["genres"]
                similarity_score = similarities[idx]
                similar_movies.append(
                    {
                        "movieId": similar_movie_id,
                        "title": title,
                        "genres": genres,
                        "similarity_score": similarity_score,
                    }
                )

        return similar_movies


class MovieRecommenderAPI:
    def __init__(self, model_path="models/recommender.pkl"):
        """Initialize the Movie Recommender API"""
        self.model_path = model_path
        self.recommender = None
        self.monitor = get_performance_monitor()
        self.load_model()

    def load_model(self):
        """Load the trained recommender model"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    self.recommender = pickle.load(f)
                print(f"Model loaded successfully from {self.model_path}")
            else:
                print(f"Model file not found at {self.model_path}")
                print("Please run main.py first to train the model")
        except Exception as e:
            print(f"Error loading model: {e}")

    def get_user_recommendations(self, user_id, n_recommendations=10):
        """Get movie recommendations for a specific user"""
        if self.recommender is None:
            return {"error": "Model not loaded. Please run main.py first."}

        try:
            start = time.time()
            recommendations = self.recommender.get_recommendations(
                user_id, n_recommendations
            )
            latency = time.time() - start
            self.monitor.record_recommendation_request(
                user_id=user_id,
                algorithm="svd",
                num_recommendations=len(recommendations),
                response_time=latency,
            )
            return {
                "user_id": user_id,
                "recommendations": recommendations,
                "count": len(recommendations),
                "latency": latency,
            }
        except Exception as e:
            self.monitor.logger.log_error_with_context(
                e, {"user_id": user_id, "action": "get_user_recommendations"}
            )
            return {"error": f"Error getting recommendations: {e}"}

    def get_similar_movies(self, movie_id, n_similar=10):
        """Get similar movies for a specific movie"""
        if self.recommender is None:
            return {"error": "Model not loaded. Please run main.py first."}

        try:
            similar_movies = self.recommender.get_similar_movies(movie_id, n_similar)
            return {
                "movie_id": movie_id,
                "similar_movies": similar_movies,
                "count": len(similar_movies),
            }
        except Exception as e:
            return {"error": f"Error getting similar movies: {e}"}

    def get_movie_info(self, movie_id):
        """Get information about a specific movie"""
        if self.recommender is None:
            return {"error": "Model not loaded. Please run main.py first."}

        try:
            movie_info = self.recommender.movies_df[
                self.recommender.movies_df["movieId"] == movie_id
            ]
            if not movie_info.empty:
                return {
                    "movie_id": movie_id,
                    "title": movie_info.iloc[0]["title"],
                    "genres": movie_info.iloc[0]["genres"],
                }
            else:
                return {"error": f"Movie with ID {movie_id} not found"}
        except Exception as e:
            return {"error": f"Error getting movie info: {e}"}

    def get_available_users(self, limit=10):
        """Get a list of available users in the system"""
        if self.recommender is None:
            return {"error": "Model not loaded. Please run main.py first."}

        try:
            users = list(self.recommender.user_mapping.keys())[:limit]
            return {
                "users": users,
                "count": len(users),
                "total_users": len(self.recommender.user_mapping),
            }
        except Exception as e:
            return {"error": f"Error getting users: {e}"}

    def get_available_movies(self, limit=10):
        """Get a list of available movies in the system"""
        if self.recommender is None:
            return {"error": "Model not loaded. Please run main.py first."}

        try:
            movies = self.recommender.movies_df.head(limit)[
                ["movieId", "title", "genres"]
            ].to_dict("records")
            return {
                "movies": movies,
                "count": len(movies),
                "total_movies": len(self.recommender.movies_df),
            }
        except Exception as e:
            return {"error": f"Error getting movies: {e}"}

    def search_movies(self, query, limit=10):
        """Search for movies by title"""
        if self.recommender is None:
            return {"error": "Model not loaded. Please run main.py first."}

        try:
            query = query.lower()
            matching_movies = (
                self.recommender.movies_df[
                    self.recommender.movies_df["title"].str.lower().str.contains(query)
                ]
                .head(limit)[["movieId", "title", "genres"]]
                .to_dict("records")
            )

            return {
                "query": query,
                "movies": matching_movies,
                "count": len(matching_movies),
            }
        except Exception as e:
            return {"error": f"Error searching movies: {e}"}

    def get_model_stats(self):
        """Get statistics about the trained model"""
        if self.recommender is None:
            return {"error": "Model not loaded. Please run main.py first."}

        try:
            stats = {
                "total_users": len(self.recommender.user_mapping),
                "total_movies": len(self.recommender.item_mapping),
                "svd_components": self.recommender.n_components,
                "explained_variance_ratio": float(
                    self.recommender.svd.explained_variance_ratio_.sum()
                ),
            }
            self.monitor.record_model_performance(
                "explained_variance_ratio", stats["explained_variance_ratio"]
            )
            return stats
        except Exception as e:
            self.monitor.logger.log_error_with_context(e, {"action": "get_model_stats"})
            return {"error": f"Error getting model stats: {e}"}

    def ask_llm(self, query, model="phi", n_recommendations=10):
        """Get movie recommendations using natural language queries via LLM"""
        if self.recommender is None:
            return {"error": "Model not loaded. Please run main.py first."}
        # Force model to phi or gemma:2b only
        if model not in ["phi", "gemma:2b"]:
            print(f"[DEBUG] Invalid model '{model}' requested, defaulting to 'phi'.")
            model = "phi"
        print(f"[DEBUG] Using LLM model: {model}")
        try:
            start = time.time()
            # Create a simple, focused prompt
            prompt = f"""Suggest {n_recommendations} movies for: \"{query}\"\n\nRespond with ONLY movie titles, one per line. Example:\n1. Movie Title 1\n2. Movie Title 2\n\nYour recommendations:"""
            # Get LLM response
            response = ollama.chat(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            llm_response = response["message"]["content"]
            # Try to extract movie titles from LLM response
            movie_titles = self._extract_movie_titles(llm_response)
            # Get actual movie data for found titles
            recommended_movies = []
            for title in movie_titles[:n_recommendations]:
                movie_data = self.search_movies(title, 1)
                if movie_data.get("movies"):
                    recommended_movies.append(movie_data["movies"][0])
            # Fallback: if LLM didn't find enough movies, use search-based approach
            if len(recommended_movies) < 3:
                # Extract keywords from query for search
                keywords = self._extract_keywords(query)
                fallback_movies = []
                for keyword in keywords[:3]:  # Try top 3 keywords
                    search_result = self.search_movies(keyword, 5)
                    if search_result.get("movies"):
                        fallback_movies.extend(search_result["movies"])
                # Remove duplicates and add to recommendations
                seen_titles = {movie["title"] for movie in recommended_movies}
                for movie in fallback_movies:
                    if (
                        movie["title"] not in seen_titles
                        and len(recommended_movies) < n_recommendations
                    ):
                        recommended_movies.append(movie)
                        seen_titles.add(movie["title"])
            latency = time.time() - start
            self.monitor.record_llm_query(
                model=model,
                query=query,
                response_time=latency,
                num_results=len(recommended_movies),
                success=True,
            )
            return {
                "query": query,
                "llm_response": llm_response,
                "recommended_movies": recommended_movies,
                "count": len(recommended_movies),
                "model_used": model,
                "used_fallback": len(recommended_movies) < 3,
            }
        except Exception as e:
            self.monitor.record_llm_query(
                model=model,
                query=query,
                response_time=0.0,
                num_results=0,
                success=False,
            )
            self.monitor.logger.log_error_with_context(
                e, {"query": query, "action": "ask_llm"}
            )
            return {"error": f"Error processing LLM query: {e}"}

    def _extract_movie_titles(self, text):
        """Extract movie titles from LLM response"""
        lines = text.split("\n")
        titles = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering patterns (1., 1), etc.)
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)

            # Remove common prefixes
            cleaned = re.sub(
                r"^(movie|film|title):\s*", "", cleaned, flags=re.IGNORECASE
            )

            # Remove quotes if present
            cleaned = cleaned.strip("\"'")

            # Skip if too short or contains programming keywords
            if (
                len(cleaned) > 3
                and not cleaned.startswith("-")
                and not cleaned.startswith("*")
                and "programming" not in cleaned.lower()
                and "code" not in cleaned.lower()
                and "function" not in cleaned.lower()
                and "algorithm" not in cleaned.lower()
                and "database" not in cleaned.lower()
            ):
                titles.append(cleaned)

        return titles[:10]  # Return top 10 extracted titles

    def _extract_keywords(self, query):
        """Extract relevant keywords from a query for search"""
        # Common movie-related keywords
        movie_keywords = [
            "action",
            "comedy",
            "drama",
            "horror",
            "sci-fi",
            "thriller",
            "romance",
            "adventure",
            "fantasy",
            "mystery",
            "crime",
            "war",
            "western",
            "musical",
            "documentary",
            "animation",
            "family",
            "children",
            "teen",
            "adult",
            "romantic",
            "funny",
            "scary",
            "exciting",
            "sad",
            "happy",
            "love",
            "fighting",
            "space",
            "magic",
            "detective",
            "police",
            "soldier",
            "cowboy",
            "singing",
            "real",
            "cartoon",
            "kid",
            "young",
            "grown",
        ]

        # Extract keywords from query
        query_lower = query.lower()
        found_keywords = []

        # Look for genre keywords
        for keyword in movie_keywords:
            if keyword in query_lower:
                found_keywords.append(keyword)

        # Look for common movie titles in the query
        if not found_keywords:
            common_movies = [
                "inception",
                "shawshank",
                "titanic",
                "star wars",
                "matrix",
                "die hard",
                "jaws",
                "godfather",
                "pulp fiction",
                "forrest gump",
                "goodfellas",
                "fight club",
                "interstellar",
                "dark knight",
                "avatar",
                "toy story",
                "lion king",
                "frozen",
                "finding nemo",
            ]
            for movie in common_movies:
                if movie in query_lower:
                    found_keywords.append(movie)

        # Look for time periods
        time_keywords = [
            "80s",
            "90s",
            "2000s",
            "2010s",
            "2020s",
            "old",
            "new",
            "recent",
            "classic",
        ]
        for time_word in time_keywords:
            if time_word in query_lower:
                found_keywords.append(time_word)

        # If still no keywords, extract any meaningful words
        if not found_keywords:
            words = query_lower.split()
            meaningful_words = [
                word
                for word in words
                if len(word) > 3
                and word
                not in ["give", "me", "want", "like", "show", "find", "recommend"]
            ]
            found_keywords = meaningful_words[:3]  # Take top 3 meaningful words

        return found_keywords

    def explain_recommendations(self, user_id, model="deepseek-coder"):
        """Explain why certain movies were recommended for a user"""
        if self.recommender is None:
            return {"error": "Model not loaded. Please run main.py first."}

        try:
            # Get recommendations for the user
            recommendations = self.get_user_recommendations(user_id, 5)

            if "error" in recommendations:
                return recommendations

            # Create context for LLM
            movies_text = ", ".join(
                [rec["title"] for rec in recommendations["recommendations"]]
            )

            prompt = f"Briefly explain why these movies were recommended: {movies_text}"

            # Get LLM response
            response = ollama.chat(
                model=model, messages=[{"role": "user", "content": prompt}]
            )

            return {
                "user_id": user_id,
                "recommendations": recommendations["recommendations"],
                "explanation": response["message"]["content"],
                "model_used": model,
            }

        except Exception as e:
            return {"error": f"Error explaining recommendations: {e}"}


# Example usage functions
def example_usage():
    """Example of how to use the MovieRecommenderAPI"""
    api = MovieRecommenderAPI()

    # Get model statistics
    print("Model Statistics:")
    print(api.get_model_stats())

    # Get some available users
    print("\nAvailable Users:")
    print(api.get_available_users(5))

    # Get some available movies
    print("\nAvailable Movies:")
    print(api.get_available_movies(5))

    # Get recommendations for a user (if available)
    users = api.get_available_users(1)
    if "users" in users and users["users"]:
        user_id = users["users"][0]
        print(f"\nRecommendations for user {user_id}:")
        print(api.get_user_recommendations(user_id, 5))

    # Search for movies
    print("\nSearching for 'Star Wars':")
    print(api.search_movies("Star Wars", 3))

    # Test LLM integration
    print("\nLLM Query - 'Give me sci-fi thrillers from the 2010s':")
    print(
        api.ask_llm(
            "Give me sci-fi thrillers from the 2010s",
            model="deepseek-coder",
            n_recommendations=5,
        )
    )


if __name__ == "__main__":
    example_usage()
