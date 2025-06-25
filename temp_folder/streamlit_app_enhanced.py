import streamlit as st
import sys
import os
import hashlib
import pandas as pd
import sqlite3

# Add the api directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

from database import MovieDatabase
from api.recommender import MovieRecommenderAPI

# Helper for password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Initialize database and recommender
@st.cache_resource
def get_db():
    return MovieDatabase()

def get_recommender():
    return MovieRecommenderAPI()

db = get_db()
recommender = get_recommender()

# --- Authentication ---
def login_form():
    st.subheader("Sign In")
    username = st.text_input("Username", key="login_username_form")
    password = st.text_input("Password", type="password", key="login_password_form")
    if st.button("Sign In", key="login_button_form"):
        if not username or not password:
            st.error("Please enter both username and password.")
            return
        user = db.get_user_by_username(username)
        if user:
            if user['password_hash'] == hash_password(password):
                st.session_state['user'] = user
                st.success(f"Welcome, {username}! (User ID: {user['id']})")
                st.rerun()
            else:
                st.error("Invalid password.")
        else:
            st.error("User not found. Please sign up first.")

def signup_form():
    st.subheader("Sign Up")
    username = st.text_input("New Username", key="signup_username_form")
    email = st.text_input("Email", key="signup_email_form")
    password = st.text_input("Password", type="password", key="signup_password_form")
    if st.button("Create Account", key="signup_button_form"):
        if db.create_user(username, email, hash_password(password)):
            st.success("Account created! Please sign in.")
        else:
            st.error("Username or email already exists.")

def logout():
    if st.button("Logout", key="logout_button_form"):
        st.session_state.pop('user', None)
        st.success("Logged out.")
        st.rerun()

# --- Helper: Check if user has ratings or watchlist ---
def user_has_ratings_or_watchlist(user_id):
    ratings = db.get_user_ratings(user_id)
    watchlist = db.get_watchlist(user_id)
    return bool(ratings or watchlist)

# --- Movie Selection and Hybrid Recommendation ---
def movie_selection_and_recommend():
    st.subheader("Welcome! Let's get to know your movie preferences")
    st.write("Select at least 5 movies you've watched and enjoyed to get personalized recommendations:")
    top_movies = recommender.get_available_movies(50)
    if "error" in top_movies:
        st.error(f"Could not load movies: {top_movies['error']}")
        return False
    if not top_movies.get('movies'):
        st.warning("No movies available for selection. Please check your model and data.")
        return False
    selected_movies = []
    cols = st.columns(2)
    for i, movie in enumerate(top_movies['movies']):
        col_idx = i % 2
        with cols[col_idx]:
            if st.checkbox(f"**{movie['title']}** ({movie['genres']})", key=f"movie_select_{movie['movieId']}"):
                selected_movies.append({
                    'movieId': movie['movieId'],
                    'title': movie['title'],
                    'genres': movie['genres']
                })
    st.write(f"Selected {len(selected_movies)} movies")
    show_recs = False
    if st.button("Generate Recommendations", key="generate_recs_btn"):
        if len(selected_movies) >= 5:
            st.session_state['temp_user_movies'] = selected_movies
            st.session_state['show_recommendations'] = True
            show_recs = True
        else:
            st.error("Please select at least 5 movies to generate recommendations.")
    if st.session_state.get('show_recommendations', False) or show_recs:
        selected_movies = st.session_state.get('temp_user_movies', selected_movies)
        st.success(f"Based on your {len(selected_movies)} movie selections:")
        st.subheader("Movies You Selected:")
        for movie in selected_movies:
            st.write(f"• **{movie['title']}** ({movie['genres']})")
        st.subheader("Recommended Movies for You:")
        # --- Hybrid Recommendation Logic ---
        if 'user' in st.session_state:
            user = st.session_state['user']
            if user_has_ratings_or_watchlist(user['id']):
                st.info("Recommendations are based on your ratings and watchlist (collaborative filtering).")
                n = st.slider("How many recommendations?", 5, 20, 10, key="cf_n_recs_slider")
                recs = recommender.get_user_recommendations(user['id'], n)
                if "error" not in recs:
                    for i, rec in enumerate(recs['recommendations'], 1):
                        st.write(f"{i}. **{rec['title']}** ({rec['genres']}) - Score: {rec['similarity_score']:.3f}")
                        if st.button(f"Add to Watchlist", key=f"add_watchlist_cf_{rec['movieId']}"):
                            if db.add_to_watchlist(user['id'], rec['movieId']):
                                st.success("Added to watchlist!")
                else:
                    st.error(recs["error"])
            else:
                st.info("Recommendations are based on your selected movies (content-based). Sign in and rate movies or add to your watchlist for even better recommendations!")
                all_recommendations = []
                for movie in selected_movies:
                    similar = recommender.get_similar_movies(movie['movieId'], 5)
                    if "error" not in similar:
                        all_recommendations.extend(similar['similar_movies'])
                unique_recs = {}
                for rec in all_recommendations:
                    if rec['movieId'] not in unique_recs:
                        unique_recs[rec['movieId']] = rec
                    else:
                        if rec['similarity_score'] > unique_recs[rec['movieId']]['similarity_score']:
                            unique_recs[rec['movieId']] = rec
                sorted_recs = sorted(unique_recs.values(), key=lambda x: x['similarity_score'], reverse=True)
                if len(sorted_recs) > 0:
                    for i, rec in enumerate(sorted_recs[:15], 1):
                        st.write(f"{i}. **{rec['title']}** ({rec['genres']}) - Score: {rec['similarity_score']:.3f}")
                        if 'user' in st.session_state:
                            if st.button(f"Add to Watchlist", key=f"add_watchlist_cb_{rec['movieId']}"):
                                if db.add_to_watchlist(user['id'], rec['movieId']):
                                    st.success("Added to watchlist!")
                else:
                    st.info("No recommendations found. Please try different movie selections.")
            st.info("Want to save your ratings and watchlist? Sign up or sign in below!")
            with st.expander("Sign Up / Sign In"):
                tabs = st.tabs(["Sign In", "Sign Up"])
                with tabs[0]:
                    login_form()
                with tabs[1]:
                    signup_form()
            # --- AI Assistant Section ---
            st.markdown("---")
            st.header("🤖 AI Movie Assistant")
            st.markdown("Ask for movie recommendations in natural language!")
            available_models = ["phi", "gemma:2b"]
            selected_model = st.selectbox("Select AI Model:", available_models, index=0, key="ai_model_select")
            query = st.text_area(
                "Ask for movie recommendations:",
                placeholder="Examples:\n- Give me sci-fi thrillers from the 2010s\n- I want romantic comedies\n- Recommend action movies like Die Hard\n- Show me horror movies from the 80s",
                height=100,
                key="ai_query_area"
            )
            n_recommendations = st.slider("Number of recommendations:", 1, 20, 10, key="ai_n_recs_slider")
            if st.button("🤖 Ask AI Assistant", key="ai_ask_btn"):
                if query.strip():
                    with st.spinner("AI is processing your request..."):
                        result = recommender.ask_llm(query, model=selected_model, n_recommendations=n_recommendations)
                    if "error" not in result:
                        st.success(f"AI generated {result['count']} recommendations!")
                        st.markdown("### 🤖 AI Response")
                        st.write(result["llm_response"])
                        if result.get("used_fallback", False):
                            st.info("⚠️ AI couldn't find enough specific movies, so I used keyword search as a fallback.")
                        if result["recommended_movies"]:
                            # Filter out movies already rated or in watchlist
                            rated_movie_ids = set(r["movie_id"] for r in db.get_user_ratings(user["id"])) if user else set()
                            watchlist_movie_ids = set(db.get_watchlist(user["id"])) if user else set()
                            filtered_movies = [movie for movie in result["recommended_movies"] if movie["movieId"] not in rated_movie_ids and movie["movieId"] not in watchlist_movie_ids]
                            st.markdown("### 🎬 Recommended Movies")
                            for i, movie in enumerate(filtered_movies, 1):
                                with st.expander(f"{i}. {movie['title']}"):
                                    st.write(f"**Genres:** {movie['genres']}")
                                    st.write(f"**Movie ID:** {movie['movieId']}")
                                    if user:
                                        if st.button(f"Add to Watchlist", key=f"add_watchlist_ai_{movie['movieId']}"):
                                            if db.add_to_watchlist(user['id'], movie['movieId']):
                                                st.success("Added to watchlist!")
                        else:
                            st.info("No specific movies found in the database matching the AI suggestions.")
                            st.markdown("**💡 Tip:** Try using more specific genre terms like 'action', 'comedy', 'drama', or specific movie titles.")
                        st.info(f"Generated using {result['model_used']} model")
                    else:
                        st.error(result["error"])
                        st.markdown("**💡 Tip:** Try using different keywords or a different AI model.")
                else:
                    st.warning("Please enter a query first.")
            st.markdown("### 💡 Example Queries")
            st.markdown("""
            Try these example queries:
            - **"Give me sci-fi thrillers from the 2010s"**
            - **"I want romantic comedies with good ratings"**
            - **"Recommend action movies like Die Hard"**
            - **"Show me horror movies from the 80s"**
            - **"What are some good drama movies?"**
            - **"I like comedy and adventure movies"**
            """)
            return True
    return False

# --- Helper: Update SVD model after new rating ---
def update_svd_after_rating(user_id, movie_id, rating):
    # Get all ratings from the database
    all_ratings = []
    for uid in db.get_all_user_ids():
        for r in db.get_user_ratings(uid):
            all_ratings.append({'userId': uid, 'movieId': r['movie_id'], 'rating': r['rating']})
    ratings_df = pd.DataFrame(all_ratings)
    movies_df = recommender.recommender.movies_df
    user_item_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
    n_users, n_movies = user_item_matrix.shape
    n_components = max(2, min(50, n_users - 1, n_movies - 1))
    recommender.recommender = recommender.recommender.__class__(user_item_matrix, movies_df, n_components=n_components)
    recommender.recommender.fit()

# --- Add get_all_user_ids to db if not present ---
if not hasattr(db, 'get_all_user_ids'):
    def get_all_user_ids():
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users")
        ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return ids
    db.get_all_user_ids = get_all_user_ids

# --- Helper: Remove from watchlist if rated ---
def remove_from_watchlist_if_rated(user_id, movie_id):
    user_ratings = db.get_user_ratings(user_id)
    rated_movie_ids = {r['movie_id'] for r in user_ratings}
    if movie_id in db.get_watchlist(user_id) and movie_id in rated_movie_ids:
        db.remove_from_watchlist(user_id, movie_id)

# --- Add remove_from_watchlist to db if not present ---
if not hasattr(db, 'remove_from_watchlist'):
    def remove_from_watchlist(user_id, movie_id):
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM watchlist WHERE user_id = ? AND movie_id = ?", (user_id, movie_id))
        conn.commit()
        conn.close()
        return True
    db.remove_from_watchlist = remove_from_watchlist

# --- Main App ---
def main():
    st.set_page_config(page_title="Movie Recommender Enhanced", page_icon="🎬", layout="wide")
    st.title("🎬 Movie Recommender System (Enhanced)")
    # Always define user safely
    if 'user' in st.session_state:
        user = st.session_state['user']
    else:
        user = None
    # If not logged in, show welcome with sign in/up
    if not user:
        st.header("Welcome to the Movie Recommender!")
        st.markdown("""
        - Sign in to get personalized recommendations
        - New here? Sign up to start rating and get suggestions
        """)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sign In", key="welcome_signin_btn"):
                st.session_state['show_login'] = True
        with col2:
            if st.button("Sign Up", key="welcome_signup_btn"):
                st.session_state['show_signup'] = True
        if st.session_state.get('show_login', False):
            login_form()
        if st.session_state.get('show_signup', False):
            st.subheader("Sign Up")
            signup_form()
            return
    # If logged in, check if user has ratings, else show cold start
    user_ratings = db.get_user_ratings(user['id']) if user else []
    if user and (not user_ratings or len(user_ratings) < 2):
        st.info("Let's get to know your movie preferences! Select at least 2 movies you've watched and enjoyed to get started.")
        movie_selection_and_recommend()
        return
    if user:
        st.sidebar.write(f"Logged in as: {user['username']}")
        logout()
    page = st.sidebar.radio("Navigation", [
        "Home", "Recommendations", "Rate Movies", "Watchlist", "Search Movies", "Profile", "AI Assistant"
    ])
    if page == "Home":
        st.header("Welcome!")
        st.markdown("""
        - Get personalized recommendations
        - Rate movies and build your watchlist
        - Search and explore the movie database
        """)
        stats = recommender.get_model_stats()
        if "error" not in stats:
            st.metric("Total Users", stats["total_users"])
            st.metric("Total Movies", stats["total_movies"])
        else:
            st.error(stats["error"])
    elif page == "Recommendations":
        st.header("🎬 Your Personalized Recommendations")
        if user:
            user_ratings = db.get_user_ratings(user['id'])
            if len(user_ratings) >= 2:
                n = st.slider("How many recommendations?", 5, 20, 10, key="recs_n_slider")
                recs = recommender.get_user_recommendations(user['id'], n)
                if "error" not in recs:
                    st.success(f"Here are your top {n} recommendations:")
                    for i, rec in enumerate(recs['recommendations'], 1):
                        st.write(f"{i}. **{rec['title']}** ({rec['genres']}) - Score: {rec['similarity_score']:.3f}")
                        if st.button(f"Add to Watchlist", key=f"add_watchlist_recs_{rec['movieId']}"):
                            if db.add_to_watchlist(user['id'], rec['movieId']):
                                st.success("Added to watchlist!")
                else:
                    st.error(recs["error"])
            else:
                st.info("Rate at least 2 movies to get personalized recommendations!")
                st.markdown("Go to the 'Rate Movies' page to rate more movies.")
        else:
            st.info("Please log in to get personalized recommendations.")
    elif page == "Rate Movies":
        st.header("Rate Movies")
        search = st.text_input("Search for a movie to rate:", key="rate_search")
        if search:
            results = recommender.search_movies(search, 10)
            if "error" not in results:
                for movie in results['movies']:
                    st.write(f"**{movie['title']}** ({movie['genres']})")
                    rating = st.slider(f"Your rating for {movie['title']}", 0.0, 5.0, 3.0, 0.5, key=f"rate_{movie['movieId']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Submit Rating for {movie['title']}", key=f"submit_{movie['movieId']}"):
                            if user:
                                if db.add_rating(user['id'], movie['movieId'], rating):
                                    st.success("Rating submitted!")
                                    update_svd_after_rating(user['id'], movie['movieId'], rating)
                                    remove_from_watchlist_if_rated(user['id'], movie['movieId'])
                            else:
                                st.info("Please log in to save ratings.")
                    with col2:
                        if user:
                            if st.button(f"Add to Watchlist", key=f"add_watchlist_rate_{movie['movieId']}"):
                                if db.add_to_watchlist(user['id'], movie['movieId']):
                                    st.success("Added to watchlist!")
            else:
                st.error(results["error"])
    elif page == "Watchlist":
        st.header("Your Watchlist")
        if user:
            watchlist = db.get_watchlist(user['id'])
            rated_movie_ids = {r['movie_id'] for r in db.get_user_ratings(user['id'])}
            filtered_watchlist = [mid for mid in watchlist if mid not in rated_movie_ids]
            if filtered_watchlist:
                for movie_id in filtered_watchlist:
                    info = recommender.get_movie_info(movie_id)
                    if "error" not in info:
                        st.write(f"**{info['title']}** ({info['genres']})")
                    else:
                        st.write(f"Movie ID {movie_id}")
            else:
                st.info("Your watchlist is empty.")
        else:
            st.info("Please log in to view your watchlist.")
    elif page == "Search Movies":
        st.header("Search Movies")
        query = st.text_input("Enter movie title:", key="search_movies")
        if query:
            results = recommender.search_movies(query, 20)
            if "error" not in results:
                for movie in results['movies']:
                    st.write(f"**{movie['title']}** ({movie['genres']})")
                    if user:
                        if st.button(f"Add to Watchlist", key=f"add_watchlist_search_{movie['movieId']}"):
                            if db.add_to_watchlist(user['id'], movie['movieId']):
                                st.success("Added to watchlist!")
            else:
                st.error(results["error"])
    elif page == "Profile":
        st.header("Profile & Preferences")
        if user:
            prefs = db.get_user_preferences(user['id'])
            genres = st.text_input("Favorite genres (comma separated)", value=", ".join(prefs['favorite_genres']) if prefs else "", key="profile_genres")
            decades = st.text_input("Preferred decades (comma separated)", value=", ".join(prefs['preferred_decades']) if prefs else "", key="profile_decades")
            max_rating = st.slider("Max rating", 1.0, 5.0, prefs['max_rating'] if prefs else 5.0, 0.5, key="profile_max_rating")
            if st.button("Save Preferences", key="profile_save_button"):
                fav_genres = [g.strip() for g in genres.split(",") if g.strip()]
                pref_decades = [d.strip() for d in decades.split(",") if d.strip()]
                if db.save_user_preferences(user['id'], fav_genres, pref_decades, max_rating):
                    st.success("Preferences saved!")
                else:
                    st.error("Failed to save preferences.")
            # Watched/Rated Movies Section
            st.subheader("🎬 Watched/Rated Movies")
            user_ratings = db.get_user_ratings(user['id'])
            if user_ratings:
                for r in user_ratings:
                    info = recommender.get_movie_info(r['movie_id'])
                    if "error" not in info:
                        st.write(f"**{info['title']}** ({info['genres']}) - Your Rating: {r['rating']}")
                    else:
                        st.write(f"Movie ID {r['movie_id']} - Your Rating: {r['rating']}")
            else:
                st.info("You haven't rated any movies yet.")
        else:
            st.info("Please log in to manage your profile.")
    elif page == "AI Assistant":
        st.header("🤖 AI Movie Assistant")
        st.markdown("Ask for movie recommendations in natural language!")
        prompt = st.text_input("What kind of movies are you looking for?", key="ai_prompt")
        n = st.slider("How many results?", 5, 20, 10, key="ai_n_slider")
        if st.button("Get AI Recommendations", key="ai_get_btn") and prompt:
            with st.spinner("Thinking..."):
                ai_results = recommender.ask_llm(prompt, n_recommendations=30)
                print(f"[DEBUG] LLM raw results for prompt '{prompt}': {ai_results}")
                if "error" in ai_results:
                    st.error(f"AI error: {ai_results['error']}")
                else:
                    llm_response = ai_results.get('llm_response', None)
                    if llm_response:
                        st.caption(f"💬 LLM says: {llm_response}")
                    candidates = ai_results.get('recommended_movies', [])
                    personalized_candidates = []
                    tag = ''
                    if user:
                        user_ratings = db.get_user_ratings(user['id'])
                        if len(user_ratings) >= 2:
                            svd_recs = recommender.get_user_recommendations(user['id'], n_recommendations=100)
                            svd_ids = [rec['movieId'] for rec in svd_recs.get('recommendations', [])]
                            personalized_candidates = sorted(
                                [c for c in candidates if c['movieId'] in svd_ids],
                                key=lambda x: svd_ids.index(x['movieId']) if x['movieId'] in svd_ids else 9999
                            )
                            seen = {r['movieId'] for r in personalized_candidates}
                            personalized_candidates += [c for c in candidates if c['movieId'] not in seen]
                            # Fallback: fill up to n with SVD recs if needed
                            if len(personalized_candidates) < n:
                                for rec in svd_recs.get('recommendations', []):
                                    if rec['movieId'] not in seen:
                                        personalized_candidates.append({
                                            'movieId': rec['movieId'],
                                            'title': rec['title'],
                                            'genres': rec['genres']
                                        })
                                        seen.add(rec['movieId'])
                                    if len(personalized_candidates) >= n:
                                        break
                            personalized_candidates = personalized_candidates[:n]
                            if personalized_candidates and any(c['movieId'] in svd_ids for c in personalized_candidates):
                                tag = ':blue-background[LLM + SVD Personalized Results]'
                                st.markdown(tag)
                                st.info("Results are personalized using your ratings and SVD model! (with fallback if needed)")
                            else:
                                tag = ':orange-background[LLM Results Only]'
                                st.markdown(tag)
                                st.warning("No personalized matches found. Showing general AI results.")
                                personalized_candidates = candidates[:n]
                        else:
                            tag = ':orange-background[LLM Results Only]'
                            st.markdown(tag)
                            personalized_candidates = candidates[:n]
                            st.info("Rate more movies for better personalization!")
                    else:
                        tag = ':orange-background[LLM Results Only]'
                        st.markdown(tag)
                        personalized_candidates = candidates[:n]
                        st.info("Sign in and rate movies for personalized results!")
                    # Fallback: if still fewer than n, fill with content-based (similar movies to top LLM result)
                    if len(personalized_candidates) < n and candidates:
                        # Try to get similar movies to the first LLM result
                        first_movie_id = candidates[0]['movieId']
                        similar = recommender.get_similar_movies(first_movie_id, n_similar=n)
                        if isinstance(similar, dict) and 'similar_movies' in similar:
                            for rec in similar['similar_movies']:
                                if rec['movieId'] not in {c['movieId'] for c in personalized_candidates}:
                                    personalized_candidates.append(rec)
                                if len(personalized_candidates) >= n:
                                    break
                    personalized_candidates = personalized_candidates[:n]
                    if personalized_candidates:
                        # Filter out movies already rated or in watchlist
                        rated_movie_ids = set(r["movie_id"] for r in db.get_user_ratings(user["id"])) if user else set()
                        watchlist_movie_ids = set(db.get_watchlist(user["id"])) if user else set()
                        filtered_movies = [movie for movie in personalized_candidates if movie["movieId"] not in rated_movie_ids and movie["movieId"] not in watchlist_movie_ids]
                        st.markdown("### 🎬 Recommended Movies")
                        for i, movie in enumerate(filtered_movies, 1):
                            with st.expander(f"{i}. {movie['title']}"):
                                st.write(f"**Genres:** {movie['genres']}")
                                st.write(f"**Movie ID:** {movie['movieId']}")
                                if user:
                                    if st.button(f"Add to Watchlist", key=f"add_watchlist_ai_{movie['movieId']}"):
                                        if db.add_to_watchlist(user['id'], movie['movieId']):
                                            st.success("Added to watchlist!")
                    else:
                        st.info("No results found. Try a different prompt!")

if __name__ == "__main__":
    main()
