import streamlit as st
import sys
import os
import hashlib
import pandas as pd
import sqlite3
import time
from utils.monitoring import get_performance_monitor
import numpy as np
import random

# Add the api directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "api"))

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
            if user["password_hash"] == hash_password(password):
                # --- A/B group assignment ---
                if "ab_group" not in user or user["ab_group"] not in ["A", "B"]:
                    user["ab_group"] = random.choice(["A", "B"])
                st.session_state["user"] = user
                st.session_state["authenticated"] = True
                st.success(
                    f"Welcome, {username}! (User ID: {user['id']}) [Group {user['ab_group']}]"
                )
                monitor.record_user_action(
                    user["id"],
                    "login",
                    {"username": username, "ab_group": user["ab_group"]},
                )
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
            # Assign A/B group on signup
            ab_group = random.choice(["A", "B"])
            user = db.get_user_by_username(username)
            if user:
                user["ab_group"] = ab_group
                st.session_state["user"] = user
                st.session_state["authenticated"] = True
                st.success(
                    f"Account created! You are in group {ab_group}. Please sign in."
                )
                monitor.record_user_action(
                    -1,
                    "signup",
                    {"username": username, "email": email, "ab_group": ab_group},
                )
                st.rerun()
            else:
                st.success("Account created! Please sign in.")
        else:
            st.error("Username or email already exists.")


def logout():
    if st.button("Logout", key="logout_button_form"):
        if "user" in st.session_state:
            user = st.session_state["user"]
            monitor.record_user_action(
                user["id"], "logout", {"ab_group": user.get("ab_group", "Unknown")}
            )
        st.session_state.pop("user", None)
        st.session_state.pop("authenticated", None)
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
    st.write(
        "Select at least 5 movies you've watched and enjoyed to get personalized recommendations:"
    )
    top_movies = recommender.get_available_movies(50)
    if "error" in top_movies:
        st.error(f"Could not load movies: {top_movies['error']}")
        return False
    if not top_movies.get("movies"):
        st.warning(
            "No movies available for selection. Please check your model and data."
        )
        return False
    selected_movies = []
    cols = st.columns(2)
    for i, movie in enumerate(top_movies["movies"]):
        col_idx = i % 2
        with cols[col_idx]:
            if st.checkbox(
                f"**{movie['title']}** ({movie['genres']})",
                key=f"movie_select_{movie['movieId']}",
            ):
                selected_movies.append(
                    {
                        "movieId": movie["movieId"],
                        "title": movie["title"],
                        "genres": movie["genres"],
                    }
                )
    st.write(f"Selected {len(selected_movies)} movies")
    show_recs = False
    if st.button("Generate Recommendations", key="generate_recs_btn"):
        if len(selected_movies) >= 5:
            st.session_state["temp_user_movies"] = selected_movies
            st.session_state["show_recommendations"] = True
            show_recs = True
            monitor.record_user_action(
                st.session_state.get("user", {}).get("id", -1),
                "generate_recommendations",
                {
                    "num_selected": len(selected_movies),
                    "ab_group": st.session_state.get("user", {}).get("ab_group", "NA"),
                },
            )
        else:
            st.error("Please select at least 5 movies to generate recommendations.")
    if st.session_state.get("show_recommendations", False) or show_recs:
        selected_movies = st.session_state.get("temp_user_movies", selected_movies)
        st.success(f"Based on your {len(selected_movies)} movie selections:")
        st.subheader("Movies You Selected:")
        for movie in selected_movies:
            st.write(f"\u2022 **{movie['title']}** ({movie['genres']})")
        st.subheader("Recommended Movies for You:")
        # --- A/B Recommendation Logic ---
        if "user" in st.session_state:
            user = st.session_state["user"]
            ab_group = user.get("ab_group", "A")
            st.info(
                f"You are in group {ab_group}. Recommendations are based on {('collaborative filtering' if ab_group == 'A' else 'content-based filtering')}."
            )
            n = st.slider(
                "How many recommendations?", 5, 20, 10, key="ab_n_recs_slider"
            )
            if ab_group == "A":
                recs = recommender.get_user_recommendations(user["id"], n)
                algo_label = "collaborative"
            else:
                # Aggregate content-based recommendations from selected movies
                all_recommendations = []
                for movie in selected_movies:
                    similar = recommender.get_similar_movies(movie["movieId"], 5)
                    if "error" not in similar:
                        all_recommendations.extend(similar["similar_movies"])
                unique_recs = {}
                for rec in all_recommendations:
                    if rec["movieId"] not in unique_recs:
                        unique_recs[rec["movieId"]] = rec
                    else:
                        if (
                            rec["similarity_score"]
                            > unique_recs[rec["movieId"]]["similarity_score"]
                        ):
                            unique_recs[rec["movieId"]] = rec
                sorted_recs = sorted(
                    unique_recs.values(),
                    key=lambda x: x["similarity_score"],
                    reverse=True,
                )
                recs = {
                    "recommendations": sorted_recs[:n],
                    "count": len(sorted_recs[:n]),
                    "latency": None,
                }
                algo_label = "content-based"
            if "error" not in recs:
                monitor.record_user_action(
                    user["id"],
                    "view_recommendations",
                    {
                        "num_recommendations": recs["count"],
                        "latency": recs.get("latency", None),
                        "ab_group": ab_group,
                        "algorithm": algo_label,
                    },
                )
                for i, rec in enumerate(recs["recommendations"], 1):
                    st.write(
                        f"{i}. **{rec['title']}** ({rec['genres']}) - Score: {rec.get('similarity_score', 0):.3f}"
                    )
                    if st.button(
                        f"Add to Watchlist", key=f"add_watchlist_ab_{rec['movieId']}"
                    ):
                        if db.add_to_watchlist(user["id"], rec["movieId"]):
                            st.success("Added to watchlist!")
                            monitor.record_user_action(
                                user["id"],
                                "add_to_watchlist",
                                {
                                    "movie_id": rec["movieId"],
                                    "ab_group": ab_group,
                                    "algorithm": algo_label,
                                },
                            )
            else:
                st.error(recs["error"])
        else:
            st.info(
                "Sign in to get personalized recommendations and participate in A/B testing!"
            )
    # --- AI Assistant Section: Always show for all users, even before 5 movies are selected ---
    st.markdown("---")
    st.header("🤖 AI Movie Assistant")
    st.markdown("Ask for movie recommendations in natural language!")
    available_models = ["phi", "gemma:2b"]
    selected_model = st.selectbox(
        "Select AI Model:", available_models, index=0, key="ai_model_select"
    )
    query = st.text_area(
        "Ask for movie recommendations:",
        placeholder="Examples:\n- Give me sci-fi thrillers from the 2010s\n- I want romantic comedies\n- Recommend action movies like Die Hard\n- Show me horror movies from the 80s",
        height=100,
        key="ai_query_area",
    )
    n_recommendations = st.slider(
        "Number of recommendations:", 1, 20, 10, key="ai_n_recs_slider"
    )
    if st.button("🤖 Ask AI Assistant", key="ai_ask_btn"):
        if query.strip():
            with st.spinner("AI is processing your request..."):
                result = recommender.ask_llm(
                    query, model=selected_model, n_recommendations=n_recommendations
                )
            if "error" not in result:
                st.success(f"AI generated {result['count']} recommendations!")
                monitor.record_user_action(
                    st.session_state.get("user", {}).get("id", -1),
                    "ai_assistant_query",
                    {
                        "query": query,
                        "model": selected_model,
                        "num_results": result["count"],
                        "ab_group": st.session_state.get("user", {}).get(
                            "ab_group", "Unknown"
                        ),
                    },
                )
                st.markdown("### 🤖 AI Response")
                st.write(result["llm_response"])
                if result.get("used_fallback", False):
                    st.info(
                        "⚠️ AI couldn't find enough specific movies, so I used keyword search as a fallback."
                    )
                if result["recommended_movies"]:
                    user = st.session_state.get("user", None)
                    rated_movie_ids = (
                        set(r["movie_id"] for r in db.get_user_ratings(user["id"]))
                        if user
                        else set()
                    )
                    watchlist_movie_ids = (
                        set(db.get_watchlist(user["id"])) if user else set()
                    )
                    filtered_movies = [
                        movie
                        for movie in result["recommended_movies"]
                        if movie["movieId"] not in rated_movie_ids
                        and movie["movieId"] not in watchlist_movie_ids
                    ]
                    st.markdown("### 🎬 Recommended Movies")
                    for i, movie in enumerate(filtered_movies, 1):
                        with st.expander(f"{i}. {movie['title']}"):
                            st.write(f"**Genres:** {movie['genres']}")
                            st.write(f"**Movie ID:** {movie['movieId']}")
                            if user:
                                if st.button(
                                    f"Add to Watchlist",
                                    key=f"add_watchlist_ai_{movie['movieId']}",
                                ):
                                    if db.add_to_watchlist(
                                        user["id"], movie["movieId"]
                                    ):
                                        st.success("Added to watchlist!")
                                        monitor.record_user_action(
                                            user["id"],
                                            "add_to_watchlist",
                                            {
                                                "movie_id": movie["movieId"],
                                                "ab_group": user.get(
                                                    "ab_group", "Unknown"
                                                ),
                                            },
                                        )
                else:
                    st.info(
                        "No specific movies found in the database matching the AI suggestions."
                    )
                    st.markdown(
                        "**💡 Tip:** Try using more specific genre terms like 'action', 'comedy', 'drama', or specific movie titles."
                    )
                st.info(f"Generated using {result['model_used']} model")
            else:
                st.error(result["error"])
                st.markdown(
                    "**💡 Tip:** Try using different keywords or a different AI model."
                )
        else:
            st.warning("Please enter a query first.")
    st.markdown("### 💡 Example Queries")
    st.markdown(
        """
    Try these example queries:
    - **"Give me sci-fi thrillers from the 2010s"**
    - **"I want romantic comedies with good ratings"**
    - **"Recommend action movies like Die Hard"**
    - **"Show me horror movies from the 80s"**
    - **"What are some good drama movies?"**
    - **"I like comedy and adventure movies"**
    """
    )
    return True


# --- Helper: Update SVD model after new rating ---
def update_svd_after_rating(user_id, movie_id, rating):
    # Get all ratings from the database
    all_ratings = []
    for uid in db.get_all_user_ids():
        for r in db.get_user_ratings(uid):
            all_ratings.append(
                {"userId": uid, "movieId": r["movie_id"], "rating": r["rating"]}
            )
    ratings_df = pd.DataFrame(all_ratings)
    movies_df = recommender.recommender.movies_df
    user_item_matrix = ratings_df.pivot_table(
        index="userId", columns="movieId", values="rating", fill_value=0
    )
    n_users, n_movies = user_item_matrix.shape
    n_components = max(2, min(50, n_users - 1, n_movies - 1))
    recommender.recommender = recommender.recommender.__class__(
        user_item_matrix, movies_df, n_components=n_components
    )
    recommender.recommender.fit()


# --- Add get_all_user_ids to db if not present ---
if not hasattr(db, "get_all_user_ids"):

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
    rated_movie_ids = {r["movie_id"] for r in user_ratings}
    if movie_id in db.get_watchlist(user_id) and movie_id in rated_movie_ids:
        db.remove_from_watchlist(user_id, movie_id)


# --- Add remove_from_watchlist to db if not present ---
if not hasattr(db, "remove_from_watchlist"):

    def remove_from_watchlist(user_id, movie_id):
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM watchlist WHERE user_id = ? AND movie_id = ?",
            (user_id, movie_id),
        )
        conn.commit()
        conn.close()
        return True

    db.remove_from_watchlist = remove_from_watchlist


# --- Main App ---
def main():
    st.set_page_config(
        page_title="Movie Recommender", layout="wide", initial_sidebar_state="auto"
    )
    if "session_start" not in st.session_state:
        st.session_state["session_start"] = time.time()
    # --- Main App Logic ---
    sidebar_options = [
        "🎬 Movie Recommendations",
        "⭐ Your Ratings",
        "🍿 Your Watchlist",
        "🔍 Search Movies",
        "📊 Metrics Dashboard",
    ]
    if not st.session_state.get("authenticated"):
        st.title("🎬 Movie Recommender System")
        st.write("Sign in or sign up to get started.")
        tabs = st.tabs(["Sign In", "Sign Up"])
        with tabs[0]:
            login_form()
        with tabs[1]:
            signup_form()
    else:
        user = st.session_state["user"]
        with st.sidebar:
            st.header(f"👋 Welcome, {user['username']}")
            st.write(f"User ID: {user['id']}")
            # Debug: Show A/B group
            if "ab_group" in user:
                st.write(
                    f"🔬 A/B Group: {user['ab_group']} ({'Collaborative' if user['ab_group'] == 'A' else 'Content-Based'} Filtering)"
                )
            else:
                st.write("🔬 A/B Group: Not assigned")
            st.markdown("---")
            app_mode = st.radio(
                "Choose your mode:", sidebar_options, key="app_mode_radio"
            )
            st.markdown("---")
            logout()
            st.markdown("---")
            st.info(
                "This app uses collaborative and content-based filtering, plus an AI assistant for recommendations."
            )
        # --- Main Content ---
        if app_mode == "🎬 Movie Recommendations":
            st.title("🎬 Movie Recommendations")
            movie_selection_and_recommend()
        elif app_mode == "⭐ Your Ratings":
            st.title("⭐ Your Ratings")
            ratings = db.get_user_ratings(user["id"])
            if ratings:
                st.write(f"You have rated {len(ratings)} movies:")
                for r in ratings:
                    movie = db.get_movie_by_id(r["movie_id"])
                    title = movie["title"] if movie else f"Movie ID {r['movie_id']}"
                    st.write(f"- {title}: {r['rating']} stars")
                    monitor.record_user_action(
                        user["id"],
                        "view_rating",
                        {
                            "movie_id": r["movie_id"],
                            "rating": r["rating"],
                            "ab_group": user.get("ab_group", "Unknown"),
                        },
                    )
            else:
                st.info("You haven't rated any movies yet.")
        elif app_mode == "🍿 Your Watchlist":
            st.title("🍿 Your Watchlist")
            watchlist = db.get_watchlist(user["id"])
            if watchlist:
                st.write(f"You have {len(watchlist)} movies in your watchlist:")
                for movie_id in watchlist:
                    movie_details = db.get_movie_by_id(movie_id)
                    title = (
                        movie_details["title"]
                        if movie_details
                        else f"Movie ID {movie_id}"
                    )
                    st.write(f"- {title}")
                    monitor.record_user_action(
                        user["id"],
                        "view_watchlist",
                        {
                            "movie_id": movie_id,
                            "ab_group": user.get("ab_group", "Unknown"),
                        },
                    )
            else:
                st.info("Your watchlist is empty.")
        elif app_mode == "🔍 Search Movies":
            st.title("🔍 Search Movies")
            search_query = st.text_input("Search for a movie by title:")
            if search_query:
                search_results = db.search_movies(search_query)
                if search_results:
                    for movie in search_results:
                        st.write(f"- {movie['title']} ({movie['genres']})")
                        monitor.record_user_action(
                            user["id"],
                            "search_movie",
                            {
                                "query": search_query,
                                "movie_id": movie["movieId"],
                                "ab_group": user.get("ab_group", "Unknown"),
                            },
                        )
                else:
                    st.info("No movies found.")
        elif app_mode == "📊 Metrics Dashboard":
            st.title("📊 Metrics Dashboard")
            st.write("Below are key metrics and system stats for this recommender app.")

            # Add clear metrics button
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🗑️ Clear Old Metrics", key="clear_metrics_btn"):
                    monitor.clear_metrics()
                    st.success(
                        "Old metrics cleared! New actions will be properly labeled."
                    )
                    st.rerun()

            perf_summary = monitor.get_performance_summary()
            # --- System Metrics ---
            st.subheader("System Metrics")
            sys_metrics = perf_summary.get("system", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                cpu = sys_metrics.get("system.cpu_percent", {})
                st.metric("CPU Usage (%)", cpu.get("latest", "N/A"))
                if cpu:
                    st.line_chart(
                        np.array(
                            [cpu.get("min", 0), cpu.get("mean", 0), cpu.get("max", 0)]
                        ),
                        height=100,
                    )
            with col2:
                mem = sys_metrics.get("system.memory_percent", {})
                st.metric("Memory Usage (%)", mem.get("latest", "N/A"))
                if mem:
                    st.line_chart(
                        np.array(
                            [mem.get("min", 0), mem.get("mean", 0), mem.get("max", 0)]
                        ),
                        height=100,
                    )
            with col3:
                disk = sys_metrics.get("system.disk_percent", {})
                st.metric("Disk Usage (%)", disk.get("latest", "N/A"))
                if disk:
                    st.line_chart(
                        np.array(
                            [
                                disk.get("min", 0),
                                disk.get("mean", 0),
                                disk.get("max", 0),
                            ]
                        ),
                        height=100,
                    )
            # --- Recommendation Metrics ---
            st.subheader("Recommendation Metrics")
            rec_metrics = perf_summary.get("recommendations", {})
            if rec_metrics:
                # Show metric cards for each recommendation metric
                rec_cols = st.columns(len(rec_metrics))
                for idx, (metric, values) in enumerate(rec_metrics.items()):
                    with rec_cols[idx]:
                        st.metric(
                            metric.replace("_", " ").title(),
                            round(values.get("latest", 0), 2),
                        )
                        if values.get("count", 0) > 1:
                            st.line_chart(
                                [
                                    values.get("min", 0),
                                    values.get("mean", 0),
                                    values.get("max", 0),
                                    values.get("latest", 0),
                                ],
                                height=100,
                            )
            else:
                st.info("No recommendation metrics yet. Generate some recommendations!")
            # --- LLM (AI Assistant) Metrics ---
            st.subheader("LLM (AI Assistant) Metrics")
            llm_metrics = perf_summary.get("llm", {})
            if llm_metrics:
                llm_cols = st.columns(len(llm_metrics))
                for idx, (metric, values) in enumerate(llm_metrics.items()):
                    with llm_cols[idx]:
                        st.metric(
                            metric.replace("_", " ").title(),
                            round(values.get("latest", 0), 2),
                        )
                        if values.get("count", 0) > 1:
                            st.line_chart(
                                [
                                    values.get("min", 0),
                                    values.get("mean", 0),
                                    values.get("max", 0),
                                    values.get("latest", 0),
                                ],
                                height=100,
                            )
            else:
                st.info(
                    "No AI assistant metrics yet. Use the AI assistant to see metrics here!"
                )
            # --- User Actions ---
            st.subheader("User Actions (A/B Groups)")
            user_actions = perf_summary.get("user_actions", {})
            if user_actions:
                ua_rows = []
                for key, data in user_actions.items():
                    # Parse key format: "action_ab_group"
                    if "_" in key:
                        action = data.get("action", key.split("_")[0])
                        ab_group = data.get("ab_group", key.split("_")[1])
                    else:
                        action = data.get("action", key)
                        ab_group = data.get("ab_group", "All")
                    ua_rows.append(
                        {
                            "Action": action.replace("_", " ").title(),
                            "Count": data["count"],
                            "Unique Users": data.get("unique_users", 0),
                            "Last Performed": (
                                data["last_timestamp"][:19]
                                if data["last_timestamp"]
                                else "Never"
                            ),
                            "A/B Group": ab_group,
                        }
                    )
                ua_df = pd.DataFrame(ua_rows)
                st.dataframe(ua_df, use_container_width=True)
            else:
                st.info("No user actions recorded yet.")
            # --- Model Performance ---
            st.subheader("Model Performance (A/B Groups)")

            # Add button to trigger model evaluation
            if st.button("🔍 Evaluate Model Performance", key="eval_model_btn"):
                with st.spinner("Evaluating model performance..."):
                    try:
                        model_stats = recommender.get_model_stats()
                        if "error" not in model_stats:
                            st.success("Model evaluation completed!")
                            # Display model stats
                            st.write("**Model Statistics:**")
                            for key, value in model_stats.items():
                                if key != "error":
                                    st.write(f"- {key}: {value}")
                        else:
                            st.error(f"Model evaluation failed: {model_stats['error']}")
                    except Exception as e:
                        st.error(f"Error during model evaluation: {e}")

            model_perf = perf_summary.get("model_performance", {})
            if model_perf:
                rows = []
                for metric, algos in model_perf.items():
                    for algo, vals in algos.items():
                        ab_group = algo  # Here, algo is the ab_group label
                        rows.append(
                            {
                                "Metric": metric,
                                "A/B Group": ab_group,
                                "Latest": round(vals["latest"], 4),
                                "Min": round(vals["min"], 4),
                                "Max": round(vals["max"], 4),
                                "Mean": round(vals["mean"], 4),
                                "Count": vals["count"],
                            }
                        )
                mp_df = pd.DataFrame(rows)
                st.dataframe(mp_df, use_container_width=True)
            else:
                st.info(
                    "No model performance metrics yet. Click the 'Evaluate Model Performance' button above to generate metrics."
                )

            # --- A/B Testing Analysis ---
            st.subheader("🔬 A/B Testing Analysis")

            # Calculate A/B testing statistics
            user_actions = perf_summary.get("user_actions", {})
            if user_actions:
                # Group actions by A/B group
                group_a_actions = {}
                group_b_actions = {}
                group_a_users = set()
                group_b_users = set()

                for key, data in user_actions.items():
                    action = data.get(
                        "action", key.split("_")[0] if "_" in key else key
                    )
                    ab_group = data.get(
                        "ab_group", key.split("_")[1] if "_" in key else "Unknown"
                    )
                    unique_users = data.get("unique_users", 0)

                    if ab_group == "A":
                        if action not in group_a_actions:
                            group_a_actions[action] = 0
                        group_a_actions[action] += data["count"]
                        group_a_users.add(unique_users)
                    elif ab_group == "B":
                        if action not in group_b_actions:
                            group_b_actions[action] = 0
                        group_b_actions[action] += data["count"]
                        group_b_users.add(unique_users)

                # Display comparison
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📊 Group A (Collaborative Filtering)**")
                    if group_a_actions:
                        st.metric("Total Actions", sum(group_a_actions.values()))
                        st.metric("Unique Users", len(group_a_users))
                        st.markdown("**Actions:**")
                        for action, count in group_a_actions.items():
                            st.metric(f"{action.replace('_', ' ').title()}", count)
                    else:
                        st.info("No Group A data yet")

                with col2:
                    st.markdown("**📊 Group B (Content-Based Filtering)**")
                    if group_b_actions:
                        st.metric("Total Actions", sum(group_b_actions.values()))
                        st.metric("Unique Users", len(group_b_users))
                        st.markdown("**Actions:**")
                        for action, count in group_b_actions.items():
                            st.metric(f"{action.replace('_', ' ').title()}", count)
                    else:
                        st.info("No Group B data yet")

                # Calculate engagement metrics
                if group_a_actions and group_b_actions:
                    st.markdown("---")
                    st.markdown("**📈 Engagement Comparison**")

                    # Key engagement metrics
                    engagement_metrics = {}
                    for action in [
                        "view_recommendations",
                        "add_to_watchlist",
                        "generate_recommendations",
                    ]:
                        a_count = group_a_actions.get(action, 0)
                        b_count = group_b_actions.get(action, 0)
                        total_a = sum(group_a_actions.values())
                        total_b = sum(group_b_actions.values())

                        if total_a > 0 and total_b > 0:
                            a_rate = (a_count / total_a) * 100
                            b_rate = (b_count / total_b) * 100
                            improvement = (
                                ((b_rate - a_rate) / a_rate) * 100 if a_rate > 0 else 0
                            )

                            engagement_metrics[action] = {
                                "Group A Rate": round(a_rate, 2),
                                "Group B Rate": round(b_rate, 2),
                                "Improvement": round(improvement, 2),
                            }

                    if engagement_metrics:
                        eng_df = pd.DataFrame(engagement_metrics).T
                        st.dataframe(eng_df, use_container_width=True)

                        # Summary insights
                        st.markdown("**💡 Key Insights:**")
                        for action, metrics in engagement_metrics.items():
                            if metrics["Improvement"] > 0:
                                st.success(
                                    f"✅ {action.replace('_', ' ').title()}: Group B performs {metrics['Improvement']}% better"
                                )
                            elif metrics["Improvement"] < 0:
                                st.error(
                                    f"❌ {action.replace('_', ' ').title()}: Group A performs {abs(metrics['Improvement'])}% better"
                                )
                            else:
                                st.info(
                                    f"⚖️ {action.replace('_', ' ').title()}: Both groups perform similarly"
                                )

                # Statistical significance (basic)
                if group_a_actions and group_b_actions:
                    st.markdown("---")
                    st.markdown("**📊 Statistical Summary**")

                    total_a_actions = sum(group_a_actions.values())
                    total_b_actions = sum(group_b_actions.values())

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Group A Actions", total_a_actions)
                    with col2:
                        st.metric("Total Group B Actions", total_b_actions)
                    with col3:
                        if total_a_actions > 0 and total_b_actions > 0:
                            ratio = total_b_actions / total_a_actions
                            st.metric("B/A Ratio", f"{ratio:.2f}")

            else:
                st.info(
                    "No A/B testing data available yet. Create multiple users and interact with the app to see group comparisons."
                )

        # Log session duration at end of session
        if "session_start" in st.session_state:
            session_duration = time.time() - st.session_state["session_start"]
            monitor.record_user_action(
                user["id"],
                "session_duration",
                {
                    "duration_seconds": session_duration,
                    "ab_group": user.get("ab_group", "Unknown"),
                },
            )


if __name__ == "__main__":
    monitor = get_performance_monitor()
    monitor.start_system_monitoring(interval_seconds=60)

    # Initialize model performance metrics on startup
    try:
        recommender.get_model_stats()
        print("Initial model performance metrics logged.")
    except Exception as e:
        print(f"Could not initialize model performance metrics: {e}")

    main()
