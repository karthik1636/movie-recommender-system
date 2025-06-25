import streamlit as st
import sys
import os

# Add the api directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

from api.recommender import MovieRecommenderAPI

# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# Title and description
st.title("🎬 Movie Recommender System")
st.markdown("A local movie recommendation system using collaborative filtering and AI")

# Initialize the API
@st.cache_resource
def load_recommender():
    return MovieRecommenderAPI()

recommender = load_recommender()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Home", "User Recommendations", "Similar Movies", "Movie Search", "AI Assistant", "System Info"]
)

if page == "Home":
    st.header("Welcome to the Movie Recommender System!")
    
    st.markdown("""
    This system provides movie recommendations using collaborative filtering based on the MovieLens dataset.
    
    **Features:**
    - 🎯 Get personalized movie recommendations for users
    - 🔍 Find similar movies
    - 📚 Search through the movie database
    - 🤖 AI-powered natural language queries
    - 📊 View system statistics
    
    **How it works:**
    The system uses Singular Value Decomposition (SVD) to learn user preferences and movie characteristics
    from the rating data, then recommends movies based on similarity scores.
    
    **New AI Features:**
    - Ask for movies in natural language (e.g., "Give me sci-fi thrillers from the 2010s")
    - Get explanations for why movies were recommended
    - Powered by local LLM (Llama3) via Ollama
    """)
    
    # Show model stats
    st.subheader("System Statistics")
    stats = recommender.get_model_stats()
    
    if "error" not in stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", stats["total_users"])
        with col2:
            st.metric("Total Movies", stats["total_movies"])
        with col3:
            st.metric("SVD Components", stats["svd_components"])
        with col4:
            st.metric("Explained Variance", f"{stats['explained_variance_ratio']:.3f}")
    else:
        st.error(stats["error"])

elif page == "User Recommendations":
    st.header("Get Movie Recommendations")
    
    # Get available users
    users_data = recommender.get_available_users(100)
    
    if "error" not in users_data:
        st.write(f"Available users: {users_data['total_users']}")
        
        # User selection
        selected_user = st.selectbox(
            "Select a user:",
            users_data["users"],
            format_func=lambda x: f"User {x}"
        )
        
        # Number of recommendations
        n_recommendations = st.slider("Number of recommendations:", 1, 20, 10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Get Recommendations"):
                with st.spinner("Getting recommendations..."):
                    recommendations = recommender.get_user_recommendations(selected_user, n_recommendations)
                
                if "error" not in recommendations:
                    st.success(f"Found {recommendations['count']} recommendations for User {selected_user}")
                    
                    # Display recommendations
                    for i, rec in enumerate(recommendations['recommendations'], 1):
                        with st.expander(f"{i}. {rec['title']} (Score: {rec['similarity_score']:.3f})"):
                            st.write(f"**Genres:** {rec['genres']}")
                            st.write(f"**Movie ID:** {rec['movieId']}")
                            st.write(f"**Similarity Score:** {rec['similarity_score']:.4f}")
                else:
                    st.error(recommendations["error"])
        
        with col2:
            if st.button("🤖 Explain Recommendations"):
                with st.spinner("AI is analyzing the recommendations..."):
                    explanation = recommender.explain_recommendations(selected_user, model="deepseek-coder")
                
                if "error" not in explanation:
                    st.success("AI Explanation Generated!")
                    st.markdown("### 🤖 AI Explanation")
                    st.write(explanation["explanation"])
                    
                    st.markdown("### 📊 Recommended Movies")
                    for i, rec in enumerate(explanation["recommendations"], 1):
                        st.write(f"{i}. **{rec['title']}** ({rec['genres']}) - Score: {rec['similarity_score']:.3f}")
                else:
                    st.error(explanation["error"])
    else:
        st.error(users_data["error"])

elif page == "Similar Movies":
    st.header("Find Similar Movies")
    
    # Movie search for selection
    search_query = st.text_input("Search for a movie to find similar ones:")
    
    if search_query:
        search_results = recommender.search_movies(search_query, 10)
        
        if "error" not in search_results and search_results["count"] > 0:
            st.write(f"Found {search_results['count']} movies matching '{search_query}'")
            
            # Movie selection
            movie_options = {f"{movie['title']} (ID: {movie['movieId']})": movie['movieId'] 
                           for movie in search_results['movies']}
            
            selected_movie = st.selectbox("Select a movie:", list(movie_options.keys()))
            selected_movie_id = movie_options[selected_movie]
            
            # Number of similar movies
            n_similar = st.slider("Number of similar movies:", 1, 20, 10)
            
            if st.button("Find Similar Movies"):
                with st.spinner("Finding similar movies..."):
                    similar_movies = recommender.get_similar_movies(selected_movie_id, n_similar)
                
                if "error" not in similar_movies:
                    st.success(f"Found {similar_movies['count']} similar movies")
                    
                    # Display similar movies
                    for i, movie in enumerate(similar_movies['similar_movies'], 1):
                        with st.expander(f"{i}. {movie['title']} (Score: {movie['similarity_score']:.3f})"):
                            st.write(f"**Genres:** {movie['genres']}")
                            st.write(f"**Movie ID:** {movie['movieId']}")
                            st.write(f"**Similarity Score:** {movie['similarity_score']:.4f}")
                else:
                    st.error(similar_movies["error"])
        elif "error" in search_results:
            st.error(search_results["error"])
        else:
            st.warning(f"No movies found matching '{search_query}'")

elif page == "Movie Search":
    st.header("Search Movies")
    
    # Search functionality
    search_query = st.text_input("Enter movie title to search:")
    
    if search_query:
        search_results = recommender.search_movies(search_query, 20)
        
        if "error" not in search_results:
            st.write(f"Found {search_results['count']} movies matching '{search_query}'")
            
            # Display search results
            for movie in search_results['movies']:
                with st.expander(f"{movie['title']} (ID: {movie['movieId']})"):
                    st.write(f"**Genres:** {movie['genres']}")
                    st.write(f"**Movie ID:** {movie['movieId']}")
        else:
            st.error(search_results["error"])

elif page == "AI Assistant":
    st.header("🤖 AI Movie Assistant")
    st.markdown("Ask for movie recommendations in natural language!")
    
    # Model selection
    available_models = ["phi", "gemma:2b"]
    selected_model = st.selectbox("Select AI Model:", available_models, index=0)
    
    # Query input
    query = st.text_area(
        "Ask for movie recommendations:",
        placeholder="Examples:\n- Give me sci-fi thrillers from the 2010s\n- I want romantic comedies\n- Recommend action movies like Die Hard\n- Show me horror movies from the 80s",
        height=100
    )
    
    # Number of recommendations
    n_recommendations = st.slider("Number of recommendations:", 1, 20, 10)
    
    if st.button("🤖 Ask AI Assistant"):
        if query.strip():
            with st.spinner("AI is processing your request..."):
                result = recommender.ask_llm(query, model=selected_model, n_recommendations=n_recommendations)
            
            if "error" not in result:
                st.success(f"AI generated {result['count']} recommendations!")
                
                # Display AI response
                st.markdown("### 🤖 AI Response")
                st.write(result["llm_response"])
                
                # Show if fallback was used
                if result.get("used_fallback", False):
                    st.info("⚠️ AI couldn't find enough specific movies, so I used keyword search as a fallback.")
                
                # Display recommended movies
                if result["recommended_movies"]:
                    st.markdown("### 🎬 Recommended Movies")
                    for i, movie in enumerate(result["recommended_movies"], 1):
                        with st.expander(f"{i}. {movie['title']}"):
                            st.write(f"**Genres:** {movie['genres']}")
                            st.write(f"**Movie ID:** {movie['movieId']}")
                else:
                    st.info("No specific movies found in the database matching the AI suggestions.")
                    st.markdown("**💡 Tip:** Try using more specific genre terms like 'action', 'comedy', 'drama', or specific movie titles.")
                
                # Show model info
                st.info(f"Generated using {result['model_used']} model")
            else:
                st.error(result["error"])
                st.markdown("**💡 Tip:** Try using different keywords or a different AI model.")
        else:
            st.warning("Please enter a query first.")
    
    # Example queries
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

elif page == "System Info":
    st.header("System Information")
    
    st.subheader("About the System")
    st.markdown("""
    **Technology Stack:**
    - **Python:** Core programming language
    - **Pandas:** Data manipulation and analysis
    - **NumPy:** Numerical computing
    - **Scikit-learn:** Machine learning algorithms (SVD)
    - **Streamlit:** Web interface
    - **Ollama:** Local LLM integration
    - **Llama3:** AI model for natural language processing
    
    **Algorithm:**
    - **Collaborative Filtering** using Singular Value Decomposition (SVD)
    - **Cosine Similarity** for finding similar users and movies
    - **Matrix Factorization** to learn latent features
    - **AI Integration** for natural language queries and explanations
    
    **Dataset:**
    - **MovieLens Small Dataset** (100,000 ratings from 600 users on 9,000 movies)
    - Automatically downloaded and processed
    - Includes movie titles, genres, and user ratings
    
    **AI Features:**
    - **Natural Language Queries:** Ask for movies in plain English
    - **Recommendation Explanations:** AI explains why movies were recommended
    - **Local Processing:** All AI runs locally via Ollama
    - **Multiple Models:** Support for Llama3, Llama3.1, and DeepSeek-Coder
    """)
    
    # Model statistics
    st.subheader("Model Statistics")
    stats = recommender.get_model_stats()
    
    if "error" not in stats:
        st.json(stats)
    else:
        st.error(stats["error"])
    
    # Available AI models
    st.subheader("Available AI Models")
    st.markdown("""
    The system supports these local AI models via Ollama:
    - **llama3:latest** (4.7 GB) - General purpose model
    - **llama3.1:latest** (4.7 GB) - Updated version
    - **deepseek-coder:latest** (776 MB) - Code-focused model
    
    To add more models, run: `ollama pull model_name`
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Scikit-learn, and Ollama")
