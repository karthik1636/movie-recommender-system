import sqlite3
import pandas as pd
import os

db_path = "movie_recommender.db"
csv_path = os.path.join("data", "ml-latest-small", "movies.csv")

# Load movies.csv
movies = pd.read_csv(csv_path)

# Connect to your database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the movies table if it doesn't exist
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS movies (
        movieId INTEGER PRIMARY KEY,
        title TEXT,
        genres TEXT
    )
"""
)

# Insert data
for _, row in movies.iterrows():
    cursor.execute(
        "INSERT OR IGNORE INTO movies (movieId, title, genres) VALUES (?, ?, ?)",
        (int(row["movieId"]), row["title"], row["genres"]),
    )

conn.commit()
conn.close()
print("Movies table created and populated.")
