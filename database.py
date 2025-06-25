import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional
import json

class MovieDatabase:
    def __init__(self, db_path="movie_recommender.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # User ratings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                movie_id INTEGER,
                rating REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, movie_id)
            )
        ''')
        
        # User watchlist table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                movie_id INTEGER,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, movie_id)
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                favorite_genres TEXT,
                preferred_decades TEXT,
                max_rating REAL DEFAULT 5.0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, username: str, email: str, password_hash: str) -> bool:
        """Create a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash)
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'password_hash': user[3],
                'created_at': user[4],
                'last_login': user[5]
            }
        return None
    
    def add_rating(self, user_id: int, movie_id: int, rating: float) -> bool:
        """Add or update a user rating"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO user_ratings (user_id, movie_id, rating) VALUES (?, ?, ?)",
                (user_id, movie_id, rating)
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
    
    def get_user_ratings(self, user_id: int) -> List[Dict]:
        """Get all ratings for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT movie_id, rating FROM user_ratings WHERE user_id = ?", (user_id,))
        ratings = cursor.fetchall()
        conn.close()
        
        return [{'movie_id': r[0], 'rating': r[1]} for r in ratings]
    
    def add_to_watchlist(self, user_id: int, movie_id: int) -> bool:
        """Add movie to user's watchlist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO watchlist (user_id, movie_id) VALUES (?, ?)",
                (user_id, movie_id)
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
    
    def get_watchlist(self, user_id: int) -> List[int]:
        """Get user's watchlist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT movie_id FROM watchlist WHERE user_id = ?", (user_id,))
        watchlist = cursor.fetchall()
        conn.close()
        
        return [item[0] for item in watchlist]
    
    def save_user_preferences(self, user_id: int, favorite_genres: List[str], 
                            preferred_decades: List[str], max_rating: float = 5.0) -> bool:
        """Save user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO user_preferences (user_id, favorite_genres, preferred_decades, max_rating) VALUES (?, ?, ?, ?)",
                (user_id, json.dumps(favorite_genres), json.dumps(preferred_decades), max_rating)
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
    
    def get_user_preferences(self, user_id: int) -> Optional[Dict]:
        """Get user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
        prefs = cursor.fetchone()
        conn.close()
        
        if prefs:
            return {
                'favorite_genres': json.loads(prefs[2]) if prefs[2] else [],
                'preferred_decades': json.loads(prefs[3]) if prefs[3] else [],
                'max_rating': prefs[4]
            }
        return None 