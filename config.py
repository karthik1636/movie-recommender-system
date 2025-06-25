"""
Configuration management for Movie Recommender System
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration settings"""

    url: str = "sqlite:///movie_recommender.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30


@dataclass
class ModelConfig:
    """Machine learning model configuration"""

    n_components: int = 50
    random_state: int = 42
    model_path: str = "models/recommender.pkl"
    dataset_url: str = (
        "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    )
    data_dir: str = "data"


@dataclass
class LLMConfig:
    """LLM configuration settings"""

    default_model: str = "phi"
    available_models: list = None
    ollama_url: str = "http://localhost:11434"
    timeout: int = 30
    max_tokens: int = 500

    def __post_init__(self):
        if self.available_models is None:
            self.available_models = ["phi", "gemma:2b"]


@dataclass
class SecurityConfig:
    """Security configuration settings"""

    secret_key: str = "your-secret-key-change-in-production"
    password_salt_rounds: int = 12
    session_timeout: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes


@dataclass
class AppConfig:
    """Application configuration settings"""

    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8501
    title: str = "Movie Recommender System"
    description: str = "AI-powered movie recommendations"
    version: str = "1.0.0"
    theme: str = "light"
    page_icon: str = "🎬"


@dataclass
class CacheConfig:
    """Caching configuration settings"""

    enabled: bool = True
    ttl: int = 3600  # 1 hour
    max_size: int = 1000
    redis_url: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging configuration settings"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


class Config:
    """Main configuration class"""

    def __init__(self, environment: str = None):
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self._load_environment_config()

    def _load_environment_config(self):
        """Load configuration based on environment"""
        if self.environment == "production":
            self._load_production_config()
        elif self.environment == "staging":
            self._load_staging_config()
        else:
            self._load_development_config()

    def _load_development_config(self):
        """Development environment configuration"""
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.llm = LLMConfig()
        self.security = SecurityConfig()
        self.app = AppConfig(debug=True)
        self.cache = CacheConfig(enabled=False)
        self.logging = LoggingConfig(level="DEBUG")

    def _load_staging_config(self):
        """Staging environment configuration"""
        self.database = DatabaseConfig(
            url=os.getenv("DATABASE_URL", "sqlite:///movie_recommender.db"), echo=True
        )
        self.model = ModelConfig()
        self.llm = LLMConfig()
        self.security = SecurityConfig(
            secret_key=os.getenv("SECRET_KEY", "staging-secret-key")
        )
        self.app = AppConfig(debug=False)
        self.cache = CacheConfig()
        self.logging = LoggingConfig(level="INFO")

    def _load_production_config(self):
        """Production environment configuration"""
        self.database = DatabaseConfig(
            url=os.getenv("DATABASE_URL", "sqlite:///movie_recommender.db"),
            echo=False,
            pool_size=20,
            max_overflow=30,
        )
        self.model = ModelConfig(n_components=100)
        self.llm = LLMConfig()
        self.security = SecurityConfig(
            secret_key=os.getenv("SECRET_KEY"),
            password_salt_rounds=16,
            session_timeout=1800,  # 30 minutes
        )
        self.app = AppConfig(debug=False)
        self.cache = CacheConfig(enabled=True, redis_url=os.getenv("REDIS_URL"))
        self.logging = LoggingConfig(level="WARNING", file="logs/app.log")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment,
            "database": self.database.__dict__,
            "model": self.model.__dict__,
            "llm": self.llm.__dict__,
            "security": self.security.__dict__,
            "app": self.app.__dict__,
            "cache": self.cache.__dict__,
            "logging": self.logging.__dict__,
        }

    def validate(self) -> bool:
        """Validate configuration settings"""
        try:
            # Validate required settings
            if (
                not self.security.secret_key
                or self.security.secret_key == "your-secret-key-change-in-production"
            ):
                raise ValueError("SECRET_KEY must be set in production")

            if self.environment == "production":
                if not os.getenv("DATABASE_URL"):
                    raise ValueError("DATABASE_URL must be set in production")

            # Validate model settings
            if self.model.n_components <= 0:
                raise ValueError("n_components must be positive")

            # Validate LLM settings
            if not self.llm.available_models:
                raise ValueError("At least one LLM model must be available")

            return True

        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def reload_config(environment: str = None) -> Config:
    """Reload configuration with new environment"""
    global config
    config = Config(environment)
    return config


# Environment-specific configuration functions
def is_development() -> bool:
    """Check if running in development environment"""
    return config.environment == "development"


def is_staging() -> bool:
    """Check if running in staging environment"""
    return config.environment == "staging"


def is_production() -> bool:
    """Check if running in production environment"""
    return config.environment == "production"


# Configuration validation on import
if not config.validate():
    print("Warning: Configuration validation failed. Check your environment variables.")
