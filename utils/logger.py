"""
Logging utilities for Movie Recommender System
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json
from pathlib import Path

from config import get_config


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors"""
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Add color to level name
        record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


class Logger:
    """Main logger class"""

    def __init__(self, name: str = "movie_recommender"):
        self.name = name
        self.config = get_config()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger with handlers and formatters"""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.config.logging.level.upper()))

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        if self.config.app.debug:
            console_formatter = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        else:
            console_formatter = StructuredFormatter()

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler (if configured)
        if self.config.logging.file:
            self._setup_file_handler(logger)

        return logger

    def _setup_file_handler(self, logger: logging.Logger):
        """Setup file handler with rotation"""
        log_file = Path(self.config.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.logging.max_bytes,
            backupCount=self.config.logging.backup_count,
        )
        file_handler.setLevel(logging.INFO)

        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    def debug(self, message: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self._log(logging.DEBUG, message, extra_fields)

    def info(self, message: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self._log(logging.INFO, message, extra_fields)

    def warning(self, message: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self._log(logging.WARNING, message, extra_fields)

    def error(self, message: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log error message"""
        self._log(logging.ERROR, message, extra_fields)

    def critical(self, message: str, extra_fields: Optional[Dict[str, Any]] = None):
        """Log critical message"""
        self._log(logging.CRITICAL, message, extra_fields)

    def _log(
        self, level: int, message: str, extra_fields: Optional[Dict[str, Any]] = None
    ):
        """Internal logging method"""
        if extra_fields:
            # Create a new record with extra fields
            record = self.logger.makeRecord(self.name, level, "", 0, message, (), None)
            record.extra_fields = extra_fields
            self.logger.handle(record)
        else:
            self.logger.log(level, message)

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        response_time: float,
        user_id: Optional[int] = None,
    ):
        """Log HTTP request details"""
        extra_fields = {
            "type": "http_request",
            "method": method,
            "path": path,
            "status_code": status_code,
            "response_time_ms": round(response_time * 1000, 2),
            "user_id": user_id,
        }

        level = logging.INFO if status_code < 400 else logging.WARNING
        self._log(level, f"{method} {path} - {status_code}", extra_fields)

    def log_recommendation(
        self,
        user_id: int,
        algorithm: str,
        num_recommendations: int,
        response_time: float,
    ):
        """Log recommendation request"""
        extra_fields = {
            "type": "recommendation",
            "user_id": user_id,
            "algorithm": algorithm,
            "num_recommendations": num_recommendations,
            "response_time_ms": round(response_time * 1000, 2),
        }

        self.info(f"Recommendation request for user {user_id}", extra_fields)

    def log_llm_query(
        self,
        model: str,
        query: str,
        response_time: float,
        num_results: int,
        success: bool,
    ):
        """Log LLM query details"""
        extra_fields = {
            "type": "llm_query",
            "model": model,
            "query": query,
            "response_time_ms": round(response_time * 1000, 2),
            "num_results": num_results,
            "success": success,
        }

        level = logging.INFO if success else logging.ERROR
        self._log(level, f"LLM query using {model}", extra_fields)

    def log_user_action(
        self, user_id: int, action: str, details: Optional[Dict] = None
    ):
        """Log user actions"""
        extra_fields = {"type": "user_action", "user_id": user_id, "action": action}

        if details:
            extra_fields.update(details)

        self.info(f"User {user_id} performed {action}", extra_fields)

    def log_model_performance(self, metric: str, value: float, algorithm: str = "svd"):
        """Log model performance metrics"""
        extra_fields = {
            "type": "model_performance",
            "metric": metric,
            "value": value,
            "algorithm": algorithm,
        }

        self.info(f"Model performance: {metric} = {value}", extra_fields)

    def log_error_with_context(self, error: Exception, context: Optional[Dict] = None):
        """Log error with additional context"""
        extra_fields = {
            "type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        if context:
            extra_fields.update(context)

        self.error(f"Error occurred: {error}", extra_fields)


# Global logger instance
_logger = None


def get_logger(name: str = "movie_recommender") -> Logger:
    """Get global logger instance"""
    global _logger
    if _logger is None:
        _logger = Logger(name)
    return _logger


def setup_logging(name: str = "movie_recommender") -> Logger:
    """Setup and return logger instance"""
    return get_logger(name)


# Convenience functions for quick logging
def log_debug(message: str, **kwargs):
    """Quick debug logging"""
    get_logger().debug(message, kwargs if kwargs else None)


def log_info(message: str, **kwargs):
    """Quick info logging"""
    get_logger().info(message, kwargs if kwargs else None)


def log_warning(message: str, **kwargs):
    """Quick warning logging"""
    get_logger().warning(message, kwargs if kwargs else None)


def log_error(message: str, **kwargs):
    """Quick error logging"""
    get_logger().error(message, kwargs if kwargs else None)


def log_critical(message: str, **kwargs):
    """Quick critical logging"""
    get_logger().critical(message, kwargs if kwargs else None)


# Performance monitoring decorator
def log_performance(func):
    """Decorator to log function performance"""

    def wrapper(*args, **kwargs):
        import time

        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Extract function name and module
            func_name = func.__name__
            module_name = func.__module__

            extra_fields = {
                "type": "function_performance",
                "function": f"{module_name}.{func_name}",
                "execution_time_ms": round(execution_time * 1000, 2),
                "success": True,
            }

            get_logger().info(f"Function {func_name} completed", extra_fields)
            return result

        except Exception as e:
            execution_time = time.time() - start_time

            extra_fields = {
                "type": "function_performance",
                "function": f"{func.__module__}.{func.__name__}",
                "execution_time_ms": round(execution_time * 1000, 2),
                "success": False,
                "error": str(e),
            }

            get_logger().error(f"Function {func.__name__} failed", extra_fields)
            raise

    return wrapper
