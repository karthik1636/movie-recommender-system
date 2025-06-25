"""
Security utilities for Movie Recommender System
"""

import hashlib
import secrets
import re
import html
import os
from typing import Optional, Tuple
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext

from config import get_config
from utils.logger import get_logger


class SecurityManager:
    """Main security management class"""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("security")
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Rate limiting storage (in production, use Redis)
        self._login_attempts = {}
        self._ip_blacklist = {}

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return self.pwd_context.hash(password)

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(password, hashed)

    def generate_token(self, user_id: int, username: str) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user_id,
            "username": username,
            "exp": datetime.utcnow()
            + timedelta(seconds=self.config.security.session_timeout),
            "iat": datetime.utcnow(),
        }

        return jwt.encode(payload, self.config.security.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, self.config.security.secret_key, algorithms=["HS256"]
            )
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid token")
            return None

    def validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"

        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"

        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"

        if not re.search(r"\d", password):
            return False, "Password must contain at least one digit"

        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False, "Password must contain at least one special character"

        return True, "Password is strong"

    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    def sanitize_input(self, text: str) -> str:
        """Sanitize user input to prevent XSS"""
        # Remove HTML tags
        text = re.sub(r"<[^>]*>", "", text)

        # Escape HTML entities
        text = html.escape(text)

        # Remove null bytes
        text = text.replace("\x00", "")

        return text.strip()

    def validate_username(self, username: str) -> Tuple[bool, str]:
        """Validate username format"""
        if len(username) < 3:
            return False, "Username must be at least 3 characters long"

        if len(username) > 30:
            return False, "Username must be less than 30 characters"

        if not re.match(r"^[a-zA-Z0-9_]+$", username):
            return False, "Username can only contain letters, numbers, and underscores"

        return True, "Username is valid"

    def check_rate_limit(self, identifier: str, max_attempts: int = None) -> bool:
        """Check if user/IP is rate limited"""
        if max_attempts is None:
            max_attempts = self.config.security.max_login_attempts

        now = datetime.utcnow()

        # Clean old entries
        self._login_attempts = {
            k: v
            for k, v in self._login_attempts.items()
            if now - v["timestamp"]
            < timedelta(seconds=self.config.security.lockout_duration)
        }

        if identifier in self._login_attempts:
            attempts = self._login_attempts[identifier]

            if attempts["count"] >= max_attempts:
                if now - attempts["timestamp"] < timedelta(
                    seconds=self.config.security.lockout_duration
                ):
                    return False  # Rate limited
                else:
                    # Reset after lockout period
                    del self._login_attempts[identifier]

        return True

    def record_login_attempt(self, identifier: str, success: bool):
        """Record a login attempt"""
        now = datetime.utcnow()

        if identifier not in self._login_attempts:
            self._login_attempts[identifier] = {"count": 0, "timestamp": now}

        if not success:
            self._login_attempts[identifier]["count"] += 1
            self._login_attempts[identifier]["timestamp"] = now
        else:
            # Reset on successful login
            if identifier in self._login_attempts:
                del self._login_attempts[identifier]

    def is_ip_blacklisted(self, ip_address: str) -> bool:
        """Check if IP address is blacklisted"""
        now = datetime.utcnow()

        # Clean old entries
        self._ip_blacklist = {k: v for k, v in self._ip_blacklist.items() if now < v}

        return ip_address in self._ip_blacklist

    def blacklist_ip(self, ip_address: str, duration_minutes: int = 60):
        """Blacklist an IP address"""
        self._ip_blacklist[ip_address] = datetime.utcnow() + timedelta(
            minutes=duration_minutes
        )
        self.logger.warning(
            f"IP {ip_address} blacklisted for {duration_minutes} minutes"
        )

    def generate_secure_random_string(self, length: int = 32) -> str:
        """Generate a secure random string"""
        return secrets.token_urlsafe(length)

    def hash_file_content(self, file_path: str) -> str:
        """Generate SHA-256 hash of file content"""
        hash_sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    def validate_file_upload(
        self, filename: str, allowed_extensions: list = None
    ) -> bool:
        """Validate file upload"""
        if allowed_extensions is None:
            allowed_extensions = [".txt", ".csv", ".json", ".png", ".jpg", ".jpeg"]

        # Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_extensions:
            return False

        # Check for path traversal attempts
        if ".." in filename or "/" in filename or "\\" in filename:
            return False

        return True

    def log_security_event(
        self, event_type: str, details: dict, user_id: Optional[int] = None
    ):
        """Log security events"""
        extra_fields = {
            "type": "security_event",
            "event_type": event_type,
            "user_id": user_id,
        }
        extra_fields.update(details)

        self.logger.warning(f"Security event: {event_type}", extra_fields)


# Global security manager instance
_security_manager = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


# Convenience functions
def hash_password(password: str) -> str:
    """Quick function to hash password"""
    return get_security_manager().hash_password(password)


def verify_password(password: str, hashed: str) -> bool:
    """Quick function to verify password"""
    return get_security_manager().verify_password(password, hashed)


def generate_token(user_id: int, username: str) -> str:
    """Quick function to generate token"""
    return get_security_manager().generate_token(user_id, username)


def verify_token(token: str) -> Optional[dict]:
    """Quick function to verify token"""
    return get_security_manager().verify_token(token)


def sanitize_input(text: str) -> str:
    """Quick function to sanitize input"""
    return get_security_manager().sanitize_input(text)


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """Quick function to validate password strength"""
    return get_security_manager().validate_password_strength(password)


def validate_email(email: str) -> bool:
    """Quick function to validate email"""
    return get_security_manager().validate_email(email)


def validate_username(username: str) -> Tuple[bool, str]:
    """Quick function to validate username"""
    return get_security_manager().validate_username(username)


def check_rate_limit(identifier: str) -> bool:
    """Quick function to check rate limit"""
    return get_security_manager().check_rate_limit(identifier)


def record_login_attempt(identifier: str, success: bool):
    """Quick function to record login attempt"""
    get_security_manager().record_login_attempt(identifier, success)


def log_security_event(event_type: str, details: dict, user_id: Optional[int] = None):
    """Quick function to log security event"""
    get_security_manager().log_security_event(event_type, details, user_id)
