from __future__ import annotations
import hashlib
import os
import secrets
import uuid
import bcrypt
from dataclasses import dataclass
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional
import rio


class User:
    def __init__(
        self,
        id: int,
        username: str,
        password_hash: str,
        email: str,
        created_at: datetime,
    ):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.email = email
        self.created_at = created_at

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()

    def verify_password(self, password: str) -> bool:
        """Verify a password against the stored hash"""
        return bcrypt.checkpw(password.encode(), self.password_hash.encode())


@dataclass
class UserSettings(rio.UserSettings):
    auth_token: Optional[str] = None
    theme_preference: str = "light"
    last_login: Optional[datetime] = None


class Session(BaseModel):
    id: str
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime


@dataclass
class UserSession:
    def __init__(self, user_id: str, valid_until: datetime):
        self.user_id = user_id

        self.valid_until = valid_until


@dataclass
class AppUser:
    id: uuid.UUID
    username: str
    email: str  # Added email field
    created_at: datetime
    last_login: Optional[datetime]
    password_hash: bytes
    password_salt: bytes
    is_active: bool = True
    failed_login_attempts: int = 0

    @classmethod
    def new_with_defaults(cls, username: str, email: str, password: str) -> AppUser:
        password_salt = os.urandom(64)
        return AppUser(
            id=uuid.uuid4(),
            username=username,
            email=email,
            created_at=datetime.now(timezone.utc),
            last_login=None,
            password_hash=cls.get_password_hash(password, password_salt),
            password_salt=password_salt,
        )

    @staticmethod
    def get_password_hash(password: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)

    def verify_password(self, password: str) -> bool:
        return secrets.compare_digest(
            self.password_hash, self.get_password_hash(password, self.password_salt)
        )
