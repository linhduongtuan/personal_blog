import sqlite3
import uuid
import asyncpg
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, ClassVar

from .data_models import Session, AppUser, User


class UserPersistence:
    pool: ClassVar[Optional[asyncpg.Pool]] = None

    def __init__(self, db_path: Path = Path.cwd() / "user.db", pool=None) -> None:
        self.conn = sqlite3.connect(db_path.resolve(), check_same_thread=False)
        self._create_user_table()
        self._create_session_table()
        self._create_login_attempts_table()
        self.sessions = {}  # Simple in-memory storage for sessions

        # Constants for login attempt tracking
        self.MAX_FAILED_ATTEMPTS = 5
        self.LOCKOUT_DURATION = timedelta(minutes=30)

    @classmethod
    async def ensure_pool(cls) -> asyncpg.Pool:
        if cls.pool is None:
            cls.pool = await asyncpg.create_pool(
                user="your_user",
                password="your_password",
                database="your_db",
                host="localhost",
            )
            await cls.init_tables()
        return cls.pool

    async def __post_init__(self) -> None:
        UserPersistence.pool = await self.ensure_pool()

    @classmethod
    async def init_tables(cls) -> None:
        pool = await cls.ensure_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Create users table
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL
                    )
                """)

                # Create sessions table
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id VARCHAR(36) PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        expires_at TIMESTAMP WITH TIME ZONE NOT NULL
                    )
                """)

                # Create login_attempts table
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS login_attempts (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        success BOOLEAN NOT NULL,
                        attempted_at TIMESTAMP WITH TIME ZONE NOT NULL
                    )
                """)

                await conn.commit()

    async def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        pool = await self.ensure_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, username, password_hash, email, created_at
                    FROM users
                    WHERE username = %s
                    """,
                    (username,),
                )
                result = await cur.fetchone()

                if result is None:
                    return None

                return User(
                    id=result[0],
                    username=result[1],
                    password_hash=result[2],
                    email=result[3],
                    created_at=result[4],
                )

    def _create_user_table(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at REAL NOT NULL,
                last_login REAL,
                password_hash BLOB NOT NULL,
                password_salt BLOB NOT NULL
            )
            """
        )
        self.conn.commit()

    def _create_session_table(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        self.conn.commit()

    def _create_login_attempts_table(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS login_attempts (
                attempt_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                attempt_time REAL NOT NULL,
                success BOOLEAN NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        self.conn.commit()

    async def create_session(self, user_id: str) -> Session:
        session_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(hours=1)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (session_id, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, user_id, created_at.timestamp(), expires_at.timestamp()),
        )
        self.conn.commit()

        return Session(
            id=session_id,
            session_id=session_id,
            user_id=user_id,
            created_at=created_at,
            expires_at=expires_at,
        )

    async def get_user_by_id(self, user_id: str) -> AppUser:
        """Get user by ID"""
        cursor = self.conn.execute(
            "SELECT id, username, email, created_at, last_login, password_hash, password_salt FROM users WHERE id = ?",
            (user_id,),
        )
        if row := cursor.fetchone():
            return AppUser(
                id=row[0],
                username=row[1],
                email=row[2],
                created_at=datetime.fromtimestamp(row[3]),
                last_login=datetime.fromtimestamp(row[4]) if row[4] else None,
                password_hash=row[5],
                password_salt=row[6],
            )
        raise KeyError(f"User with ID {user_id} not found")

    async def get_users(self) -> list["AppUser"]:
        """Get all users from the database"""
        cursor = self.conn.execute(
            "SELECT id, username, email, created_at, last_login, password_hash, password_salt FROM users"
        )
        users = []
        for row in cursor.fetchall():
            users.append(
                AppUser(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    created_at=datetime.fromtimestamp(row[3]),
                    last_login=datetime.fromtimestamp(row[4]) if row[4] else None,
                    password_hash=row[5],
                    password_salt=row[6],
                )
            )
        return users

    async def get_user_by_username(self, username: str) -> "AppUser":
        """Get a user by their username"""
        cursor = self.conn.execute(
            "SELECT id, username, email, created_at, last_login, password_hash, password_salt FROM users WHERE username = ?",
            (username,),
        )
        if row := cursor.fetchone():
            return AppUser(
                id=row[0],
                username=row[1],
                email=row[2],
                created_at=datetime.fromtimestamp(row[3]),
                last_login=datetime.fromtimestamp(row[4]) if row[4] else None,
                password_hash=row[5],
                password_salt=row[6],
            )
        raise KeyError(f"User {username} not found")

    async def handle_login_attempt(self, user_id: str, success: bool) -> None:
        """
        Record login attempt and handle account locking
        """
        attempt_id = str(uuid.uuid4())
        attempt_time = datetime.utcnow()

        cursor = self.conn.cursor()
        try:
            # Record the attempt
            cursor.execute(
                """
                INSERT INTO login_attempts (attempt_id, user_id, attempt_time, success)
                VALUES (?, ?, ?, ?)
                """,
                (attempt_id, user_id, attempt_time.timestamp(), success),
            )

            if not success:
                # Check recent failed attempts
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM login_attempts
                    WHERE user_id = ? 
                    AND success = 0
                    AND attempt_time > ?
                    """,
                    (user_id, (attempt_time - self.LOCKOUT_DURATION).timestamp()),
                )
                failed_attempts = cursor.fetchone()[0]

                if failed_attempts >= self.MAX_FAILED_ATTEMPTS:
                    # Update user's last_login to mark lockout time
                    cursor.execute(
                        """
                        UPDATE users 
                        SET last_login = ?
                        WHERE id = ?
                        """,
                        (attempt_time.timestamp(), user_id),
                    )
            else:
                # On successful login, clear failed attempts
                cursor.execute(
                    """
                    DELETE FROM login_attempts
                    WHERE user_id = ? AND success = 0
                    """,
                    (user_id,),
                )

                # Update last successful login
                cursor.execute(
                    """
                    UPDATE users 
                    SET last_login = ?
                    WHERE id = ?
                    """,
                    (attempt_time.timestamp(), user_id),
                )

            self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            raise e

    async def is_account_locked(self, user_id: str) -> bool:
        """
        Check if account is locked due to too many failed attempts
        """
        current_time = datetime.utcnow()
        check_time = (current_time - self.LOCKOUT_DURATION).timestamp()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM login_attempts
            WHERE user_id = ?
            AND success = 0
            AND attempt_time > ?
            """,
            (user_id, check_time),
        )

        failed_attempts = cursor.fetchone()[0]
        return failed_attempts >= self.MAX_FAILED_ATTEMPTS

    async def get_session(self, auth_token: str) -> Session | None:
        """Retrieve a user session by auth token."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT session_id, user_id, created_at, expires_at 
            FROM sessions 
            WHERE session_id = ?
            """,
            (auth_token,),
        )

        if row := cursor.fetchone():
            return Session(
                id=row[0],
                session_id=row[0],
                user_id=row[1],
                created_at=datetime.fromtimestamp(row[2]),
                expires_at=datetime.fromtimestamp(row[3]),
            )
        return None

    async def update_session_duration(self, user_session, new_valid_until):
        """Update the session duration for a user."""

        user_session.valid_until = new_valid_until

    async def create_user(self, username: str, password: str, email: str) -> User:
        """Create a new user"""
        pool = await self.ensure_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Check if username already exists
                existing_user = await self.get_user(username)
                if existing_user is not None:
                    raise ValueError("Username already exists")

                # Insert new user
                await cur.execute(
                    """
                    INSERT INTO users (username, password_hash, email, created_at)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id, username, password_hash, email, created_at
                """,
                    username,
                    password,
                    email,
                    datetime.now(timezone.utc),
                )

                user_data = await cur.fetchone()
                await conn.commit()

                return User(
                    id=user_data[0],
                    username=user_data[1],
                    password_hash=user_data[2],
                    email=user_data[3],
                    created_at=user_data[4],
                )


# In your sign up form component
class UserSignUpForm:
    def __init__(self, pool):
        self.pers = UserPersistence(pool)
        self.username = None

    async def on_sign_up(self):
        if self.username is None:
            raise ValueError("Username cannot be None")
        await self.pers.get_user(self.username)


async def init_app():
    async def init_app():
        pool = await asyncpg.create_pool(
            user="your_user",
            password="your_password",
            database="your_db",
            host="localhost",
        )
        return pool


async def main():
    pool = await init_app()
    signup_form = UserSignUpForm(pool)
    return pool, signup_form


# Run the async initialization
# pool, signup_form = asyncio.run(main())
