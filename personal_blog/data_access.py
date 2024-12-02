from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import List, Dict, Any


class ContentDatabase:
    """Handle all content-related database operations."""

    def __init__(self, db_path: Path = Path("content.db")) -> None:
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        cursor = self.conn.cursor()

        # Create blog_posts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blog_posts (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)

        # Create research_papers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                published_date REAL NOT NULL
            )
        """)

        self.conn.commit()

    async def get_blog_posts(self) -> List[Dict[str, Any]]:
        """Fetch blog posts from storage."""
        posts = []
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, title, content, created_at 
                FROM blog_posts 
                ORDER BY created_at DESC
            """)
            for row in cursor.fetchall():
                posts.append(
                    {
                        "id": row[0],
                        "title": row[1],
                        "content": row[2],
                        "created_at": row[3],
                    }
                )
        except Exception as e:
            print(f"Error fetching blog posts: {e}")
        return posts

    async def get_research_papers(self) -> List[Dict[str, Any]]:
        """Fetch research papers from storage."""
        papers = []
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, title, abstract, published_date 
                FROM research_papers 
                ORDER BY published_date DESC
            """)
            for row in cursor.fetchall():
                papers.append(
                    {
                        "id": row[0],
                        "title": row[1],
                        "abstract": row[2],
                        "published_date": row[3],
                    }
                )
        except Exception as e:
            print(f"Error fetching research papers: {e}")
        return papers
