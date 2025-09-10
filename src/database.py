"""SQLite database operations for tracking document hashes and metadata."""

import sqlite3
from typing import List, Tuple, Optional
from datetime import datetime
from pathlib import Path


class DocumentDatabase:
    """Manages SQLite database for document tracking."""

    def __init__(self, db_path: str):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    domain TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_overlap INTEGER NOT NULL,
                    last_ingested TIMESTAMP NOT NULL,
                    PRIMARY KEY (domain, file_path)
                )
            """)
            conn.commit()

    def get_file_info(self, domain: str, file_path: str) -> Optional[Tuple[str, int, int, datetime]]:
        """Get stored file information.

        Returns:
            Tuple of (file_hash, chunk_size, chunk_overlap, last_ingested) or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT file_hash, chunk_size, chunk_overlap, last_ingested FROM files WHERE domain = ? AND file_path = ?",
                (domain, file_path)
            )
            row = cursor.fetchone()

            if row:
                return (
                    row['file_hash'],
                    row['chunk_size'],
                    row['chunk_overlap'],
                    datetime.fromisoformat(row['last_ingested'])
                )
            return None

    def update_file_info(self, domain: str, file_path: str, file_hash: str,
                        chunk_size: int, chunk_overlap: int) -> None:
        """Update or insert file information."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO files
                (domain, file_path, file_hash, chunk_size, chunk_overlap, last_ingested)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (domain, file_path, file_hash, chunk_size, chunk_overlap, datetime.now().isoformat()))
            conn.commit()

    def get_tracked_files(self, domain: str) -> List[str]:
        """Get list of all tracked files for a domain."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_path FROM files WHERE domain = ?",
                (domain,)
            )
            return [row[0] for row in cursor.fetchall()]

    def remove_file(self, domain: str, file_path: str) -> None:
        """Remove a file from tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM files WHERE domain = ? AND file_path = ?",
                (domain, file_path)
            )
            conn.commit()

    def remove_files(self, domain: str, file_paths: List[str]) -> None:
        """Remove multiple files from tracking."""
        if not file_paths:
            return

        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join('?' * len(file_paths))
            conn.execute(
                f"DELETE FROM files WHERE domain = ? AND file_path IN ({placeholders})",
                [domain] + file_paths
            )
            conn.commit()

    def get_all_domains(self) -> List[str]:
        """Get list of all tracked domains."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT domain FROM files")
            return [row[0] for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Total files
            cursor = conn.execute("SELECT COUNT(*) as total FROM files")
            total_files = cursor.fetchone()['total']

            # Files per domain
            cursor = conn.execute("SELECT domain, COUNT(*) as count FROM files GROUP BY domain")
            domain_counts = {row['domain']: row['count'] for row in cursor.fetchall()}

            return {
                'total_files': total_files,
                'domains': domain_counts
            }
