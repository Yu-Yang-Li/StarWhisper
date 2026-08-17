import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

from config.config import settings


_SCHEMA_READY = False
_SCHEMA_PATH = None
_SCHEMA_LOCK = threading.Lock()


def get_auth_db_path() -> str:
    db_path = settings.auth_db_path
    if not os.path.isabs(db_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_root, db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return db_path


@contextmanager
def auth_db_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(get_auth_db_path(), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def ensure_auth_schema() -> None:
    global _SCHEMA_READY, _SCHEMA_PATH
    db_path = get_auth_db_path()
    if _SCHEMA_READY and _SCHEMA_PATH == db_path:
        return

    with _SCHEMA_LOCK:
        db_path = get_auth_db_path()
        if _SCHEMA_READY and _SCHEMA_PATH == db_path:
            return
        with auth_db_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    username TEXT,
                    role TEXT NOT NULL DEFAULT 'human_user',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verification_codes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    used_at TEXT,
                    request_ip TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vc_phone_purpose_created
                ON verification_codes(phone, purpose, created_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vc_phone_purpose_expires
                ON verification_codes(phone, purpose, expires_at)
                """
            )
            conn.commit()
        _SCHEMA_READY = True
        _SCHEMA_PATH = db_path
