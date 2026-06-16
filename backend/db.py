import sqlite3
import time
from pathlib import Path
from uuid import uuid4

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = str(DATA_DIR / "rag.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                file_type TEXT NOT NULL DEFAULT 'text',
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                chunk_index INTEGER NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
        """)


def get_all_documents() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT d.id, d.title, d.content, d.file_type, d.created_at,
                   COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.created_at DESC
        """).fetchall()
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "content": r["content"],
            "fileType": r["file_type"],
            "createdAt": r["created_at"],
            "chunkCount": r["chunk_count"],
        }
        for r in rows
    ]


def get_stats() -> dict:
    with get_conn() as conn:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return {"docCount": doc_count, "chunkCount": chunk_count}


def delete_document_by_id(doc_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


# ── Sessions ──────────────────────────────────────────────────────────────


def create_session(title: str) -> str:
    session_id = str(uuid4())
    now = int(time.time() * 1000)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title[:50], now, now),
        )
    return session_id


def get_sessions() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   COUNT(m.id) AS message_count
            FROM sessions s
            LEFT JOIN session_messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_session_messages(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM session_messages "
            "WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def add_session_message(session_id: str, role: str, content: str) -> None:
    now = int(time.time() * 1000)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO session_messages (id, session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(uuid4()), session_id, role, content, now),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )


def delete_session_by_id(session_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
