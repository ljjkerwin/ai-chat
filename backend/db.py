import os
import time
from uuid import uuid4
import pymysql
import pymysql.cursors
from dotenv import load_dotenv

# Load env from project root's .env.local
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env.local"))


def get_conn():
    host = os.getenv("MYSQL_HOST")
    user = os.getenv("MYSQL_USERNAME")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE", "rag")

    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )
    return conn


def init_db() -> None:
    # First connect without database to create the database if it doesn't exist
    host = os.getenv("MYSQL_HOST")
    user = os.getenv("MYSQL_USERNAME")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE", "rag")

    temp_conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        autocommit=True
    )
    try:
        with temp_conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    finally:
        temp_conn.close()

    # Now connect to the database and create tables
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id VARCHAR(64) PRIMARY KEY,
                    tenant_id VARCHAR(64) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    created_at BIGINT NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            
            # Seed default knowledge base for backward compatibility and initial setup
            cursor.execute("SELECT COUNT(*) as count FROM knowledge_bases WHERE id = '1'")
            if cursor.fetchone()["count"] == 0:
                cursor.execute(
                    "INSERT INTO knowledge_bases (id, tenant_id, name, description, created_at) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    ("1", "1", "默认知识库", "系统初始化的默认知识库", int(time.time() * 1000))
                )

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id VARCHAR(64) PRIMARY KEY,
                    kb_id VARCHAR(64) NOT NULL,
                    tenant_id VARCHAR(64) NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    content LONGTEXT NOT NULL,
                    file_type VARCHAR(50) NOT NULL DEFAULT 'text',
                    created_at BIGINT NOT NULL,
                    CONSTRAINT fk_documents_kb FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            
            # Safe migration: Add kb_id and tenant_id columns to documents table if they do not exist
            try:
                cursor.execute("SELECT kb_id, tenant_id FROM documents LIMIT 1")
            except Exception:
                try:
                    cursor.execute("ALTER TABLE documents ADD COLUMN kb_id VARCHAR(64) NOT NULL DEFAULT '1'")
                    cursor.execute("ALTER TABLE documents ADD COLUMN tenant_id VARCHAR(64) NOT NULL DEFAULT '1'")
                    cursor.execute("ALTER TABLE documents ADD CONSTRAINT fk_documents_kb FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE")
                    print("[DB] Successfully migrated documents table to support multi-KB and tenant isolation.")
                except Exception as ex:
                    print(f"[DB] Migration warning/failed: {ex}")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id VARCHAR(64) PRIMARY KEY,
                    document_id VARCHAR(64) NOT NULL,
                    content LONGTEXT NOT NULL,
                    embedding LONGTEXT,
                    chunk_index INT NOT NULL,
                    CONSTRAINT fk_chunks_document FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id VARCHAR(64) PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    created_at BIGINT NOT NULL,
                    updated_at BIGINT NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_messages (
                    id VARCHAR(64) PRIMARY KEY,
                    session_id VARCHAR(64) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    content LONGTEXT NOT NULL,
                    created_at BIGINT NOT NULL,
                    CONSTRAINT fk_messages_session FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS term_df (
                    term VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin PRIMARY KEY,
                    df INT NOT NULL DEFAULT 0
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            
            # Safe migration: ensure term_df.term column uses binary collation (utf8mb4_bin) to avoid punctuation duplicates
            try:
                cursor.execute("""
                    ALTER TABLE term_df 
                    MODIFY term VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL
                """)
            except Exception as ex:
                print(f"[DB] Migration warning/failed for term_df collation: {ex}")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_metadata (
                    `key` VARCHAR(64) PRIMARY KEY,
                    `value` VARCHAR(255) NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)


def initialize_incremental_stats() -> None:
    """Initialize incremental BM25 statistics if not already initialized."""
    with get_conn() as conn:
        with conn.cursor() as cursor:
            # Check if already initialized
            cursor.execute("SELECT `value` FROM rag_metadata WHERE `key` = 'initialized'")
            row = cursor.fetchone()
            if row and row["value"] == "true":
                return
            
            print("[DB] Initializing incremental BM25 statistics from existing chunks...")
            # If not initialized, fetch all chunks to build stats
            cursor.execute("SELECT content FROM chunks")
            rows = cursor.fetchall()
            
            # Use lazy import to avoid circular dependency
            from rag import tokenize
            
            # Calculate stats
            term_counts = {}
            total_tokens = 0
            for r in rows:
                tokens = tokenize(r["content"])
                total_tokens += len(tokens)
                for token in set(tokens):
                    term_counts[token] = term_counts.get(token, 0) + 1
            
            # Clear any existing data just in case
            cursor.execute("TRUNCATE TABLE term_df")
            cursor.execute("TRUNCATE TABLE rag_metadata")
            
            # Insert term frequencies in batches of 500
            terms_data = [(term, df) for term, df in term_counts.items()]
            batch_size = 500
            for i in range(0, len(terms_data), batch_size):
                batch = terms_data[i : i + batch_size]
                cursor.executemany(
                    "INSERT INTO term_df (term, df) VALUES (%s, %s) "
                    "ON DUPLICATE KEY UPDATE df = VALUES(df)",
                    batch
                )
            
            # Insert metadata
            cursor.execute("INSERT INTO rag_metadata (`key`, `value`) VALUES (%s, %s)", ("total_chunks", str(len(rows))))
            cursor.execute("INSERT INTO rag_metadata (`key`, `value`) VALUES (%s, %s)", ("total_tokens", str(total_tokens)))
            cursor.execute("INSERT INTO rag_metadata (`key`, `value`) VALUES (%s, %s)", ("initialized", "true"))
            
            print(f"[DB] Initialized BM25 stats: {len(rows)} chunks, {len(term_counts)} unique terms, {total_tokens} total tokens.")


def create_knowledge_base(kb_id: str, name: str, description: str, tenant_id: str) -> None:
    import time
    now = int(time.time() * 1000)
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO knowledge_bases (id, tenant_id, name, description, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (kb_id, tenant_id, name, description, now)
            )


def get_knowledge_bases(tenant_id: str) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT kb.id, kb.name, kb.description, kb.created_at,
                       COUNT(DISTINCT d.id) AS doc_count,
                       COUNT(c.id) AS chunk_count
                FROM knowledge_bases kb
                LEFT JOIN documents d ON d.kb_id = kb.id AND d.tenant_id = kb.tenant_id
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE kb.tenant_id = %s
                GROUP BY kb.id, kb.name, kb.description, kb.created_at
                ORDER BY kb.created_at DESC
            """, (tenant_id,))
            rows = cursor.fetchall()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "description": r["description"],
            "createdAt": r["created_at"],
            "docCount": r["doc_count"],
            "chunkCount": r["chunk_count"],
        }
        for r in rows
    ]


def delete_knowledge_base_by_id(kb_id: str, tenant_id: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM knowledge_bases WHERE id = %s AND tenant_id = %s",
                (kb_id, tenant_id)
            )


def get_all_documents(kb_id: str, tenant_id: str) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT d.id, d.title, d.content, d.file_type, d.created_at,
                       COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.kb_id = %s AND d.tenant_id = %s
                GROUP BY d.id, d.title, d.content, d.file_type, d.created_at
                ORDER BY d.created_at DESC
            """, (kb_id, tenant_id))
            rows = cursor.fetchall()
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


def get_stats(kb_id: str, tenant_id: str) -> dict:
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS doc_count FROM documents WHERE kb_id = %s AND tenant_id = %s", (kb_id, tenant_id))
            doc_count = cursor.fetchone()["doc_count"]
            cursor.execute("""
                SELECT COUNT(c.id) AS chunk_count 
                FROM chunks c 
                JOIN documents d ON c.document_id = d.id 
                WHERE d.kb_id = %s AND d.tenant_id = %s
            """, (kb_id, tenant_id))
            chunk_count = cursor.fetchone()["chunk_count"]
    return {"docCount": doc_count, "chunkCount": chunk_count}


def delete_document_by_id(doc_id: str, tenant_id: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM documents WHERE id = %s AND tenant_id = %s", (doc_id, tenant_id))


# ── Sessions ──────────────────────────────────────────────────────────────


def create_session(title: str) -> str:
    session_id = str(uuid4())
    now = int(time.time() * 1000)
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (%s, %s, %s, %s)",
                (session_id, title[:50], now, now),
            )
    return session_id


def get_sessions() -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT s.id, s.title, s.created_at, s.updated_at,
                       COUNT(m.id) AS message_count
                FROM sessions s
                LEFT JOIN session_messages m ON m.session_id = s.id
                GROUP BY s.id, s.title, s.created_at, s.updated_at
                ORDER BY s.updated_at DESC
            """)
            rows = cursor.fetchall()
    return [dict(r) for r in rows]


def get_session_messages(session_id: str) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, role, content, created_at FROM session_messages "
                "WHERE session_id = %s ORDER BY created_at ASC",
                (session_id,),
            )
            rows = cursor.fetchall()
    return [dict(r) for r in rows]


def add_session_message(session_id: str, role: str, content: str) -> None:
    now = int(time.time() * 1000)
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO session_messages (id, session_id, role, content, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (str(uuid4()), session_id, role, content, now),
            )
            cursor.execute(
                "UPDATE sessions SET updated_at = %s WHERE id = %s",
                (now, session_id),
            )


def delete_session_by_id(session_id: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
