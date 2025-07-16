import aiosqlite
from typing import List, Dict
from pathlib import Path

# ── paths & constants ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # …/chatbot-dev/chatbot-dev
DB_PATH  = BASE_DIR / "chat_history.db"
MAX_ROWS = 10                                              # keep only last‑10 per user

# ── table schema (now includes user_id) ───────────────────────────
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS questions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  TEXT NOT NULL,
    query    TEXT,
    created  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# ── helper: open conn & ensure table ──────────────────────────────
async def _get_conn():
    conn = await aiosqlite.connect(DB_PATH)
    await conn.execute(CREATE_SQL)
    return conn

# ── save a question for a specific user ───────────────────────────
async def save_query(user_id: str, query: str):
    conn = await _get_conn()
    # insert
    await conn.execute(
        "INSERT INTO questions (user_id, query) VALUES (?, ?)",
        (user_id, query),
    )
    # keep only the most‑recent 10 rows for that user
    await conn.execute(
        """
        DELETE FROM questions
        WHERE id NOT IN (
            SELECT id FROM questions
            WHERE user_id = ?
            ORDER BY created DESC
            LIMIT ?
        )
        AND user_id = ?;
        """,
        (user_id, MAX_ROWS, user_id),
    )
    await conn.commit()
    await conn.close()

# ── fetch recent questions for a specific user ────────────────────
async def get_recent(user_id: str, limit: int = MAX_ROWS) -> List[Dict[str, str]]:
    conn = await _get_conn()
    cur = await conn.execute(
        """
        SELECT query, created
        FROM questions
        WHERE user_id = ?
        ORDER BY created DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = await cur.fetchall()
    await conn.close()
    return [{"query": r[0], "created": r[1]} for r in rows]
