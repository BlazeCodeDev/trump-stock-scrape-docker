"""Database layer — SQLite with WAL mode, shared between web and monitor threads."""

import sqlite3
import json
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "/data/monitor.db")


def parse_published(s: str) -> float:
    """Parse a feed timestamp (RFC 822 RSS or ISO 8601) to epoch seconds; 0 if unknown."""
    if not s:
        return 0.0
    try:
        return parsedate_to_datetime(s).timestamp()   # "Sat, 30 May 2026 22:50:16 +0000"
    except Exception:
        pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()  # ISO 8601
    except Exception:
        return 0.0

DEFAULTS: dict[str, str] = {
    "feed_url":       "https://www.trumpstruth.org/feed",
    # Local AI model — bundled Ollama container (OpenAI-compatible endpoint).
    "model_url":      "http://ollama:11434/v1/chat/completions",
    "model_name":     "llama3.2:3b",
    "model_key":      "",   # optional bearer token (the bundled Ollama needs none)
    "check_interval": "300",
    "ntfy_url":       "",
    "ts_token":       "",
    "ts_account_id":  "107780257626128497",
}

# Env vars that pre-seed the DB on first boot (user can override in UI later)
_ENV_SEEDS = {
    "model_url":  "MODEL_URL",
    "model_name": "MODEL_NAME",
    "ntfy_url":   "NTFY_URL",
}

# Settings that must never be sent back to the browser
SECRET_KEYS = ("model_key", "ts_token")


def _conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS posts (
                id          TEXT PRIMARY KEY,
                text        TEXT,
                link        TEXT,
                published   TEXT,
                published_ts REAL NOT NULL DEFAULT 0,
                seen_at     TEXT NOT NULL,
                relevant    INTEGER NOT NULL DEFAULT 0,
                summary     TEXT,
                assets      TEXT DEFAULT '[]',
                direction   TEXT,
                tip         TEXT,
                urgency     TEXT,
                images      TEXT DEFAULT '[]',
                classified  INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        # Migrate older DBs that predate the `classified` column. Treat all
        # pre-existing rows as classified so they remain visible.
        cols = [r[1] for r in c.execute("PRAGMA table_info(posts)").fetchall()]
        if "classified" not in cols:
            c.execute("ALTER TABLE posts ADD COLUMN classified INTEGER NOT NULL DEFAULT 0")
            c.execute("UPDATE posts SET classified=1")
        if "published_ts" not in cols:
            c.execute("ALTER TABLE posts ADD COLUMN published_ts REAL NOT NULL DEFAULT 0")
            # Backfill from the stored raw `published` string so the high-water
            # mark is correct — otherwise old posts would look brand new.
            for row in c.execute("SELECT id, published FROM posts").fetchall():
                ts = parse_published(row["published"] or "")
                if ts:
                    c.execute("UPDATE posts SET published_ts=? WHERE id=?", (ts, row["id"]))
        if "images" not in cols:
            c.execute("ALTER TABLE posts ADD COLUMN images TEXT DEFAULT '[]'")
    # Seed from env vars only if not already stored
    for skey, ekey in _ENV_SEEDS.items():
        env_val = os.getenv(ekey, "")
        if env_val and not _raw_get(skey):
            set_setting(skey, env_val)


def _raw_get(key: str) -> str:
    c = _conn()
    row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    c.close()
    return row["value"] if row else ""


def get_setting(key: str) -> str:
    val = _raw_get(key)
    if val:
        return val
    if key in _ENV_SEEDS:
        return os.getenv(_ENV_SEEDS[key], "")
    return DEFAULTS.get(key, "")


def set_setting(key: str, value: str) -> None:
    c = _conn()
    c.execute("INSERT OR REPLACE INTO settings (key,value) VALUES (?,?)", (key, value))
    c.commit()
    c.close()


def get_all_settings() -> dict[str, str]:
    c = _conn()
    rows = c.execute("SELECT key,value FROM settings").fetchall()
    c.close()
    result = dict(DEFAULTS)
    for row in rows:
        result[row["key"]] = row["value"]
    for skey, ekey in _ENV_SEEDS.items():
        if not result.get(skey):
            result[skey] = os.getenv(ekey, "")
    return result


def is_seen(post_id: str) -> bool:
    c = _conn()
    row = c.execute("SELECT 1 FROM posts WHERE id=?", (post_id,)).fetchone()
    c.close()
    return row is not None


def count_posts() -> int:
    """Total rows (classified + baseline). 0 means we've never run before."""
    c = _conn()
    n = c.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    c.close()
    return int(n)


def max_published_ts() -> float:
    """Newest post timestamp we've already seen — the high-water mark for 'new'."""
    c = _conn()
    v = c.execute("SELECT MAX(published_ts) FROM posts").fetchone()[0]
    c.close()
    return float(v or 0)


def save_post(post: dict, classification: dict | None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    pts = float(post.get("published_ts") or 0)
    imgs = json.dumps(post.get("images") or [])
    c = _conn()
    if classification:
        c.execute(
            """INSERT OR IGNORE INTO posts
               (id,text,link,published,published_ts,seen_at,relevant,summary,assets,direction,tip,urgency,images,classified)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
            (
                post["id"], post["text"], post["link"], post.get("published", ""), pts, now,
                int(bool(classification.get("relevant"))),
                classification.get("summary", ""),
                json.dumps(classification.get("affected_assets", [])),
                classification.get("direction", "watch"),
                classification.get("tip", ""),
                classification.get("urgency", "low"),
                imgs,
            ),
        )
    else:
        c.execute(
            "INSERT OR IGNORE INTO posts (id,text,link,published,published_ts,seen_at,relevant,images) VALUES (?,?,?,?,?,?,0,?)",
            (post["id"], post["text"], post["link"], post.get("published", ""), pts, now, imgs),
        )
    c.commit()
    c.close()


def get_posts(limit: int = 50, relevant_only: bool = False) -> list[dict]:
    c = _conn()
    # Only show classified posts — baseline (seen-only) rows stay hidden.
    q = "SELECT * FROM posts WHERE classified=1"
    if relevant_only:
        q += " AND relevant=1"
    # Newest post first (by publish time), falling back to processing time.
    q += " ORDER BY published_ts DESC, seen_at DESC LIMIT ?"
    rows = c.execute(q, (limit,)).fetchall()
    c.close()
    result = []
    for row in rows:
        d = dict(row)
        for fld in ("assets", "images"):
            try:
                d[fld] = json.loads(d.get(fld) or "[]")
            except Exception:
                d[fld] = []
        result.append(d)
    return result
