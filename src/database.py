"""PostgreSQL bağlantısı ve tablo şemaları (CNPG). psycopg3 + Faz 1-3 ile uyumlu."""
import os
import json
import psycopg
from psycopg.rows import dict_row
from contextlib import contextmanager

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "medrag-postgres-rw.medrag.svc.cluster.local"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "app"),
    "user": os.getenv("DB_USER", "app"),
    "password": os.getenv("DB_PASSWORD", ""),
}

@contextmanager
def get_conn():
    conn = psycopg.connect(**DB_CONFIG, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

DDL = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(64) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE SET NULL,
    username VARCHAR(64),
    question TEXT NOT NULL,
    answer TEXT,
    sources JSONB,
    classifier_score FLOAT,
    classifier_label VARCHAR(32),
    latency_ms INT,
    model VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_query_logs_user ON query_logs(user_id, created_at DESC);
"""

def init_db():
    with get_conn() as c, c.cursor() as cur:
        cur.execute(DDL)

def insert_user(username, pw_hash):
    with get_conn() as c, c.cursor() as cur:
        cur.execute("INSERT INTO users(username, password_hash) VALUES (%s, %s) RETURNING id", (username, pw_hash))
        return cur.fetchone()["id"]

def get_user(username):
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        return cur.fetchone()

def log_query(user_id, username, question, answer, sources, score, label, latency_ms, model):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            """INSERT INTO query_logs (user_id, username, question, answer, sources,
               classifier_score, classifier_label, latency_ms, model)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (user_id, username, question, answer, json.dumps(sources), score, label, latency_ms, model))

def fetch_history(user_id, limit=20, offset=0):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            """SELECT id, question, answer, sources, classifier_score, classifier_label,
               latency_ms, created_at FROM query_logs WHERE user_id=%s
               ORDER BY created_at DESC LIMIT %s OFFSET %s""",
            (user_id, limit, offset))
        return cur.fetchall()
