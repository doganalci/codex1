"""SQLite-backed persistence for runs, violations and evidence."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    method TEXT NOT NULL,
    status TEXT NOT NULL,        -- 'draft' | 'saved'
    prompt TEXT NOT NULL,
    llm_model TEXT NOT NULL,
    embedding_model TEXT,
    rag_collection TEXT,
    rag_documents TEXT,          -- json list
    finetune_model_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS violations (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    title TEXT,
    description TEXT NOT NULL,
    category TEXT,
    severity TEXT,
    threshold TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    violation_id TEXT NOT NULL,
    document TEXT,
    page TEXT,
    clause TEXT,
    snippet TEXT,
    FOREIGN KEY(violation_id) REFERENCES violations(id) ON DELETE CASCADE
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(settings.db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.executescript(SCHEMA)
        # idempotent migration for older DBs
        cols = {r["name"] for r in c.execute("PRAGMA table_info(runs)").fetchall()}
        if "finetune_model_id" not in cols:
            c.execute("ALTER TABLE runs ADD COLUMN finetune_model_id TEXT")


def now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def create_run(
    *,
    name: str,
    method: str,
    prompt: str,
    llm_model: str,
    embedding_model: str | None,
    rag_collection: str | None,
    rag_documents: list[str] | None,
    finetune_model_id: str | None = None,
) -> str:
    rid = str(uuid.uuid4())
    ts = now()
    with _conn() as c:
        c.execute(
            """INSERT INTO runs(id, name, method, status, prompt, llm_model,
               embedding_model, rag_collection, rag_documents, finetune_model_id,
               created_at, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rid,
                name,
                method,
                "draft",
                prompt,
                llm_model,
                embedding_model,
                rag_collection,
                json.dumps(rag_documents or []),
                finetune_model_id,
                ts,
                ts,
            ),
        )
    return rid


def add_violations(run_id: str, items: Iterable[dict]) -> int:
    count = 0
    ts = now()
    with _conn() as c:
        for it in items:
            vid = str(uuid.uuid4())
            c.execute(
                """INSERT INTO violations(id, run_id, title, description, category,
                   severity, threshold, created_at) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    vid,
                    run_id,
                    it.get("title"),
                    it.get("description", ""),
                    it.get("category"),
                    it.get("severity"),
                    it.get("threshold"),
                    ts,
                ),
            )
            for ev in it.get("evidence", []) or []:
                c.execute(
                    """INSERT INTO evidence(id, violation_id, document, page, clause, snippet)
                       VALUES(?,?,?,?,?,?)""",
                    (
                        str(uuid.uuid4()),
                        vid,
                        ev.get("document"),
                        str(ev.get("page")) if ev.get("page") is not None else None,
                        ev.get("clause"),
                        ev.get("snippet"),
                    ),
                )
            count += 1
        c.execute("UPDATE runs SET updated_at=? WHERE id=?", (ts, run_id))
    return count


def mark_saved(run_id: str) -> None:
    with _conn() as c:
        c.execute("UPDATE runs SET status='saved', updated_at=? WHERE id=?", (now(), run_id))


def delete_run(run_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM runs WHERE id=?", (run_id,))


def list_runs(only_saved: bool = False) -> list[dict]:
    q = "SELECT * FROM runs"
    if only_saved:
        q += " WHERE status='saved'"
    q += " ORDER BY datetime(updated_at) DESC"
    with _conn() as c:
        rows = c.execute(q).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return dict(r) if r else None


def get_violations(run_id: str) -> list[dict]:
    with _conn() as c:
        vrows = c.execute(
            "SELECT * FROM violations WHERE run_id=? ORDER BY created_at", (run_id,)
        ).fetchall()
        out: list[dict] = []
        for v in vrows:
            d = dict(v)
            ev = c.execute(
                "SELECT document,page,clause,snippet FROM evidence WHERE violation_id=?",
                (v["id"],),
            ).fetchall()
            d["evidence"] = [dict(e) for e in ev]
            out.append(d)
    return out


def count_violations(run_id: str) -> int:
    with _conn() as c:
        r = c.execute("SELECT COUNT(*) c FROM violations WHERE run_id=?", (run_id,)).fetchone()
    return int(r["c"])
