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
    batch_no INTEGER NOT NULL DEFAULT 1,
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

CREATE TABLE IF NOT EXISTS ifc_models (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,             -- 'baseline' | 'violated'
    name TEXT NOT NULL,
    parent_id TEXT,                 -- violated -> baseline id
    llm_model TEXT NOT NULL,
    prompt TEXT,
    pool_run_id TEXT,               -- violated için kullanılan ihlal havuzu
    params_json TEXT,
    file_path TEXT NOT NULL,
    meta_path TEXT,
    labels_path TEXT,
    graph_path TEXT,
    status TEXT NOT NULL,           -- 'ok' | 'invalid' | 'partial'
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_usage (
    id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,        -- gen_naive|gen_optimized|gen_rag|gen_finetuned|embed|ifc_gen|ifc_inject|ft_prep
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    pool_run_id TEXT,
    ifc_model_id TEXT,
    collection TEXT,
    note TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ifc_violation_labels (
    id TEXT PRIMARY KEY,
    ifc_model_id TEXT NOT NULL,
    violation_id TEXT,              -- havuzdaki violation id
    title TEXT,
    category TEXT,
    severity TEXT,
    threshold TEXT,
    ifc_global_id TEXT,
    ifc_type TEXT,
    ifc_name TEXT,
    attribute TEXT,
    value_before TEXT,
    value_after TEXT,
    evidence_json TEXT,
    status TEXT NOT NULL,           -- 'applied' | 'skipped'
    reason TEXT,
    applied_at TEXT NOT NULL,
    FOREIGN KEY(ifc_model_id) REFERENCES ifc_models(id) ON DELETE CASCADE
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
        vcols = {r["name"] for r in c.execute("PRAGMA table_info(violations)").fetchall()}
        if "batch_no" not in vcols:
            c.execute("ALTER TABLE violations ADD COLUMN batch_no INTEGER NOT NULL DEFAULT 1")
        icols = {r["name"] for r in c.execute("PRAGMA table_info(ifc_models)").fetchall()}
        if icols and "graph_path" not in icols:
            c.execute("ALTER TABLE ifc_models ADD COLUMN graph_path TEXT")


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


def next_batch_no(run_id: str) -> int:
    with _conn() as c:
        r = c.execute(
            "SELECT COALESCE(MAX(batch_no),0)+1 n FROM violations WHERE run_id=?",
            (run_id,),
        ).fetchone()
    return int(r["n"])


def add_violations(run_id: str, items: Iterable[dict], batch_no: int | None = None) -> int:
    count = 0
    ts = now()
    if batch_no is None:
        batch_no = next_batch_no(run_id)
    with _conn() as c:
        for it in items:
            vid = str(uuid.uuid4())
            c.execute(
                """INSERT INTO violations(id, run_id, title, description, category,
                   severity, threshold, batch_no, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    vid,
                    run_id,
                    it.get("title"),
                    it.get("description", ""),
                    it.get("category"),
                    it.get("severity"),
                    it.get("threshold"),
                    batch_no,
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


def delete_batch(run_id: str, batch_no: int) -> int:
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM violations WHERE run_id=? AND batch_no=?",
            (run_id, batch_no),
        )
        c.execute("UPDATE runs SET updated_at=? WHERE id=?", (now(), run_id))
        return cur.rowcount


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


# ---------- IFC models ----------
def create_ifc_model(
    *,
    kind: str,
    name: str,
    parent_id: str | None,
    llm_model: str,
    prompt: str | None,
    pool_run_id: str | None,
    params: dict | None,
    file_path: str,
    meta_path: str | None,
    labels_path: str | None,
    status: str,
    error: str | None = None,
    graph_path: str | None = None,
    id: str | None = None,
) -> str:
    mid = id or str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            """INSERT INTO ifc_models(id, kind, name, parent_id, llm_model, prompt,
               pool_run_id, params_json, file_path, meta_path, labels_path,
               graph_path, status, error, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                mid, kind, name, parent_id, llm_model, prompt, pool_run_id,
                json.dumps(params or {}, ensure_ascii=False),
                file_path, meta_path, labels_path, graph_path,
                status, error, now(),
            ),
        )
    return mid


def set_ifc_graph_path(ifc_id: str, graph_path: str) -> None:
    with _conn() as c:
        c.execute("UPDATE ifc_models SET graph_path=? WHERE id=?", (graph_path, ifc_id))


def list_ifc_models(kind: str | None = None) -> list[dict]:
    q = "SELECT * FROM ifc_models"
    args: tuple = ()
    if kind:
        q += " WHERE kind=?"
        args = (kind,)
    q += " ORDER BY datetime(created_at) DESC"
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    return [dict(r) for r in rows]


def get_ifc_model(ifc_id: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM ifc_models WHERE id=?", (ifc_id,)).fetchone()
    return dict(r) if r else None


def delete_ifc_model(ifc_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM ifc_models WHERE id=?", (ifc_id,))


def add_ifc_labels(ifc_model_id: str, labels: Iterable[dict]) -> int:
    n = 0
    ts = now()
    with _conn() as c:
        for lab in labels:
            c.execute(
                """INSERT INTO ifc_violation_labels(id, ifc_model_id, violation_id,
                   title, category, severity, threshold, ifc_global_id, ifc_type,
                   ifc_name, attribute, value_before, value_after, evidence_json,
                   status, reason, applied_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()), ifc_model_id, lab.get("violation_id"),
                    lab.get("title"), lab.get("category"), lab.get("severity"),
                    lab.get("threshold"), lab.get("ifc_global_id"),
                    lab.get("ifc_type"), lab.get("ifc_name"),
                    lab.get("attribute"),
                    str(lab.get("value_before")) if lab.get("value_before") is not None else None,
                    str(lab.get("value_after")) if lab.get("value_after") is not None else None,
                    json.dumps(lab.get("evidence") or [], ensure_ascii=False),
                    lab.get("status", "applied"), lab.get("reason"), ts,
                ),
            )
            n += 1
    return n


def get_ifc_labels(ifc_model_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM ifc_violation_labels WHERE ifc_model_id=? ORDER BY applied_at",
            (ifc_model_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------- LLM token kullanımı ----------
def record_usage(
    *,
    operation: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
    pool_run_id: str | None = None,
    ifc_model_id: str | None = None,
    collection: str | None = None,
    note: str | None = None,
) -> None:
    if total_tokens is None:
        total_tokens = int(prompt_tokens) + int(completion_tokens)
    with _conn() as c:
        c.execute(
            """INSERT INTO llm_usage(id, operation, model, prompt_tokens,
               completion_tokens, total_tokens, pool_run_id, ifc_model_id,
               collection, note, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()), operation, model,
                int(prompt_tokens or 0), int(completion_tokens or 0),
                int(total_tokens or 0),
                pool_run_id, ifc_model_id, collection, note, now(),
            ),
        )


def record_usage_from_openai(usage_obj, *, operation: str, model: str,
                             **refs) -> None:
    """OpenAI response.usage objesinden veya dict'inden kaydeder."""
    if usage_obj is None:
        return
    if hasattr(usage_obj, "prompt_tokens"):
        pt = getattr(usage_obj, "prompt_tokens", 0) or 0
        ct = getattr(usage_obj, "completion_tokens", 0) or 0
        tt = getattr(usage_obj, "total_tokens", None)
    else:
        pt = usage_obj.get("prompt_tokens", 0) or 0
        ct = usage_obj.get("completion_tokens", 0) or 0
        tt = usage_obj.get("total_tokens")
    record_usage(operation=operation, model=model,
                 prompt_tokens=pt, completion_tokens=ct, total_tokens=tt,
                 **refs)


def list_usage(
    *,
    pool_run_id: str | None = None,
    ifc_model_id: str | None = None,
    collection: str | None = None,
    operation: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    where, args = [], []
    if pool_run_id:
        where.append("pool_run_id=?"); args.append(pool_run_id)
    if ifc_model_id:
        where.append("ifc_model_id=?"); args.append(ifc_model_id)
    if collection:
        where.append("collection=?"); args.append(collection)
    if operation:
        where.append("operation=?"); args.append(operation)
    q = "SELECT * FROM llm_usage"
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY datetime(created_at) DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    with _conn() as c:
        rows = c.execute(q, tuple(args)).fetchall()
    return [dict(r) for r in rows]


def usage_totals(**filters) -> dict:
    rows = list_usage(**filters)
    return {
        "calls": len(rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "total_tokens": sum(r["total_tokens"] for r in rows),
    }
