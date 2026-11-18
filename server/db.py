"""SQLite: документы, фрагменты, журнал вопросов и аудит."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snippets (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    document_id TEXT NOT NULL,
    snippet_text TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE INDEX IF NOT EXISTS idx_snippets_document_id ON snippets(document_id);

CREATE TABLE IF NOT EXISTS qa_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    needs_review INTEGER NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS audit_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    action TEXT NOT NULL,
    input TEXT NOT NULL,
    output TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    duration_ms INTEGER NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def split_paragraphs(text: str) -> list[str]:
    """Режем по пустой строке. Пустые куски выбрасываем."""
    normalized = normalize_text(text)
    parts = re.split(r"\n\s*\n", normalized)
    snippets = [part.strip() for part in parts if part.strip()]
    if not snippets and normalized:
        snippets = [normalized]
    return snippets


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.isolation_level = None
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def create_document(self, title: str, text: str, raw_input: str, started: float) -> dict:
        doc_id = str(uuid.uuid4())
        now = utc_now()
        snippets = split_paragraphs(text)
        if not snippets:
            raise ValueError("document has no snippets")
        body = {"status": "ok", "document_id": doc_id}
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO documents (id, created_at, title, text) VALUES (?, ?, ?, ?)",
                (doc_id, now, title, normalize_text(text)),
            )
            for snippet in snippets:
                conn.execute(
                    "INSERT INTO snippets (id, created_at, document_id, snippet_text) VALUES (?, ?, ?, ?)",
                    (str(uuid.uuid4()), now, doc_id, snippet),
                )
            self._insert_audit(
                conn,
                action="POST /kb/documents",
                raw_input=raw_input,
                output=body,
                status="ok",
                error=None,
                duration_ms=_elapsed_ms(started),
            )
        return body

    def list_documents(self, started: float) -> list[dict]:
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at FROM documents ORDER BY created_at DESC, id DESC"
            ).fetchall()
            body = [
                {"id": row["id"], "title": row["title"], "created_at": row["created_at"]}
                for row in rows
            ]
            self._insert_audit(
                conn,
                action="GET /kb/documents",
                raw_input="{}",
                output=body,
                status="ok",
                error=None,
                duration_ms=_elapsed_ms(started),
            )
        return body

    def list_snippets(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.id AS snippet_id, s.document_id, s.snippet_text, d.title
                FROM snippets s
                JOIN documents d ON d.id = s.document_id
                ORDER BY s.rowid
                """
            ).fetchall()
        return [
            {
                "snippet_id": row["snippet_id"],
                "document_id": row["document_id"],
                "title": row["title"],
                "snippet_text": row["snippet_text"],
            }
            for row in rows
        ]

    def get_document(self, document_id: str, started: float) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, title, created_at, text FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        body = {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "text": row["text"],
        }
        self.record_audit(
            action="GET /kb/documents/{id}",
            raw_input=json.dumps({"id": document_id}, ensure_ascii=False),
            output=body,
            status="ok",
            error=None,
            duration_ms=_elapsed_ms(started),
        )
        return body

    def list_questions(self, *, only_review: bool, limit: int, started: float) -> list[dict]:
        with self._lock:
            rows = _select_questions(self._conn, only_review, limit)
        body = [_question_brief(row) for row in rows]
        self.record_audit(
            action="GET /kb/questions",
            raw_input=json.dumps({"only_review": only_review, "limit": limit}, ensure_ascii=False),
            output=body,
            status="ok",
            error=None,
            duration_ms=_elapsed_ms(started),
        )
        return body

    def get_question(self, question_id: str, started: float) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, created_at, question, answer, sources_json, needs_review, error
                FROM qa_runs WHERE id = ?
                """,
                (question_id,),
            ).fetchone()
        if row is None:
            return None
        body = _question_full(row)
        self.record_audit(
            action="GET /kb/questions/{id}",
            raw_input=json.dumps({"id": question_id}, ensure_ascii=False),
            output=body,
            status="ok",
            error=None,
            duration_ms=_elapsed_ms(started),
        )
        return body

    def export_snapshot(self, *, only_review: bool, started: float) -> dict:
        with self._lock:
            documents = self._conn.execute(
                "SELECT id, title, created_at, text FROM documents ORDER BY created_at DESC, id DESC"
            ).fetchall()
            questions = _select_questions(self._conn, only_review, None)
        body = {
            "exported_at": utc_now(),
            "only_review": only_review,
            "documents": [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "text": row["text"],
                }
                for row in documents
            ],
            "questions": [_question_full(row) for row in questions],
        }
        self.record_audit(
            action="GET /kb/export",
            raw_input=json.dumps({"only_review": only_review}, ensure_ascii=False),
            output={"documents": len(body["documents"]), "questions": len(body["questions"])},
            status="ok",
            error=None,
            duration_ms=_elapsed_ms(started),
        )
        return body

    def save_question(
        self,
        *,
        question: str,
        body: dict,
        error: str | None,
        raw_input: str,
        started: float,
    ) -> None:
        sources = body["sources"]
        needs_review = 1 if body["needs_review"] or not sources else 0
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO qa_runs
                    (id, created_at, question, answer, sources_json, needs_review, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    utc_now(),
                    question,
                    body["answer"],
                    json.dumps(sources, ensure_ascii=False),
                    needs_review,
                    error,
                ),
            )
            self._insert_audit(
                conn,
                action="POST /kb/ask",
                raw_input=raw_input,
                output=body,
                status="ok",
                error=error,
                duration_ms=_elapsed_ms(started),
            )

    def record_audit(
        self,
        *,
        action: str,
        raw_input: str,
        output: dict | list,
        status: str,
        error: str | None,
        duration_ms: int,
    ) -> None:
        with self.transaction() as conn:
            self._insert_audit(
                conn,
                action=action,
                raw_input=raw_input,
                output=output,
                status=status,
                error=error,
                duration_ms=duration_ms,
            )

    def _insert_audit(
        self,
        conn: sqlite3.Connection,
        *,
        action: str,
        raw_input: str,
        output: dict | list,
        status: str,
        error: str | None,
        duration_ms: int,
    ) -> None:
        conn.execute(
            """
            INSERT INTO audit_runs
                (id, created_at, action, input, output, status, error, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                utc_now(),
                action,
                raw_input,
                json.dumps(output, ensure_ascii=False),
                status,
                error,
                duration_ms,
            ),
        )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def _select_questions(conn: sqlite3.Connection, only_review: bool, limit: int | None):
    sql = """
        SELECT id, created_at, question, answer, sources_json, needs_review, error
        FROM qa_runs
    """
    params: list[object] = []
    if only_review:
        sql += " WHERE needs_review = 1"
    sql += " ORDER BY created_at DESC, id DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def _question_brief(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "question": row["question"],
        "needs_review": bool(row["needs_review"]),
    }


def _question_full(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "question": row["question"],
        "answer": row["answer"],
        "sources": _load_sources(row["sources_json"]),
        "needs_review": bool(row["needs_review"]),
        "error": row["error"],
    }


def _load_sources(raw: str | None) -> list:
    if not raw:
        return []
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []
