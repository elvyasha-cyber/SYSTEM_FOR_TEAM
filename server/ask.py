"""Вопрос к базе: поиск фрагментов, ИИ-операция и источники с document_id."""

from __future__ import annotations

from server.ai_answer import INSUFFICIENT_ANSWER, answer_with_sources
from server.db import Database
from server.search import search_snippets

_CONTEXT_MAX = 20_000


def ask(db: Database, question: str) -> tuple[dict, str | None]:
    """Собрать ответ по базе. Причина ручной проверки уходит в журнал, не в HTTP."""
    hits = search_snippets(db, question)
    if not hits:
        return _insufficient(), "нет фрагментов"
    context, used = _context(hits)
    ai_body, reason = answer_with_sources(question, context)
    sources = _with_documents(ai_body.get("sources", []), used)
    needs_review = bool(ai_body.get("needs_review")) or ai_body.get("confidence") == "low" or not sources
    if needs_review:
        return (
            {
                "answer": INSUFFICIENT_ANSWER,
                "sources": sources,
                "needs_review": True,
            },
            reason or "нет опоры в контексте",
        )
    return (
        {
            "answer": ai_body["answer"],
            "sources": sources,
            "needs_review": False,
        },
        None,
    )


def _context(hits: list[dict]) -> tuple[str, list[dict]]:
    parts: list[str] = []
    used: list[dict] = []
    size = 0
    for hit in hits:
        text = hit["snippet_text"]
        extra = len(text) + (2 if parts else 0)
        if parts and size + extra > _CONTEXT_MAX:
            break
        if not parts and len(text) > _CONTEXT_MAX:
            text = text[:_CONTEXT_MAX]
        parts.append(text)
        used.append({**hit, "snippet_text": text})
        size += len(text) + (2 if len(parts) > 1 else 0)
    return "\n\n".join(parts), used


def _with_documents(sources: list, hits: list[dict]) -> list[dict]:
    attached: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        quote = source.get("quote")
        if not isinstance(quote, str) or not quote:
            continue
        for hit in hits:
            if quote not in hit["snippet_text"]:
                continue
            key = (hit["document_id"], quote)
            if key in seen:
                break
            seen.add(key)
            attached.append({"document_id": hit["document_id"], "quote": quote})
            break
    return attached


def _insufficient() -> dict:
    return {
        "answer": INSUFFICIENT_ANSWER,
        "sources": [],
        "needs_review": True,
    }
