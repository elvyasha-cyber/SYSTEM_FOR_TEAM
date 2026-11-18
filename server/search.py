"""Поиск фрагментов по словам вопроса."""

from __future__ import annotations

import math
import re

from server.db import Database

_WORD = re.compile(r"[0-9a-zа-я]+")
_MIN_MATCHES = 2
_ENDINGS = (
    "иями",
    "ями",
    "ами",
    "ого",
    "его",
    "ему",
    "ому",
    "ей",
    "ых",
    "их",
    "ыми",
    "ими",
    "ает",
    "яет",
    "ует",
    "ают",
    "яют",
    "уют",
    "ить",
    "ать",
    "ять",
    "еть",
    "ешь",
    "ишь",
    "ила",
    "ыла",
    "ала",
    "ела",
    "ий",
    "ый",
    "ой",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ов",
    "ев",
    "ам",
    "ям",
    "ах",
    "ях",
    "ом",
    "ем",
    "ия",
    "ья",
    "а",
    "я",
    "ы",
    "и",
    "у",
    "ю",
    "е",
    "о",
)
_STOP = {
    "и",
    "в",
    "во",
    "на",
    "не",
    "что",
    "как",
    "какой",
    "какая",
    "какое",
    "какие",
    "какому",
    "каком",
    "кто",
    "где",
    "когда",
    "к",
    "ко",
    "по",
    "из",
    "за",
    "от",
    "до",
    "для",
    "о",
    "об",
    "это",
    "этого",
    "этот",
    "эта",
    "эти",
    "то",
    "а",
    "но",
    "или",
    "если",
    "ли",
    "же",
    "бы",
    "уже",
    "так",
    "да",
    "нет",
    "при",
    "со",
    "с",
    "у",
    "я",
    "мы",
    "вы",
    "он",
    "она",
    "они",
    "его",
    "ее",
    "их",
    "чтобы",
    "такое",
    "такой",
    "такая",
    "без",
    "только",
    "есть",
    "быть",
    "должен",
    "должна",
    "нужно",
    "можно",
    "чем",
}


def search_snippets(db: Database, question: str, limit: int = 5) -> list[dict]:
    """Найти абзацы по словам вопроса.

    Обычный порог — два общих слова. Слово, которое есть не больше чем
    в двух абзацах, тоже проходит: иначе термин вроде «фрагмент» теряется.
    """
    query = _terms(question)
    if not query or limit < 1:
        return []
    snippets = db.list_snippets()
    prepared = [(item, _terms(item["snippet_text"])) for item in snippets]
    stems = [word for _, word in query]
    weights, doc_freq = _weights(stems, [dict(words) for _, words in prepared])
    multi: list[tuple[float, int, dict]] = []
    rare: list[tuple[float, int, dict]] = []
    for item, words in prepared:
        word_stems = dict(words)
        matched = [(surface, stem) for surface, stem in query if _in_snippet(stem, word_stems)]
        if not matched:
            continue
        score = sum(weights[stem] for _, stem in matched)
        hit = {
            "snippet_id": item["snippet_id"],
            "document_id": item["document_id"],
            "title": item["title"],
            "snippet_text": item["snippet_text"],
            "matched": [surface for surface, _ in matched],
        }
        row = (score, len(matched), hit)
        rare_word = len(matched) == 1 and 0 < doc_freq[matched[0][1]] <= 2
        if len(matched) >= _MIN_MATCHES:
            multi.append(row)
        elif rare_word:
            rare.append(row)
    multi.sort(key=lambda row: (row[0], row[1]), reverse=True)
    rare.sort(key=lambda row: (row[0], row[1]), reverse=True)
    room = max(limit - len(rare), 0)
    chosen = multi[:room] + rare
    chosen.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [hit for _, _, hit in chosen[:limit]]


def matched_words(question: str, text: str) -> list[str]:
    """Слова вопроса, которые нашлись в тексте."""
    query = _terms(question)
    words = dict(_terms(text))
    return [surface for surface, stem in query if _in_snippet(stem, words)]


def _terms(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in _WORD.findall(text.lower().replace("ё", "е")):
        if raw in _STOP or len(raw) < 3:
            continue
        stem = _stem(raw)
        if len(stem) < 3 or stem in seen:
            continue
        seen.add(stem)
        found.append((raw, stem))
    return found


def _stem(word: str) -> str:
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            return word[: -len(ending)]
    return word


def _in_snippet(query_stem: str, snippet_words: dict[str, str]) -> bool:
    return any(_same(query_stem, stem) for stem in snippet_words.values())


def _same(left: str, right: str) -> bool:
    if left == right:
        return True
    if len(left) < 5 or len(right) < 5 or len(left) != len(right) or left[:-1] != right[:-1]:
        return False
    # ответить / отвечает → ответ / отвеч
    return {left[-1], right[-1]} <= {"т", "ч"} or {left[-1], right[-1]} <= {"д", "ж"}


def _weights(query: list[str], corpus: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, int]]:
    total = len(corpus) or 1
    weights: dict[str, float] = {}
    doc_freq: dict[str, int] = {}
    for word in query:
        docs = sum(1 for stems in corpus if _in_snippet(word, stems))
        doc_freq[word] = docs
        weights[word] = math.log((total + 1) / (docs + 1)) + 1
    return weights, doc_freq
