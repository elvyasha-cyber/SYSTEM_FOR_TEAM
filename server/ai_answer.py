"""ИИ-операция: строгий JSON, проверка структуры и ручная проверка.

Черновик копирует предложения из context. Контроль качества проверяет любой черновик одинаково.
"""

from __future__ import annotations

from server.search import matched_words

INSUFFICIENT_ANSWER = "Данных недостаточно."
_CONFIDENCE = {"high", "medium", "low"}
_FIELDS = ("answer", "sources", "confidence", "needs_review")


def answer_with_sources(question: str, context: str) -> tuple[dict, str | None]:
    """Вернуть строгий JSON и причину ручной проверки, если она есть."""
    draft = draft_answer(question, context)
    return apply_quality_gate(draft, context)


def draft_answer(question: str, context: str) -> dict:
    """Черновик ответа только из предложений context."""
    picked = _pick_quotes(question, context)
    if not picked:
        return _insufficient()
    quotes = [quote for _, _, quote, _ in picked[:3]]
    confidence = "high" if picked[0][1] >= 3 else "medium"
    return {
        "answer": " ".join(quotes),
        "sources": [{"quote": quote} for quote in quotes],
        "confidence": confidence,
        "needs_review": False,
    }


def apply_quality_gate(candidate: object, context: str) -> tuple[dict, str | None]:
    """Привести ответ к схеме и включить ручную проверку, если опереться нельзя."""
    reasons: list[str] = []
    if not isinstance(candidate, dict):
        return _insufficient(), "структура ответа не объект"

    if set(candidate) - set(_FIELDS):
        reasons.append("лишние поля")
    if any(field not in candidate for field in _FIELDS):
        reasons.append("не хватает полей")

    answer = candidate.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        reasons.append("answer пустой")
        answer = INSUFFICIENT_ANSWER
    else:
        answer = answer.strip()

    confidence = candidate.get("confidence")
    if confidence not in _CONFIDENCE:
        reasons.append("confidence вне списка")
        confidence = "low"

    needs_review = candidate.get("needs_review")
    if not isinstance(needs_review, bool):
        reasons.append("needs_review не boolean")
        needs_review = True

    quotes, quote_reasons = _valid_quotes(candidate.get("sources"), context)
    reasons.extend(quote_reasons)

    if not quotes:
        reasons.append("нет опоры в контексте")
        return _insufficient(), _join(reasons)

    if _words_outside_context(answer, context):
        reasons.append("в ответе есть слова вне контекста")
        needs_review = True
        confidence = "low"

    if confidence == "low" or needs_review:
        reasons.append("нужна ручная проверка")
        return _reviewed(quotes), _join(reasons)

    return (
        {
            "answer": answer,
            "sources": [{"quote": quote} for quote in quotes],
            "confidence": confidence,
            "needs_review": False,
        },
        None,
    )


def _pick_quotes(question: str, context: str) -> list[tuple[int, int, str, list[str]]]:
    scored = [
        (sentence, matched)
        for sentence in _sentences(context)
        if (matched := matched_words(question, sentence))
    ]
    counts: dict[str, int] = {}
    for _, matched in scored:
        for word in set(matched):
            counts[word] = counts.get(word, 0) + 1
    rarest = min(counts.values()) if counts else 0
    ranked: list[tuple[int, int, str, list[str]]] = []
    for sentence, matched in scored:
        single_ok = len(matched) == 1 and counts[matched[0]] <= 3
        if len(matched) < 2 and not single_ok:
            continue
        covers_rarest = any(counts[word] == rarest for word in matched)
        ranked.append((int(covers_rarest), len(matched), sentence, matched))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    seen: set[str] = set()
    unique: list[tuple[int, int, str, list[str]]] = []
    for item in ranked:
        if item[2] in seen:
            continue
        seen.add(item[2])
        unique.append(item)
    return unique


def _sentences(context: str) -> list[str]:
    found: list[str] = []
    buffer: list[str] = []
    depth = 0
    for char in context:
        if char == "«":
            depth += 1
        elif char == "»" and depth:
            depth -= 1
        if char == "\n" and depth == 0:
            _flush(buffer, context, found)
            buffer = []
            continue
        buffer.append(char)
        if char in ".!?" and depth == 0:
            _flush(buffer, context, found)
            buffer = []
    _flush(buffer, context, found)
    return found


def _flush(buffer: list[str], context: str, found: list[str]) -> None:
    sentence = "".join(buffer).strip()
    if len(sentence) >= 20 and sentence in context and sentence not in found:
        found.append(sentence)


def _valid_quotes(sources: object, context: str) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    if not isinstance(sources, list):
        return [], ["sources не список"]
    quotes: list[str] = []
    for item in sources:
        if not isinstance(item, dict) or set(item) != {"quote"} or not isinstance(item.get("quote"), str):
            reasons.append("источник сломан")
            continue
        quote = item["quote"].strip()
        if not quote:
            reasons.append("пустая цитата")
            continue
        if quote not in context:
            reasons.append("цитата не из контекста")
            continue
        if quote not in quotes:
            quotes.append(quote)
    return quotes, reasons


def _words_outside_context(answer: str, context: str) -> bool:
    if answer == INSUFFICIENT_ANSWER:
        return False
    return any(word not in set(matched_words(answer, context)) for word in matched_words(answer, answer))


def _reviewed(quotes: list[str]) -> dict:
    return {
        "answer": INSUFFICIENT_ANSWER,
        "sources": [{"quote": quote} for quote in quotes],
        "confidence": "low",
        "needs_review": True,
    }


def _insufficient() -> dict:
    return {
        "answer": INSUFFICIENT_ANSWER,
        "sources": [],
        "confidence": "low",
        "needs_review": True,
    }


def _join(reasons: list[str]) -> str:
    unique: list[str] = []
    for reason in reasons:
        if reason not in unique:
            unique.append(reason)
    return "; ".join(unique)
