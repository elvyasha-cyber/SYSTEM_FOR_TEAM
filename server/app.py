"""Команды базы знаний: документы, вопрос и ИИ-операция."""

from __future__ import annotations

import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from server.ai_answer import answer_with_sources
from server.ask import ask
from server.db import Database, normalize_text

# В ТЗ предел длины назван, числа нет. Эти границы действуют, пока их не заменят.
TITLE_MAX = 200
TEXT_MAX = 20_000
QUESTION_MAX = 2_000
CONTEXT_MAX = 20_000
BODY_MAX = 1_000_000
ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = "/kb/documents"
ASK_PATH = "/kb/ask"
AI_PATH = "/ai/answer_with_sources"


class RequestError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)

    def body(self) -> dict:
        return {"error": self.code, "message": self.message}


class KBServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], db: Database):
        self.db = db
        super().__init__(server_address, Handler)

    def server_close(self) -> None:
        super().server_close()
        self.db.close()


class Handler(BaseHTTPRequestHandler):
    server: KBServer

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def log_message(self, fmt: str, *args) -> None:
        logging.info("%s %s", self.address_string(), fmt % args)

    def _dispatch(self, method: str) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if method == "GET" and path in {"/", "/index.html"}:
            self._panel()
            return
        if method == "GET" and path == "/kb/export":
            self._export()
            return
        parts = [part for part in path.split("/") if part]
        if method == "GET" and parts == ["kb", "questions"]:
            self._list_questions()
            return
        if method == "GET" and len(parts) == 3 and parts[:2] == ["kb", "documents"]:
            self._open_document(parts[2])
            return
        if method == "GET" and len(parts) == 3 and parts[:2] == ["kb", "questions"]:
            self._open_question(parts[2])
            return
        routes = {
            DOCUMENTS_PATH: {"GET": self._list_documents, "POST": self._create_document},
            ASK_PATH: {"POST": self._ask},
            AI_PATH: {"POST": self._answer_with_sources},
        }
        handlers = routes.get(path)
        if handlers is None:
            self._send(404, {"error": "not_found", "message": "команда не найдена"})
            return
        handler = handlers.get(method)
        if handler is None:
            self._fail(
                f"{method} {path}",
                "{}",
                RequestError(405, "method_not_allowed", "метод не поддерживается"),
                time.perf_counter(),
            )
            return
        handler()

    def _list_documents(self) -> None:
        started = time.perf_counter()
        try:
            body = self.server.db.list_documents(started)
        except Exception:
            logging.exception("list documents failed")
            self._fail(
                f"GET {DOCUMENTS_PATH}",
                "{}",
                RequestError(500, "internal_error", "внутренняя ошибка"),
                started,
            )
            return
        self._send(200, body)

    def _open_document(self, document_id: str) -> None:
        started = time.perf_counter()
        try:
            body = self.server.db.get_document(document_id, started)
        except Exception:
            logging.exception("open document failed")
            self._fail(
                "GET /kb/documents/{id}",
                json.dumps({"id": document_id}, ensure_ascii=False),
                RequestError(500, "internal_error", "внутренняя ошибка"),
                started,
            )
            return
        if body is None:
            self._fail(
                "GET /kb/documents/{id}",
                json.dumps({"id": document_id}, ensure_ascii=False),
                RequestError(404, "not_found", "документ не найден"),
                started,
            )
            return
        self._send(200, body)

    def _list_questions(self) -> None:
        started = time.perf_counter()
        try:
            body = self.server.db.list_questions(
                only_review=self._only_review(),
                limit=self._limit(),
                started=started,
            )
        except Exception:
            logging.exception("list questions failed")
            self._fail(
                "GET /kb/questions",
                "{}",
                RequestError(500, "internal_error", "внутренняя ошибка"),
                started,
            )
            return
        self._send(200, body)

    def _open_question(self, question_id: str) -> None:
        started = time.perf_counter()
        try:
            body = self.server.db.get_question(question_id, started)
        except Exception:
            logging.exception("open question failed")
            self._fail(
                "GET /kb/questions/{id}",
                json.dumps({"id": question_id}, ensure_ascii=False),
                RequestError(500, "internal_error", "внутренняя ошибка"),
                started,
            )
            return
        if body is None:
            self._fail(
                "GET /kb/questions/{id}",
                json.dumps({"id": question_id}, ensure_ascii=False),
                RequestError(404, "not_found", "вопрос не найден"),
                started,
            )
            return
        self._send(200, body)

    def _export(self) -> None:
        started = time.perf_counter()
        try:
            body = self.server.db.export_snapshot(only_review=self._only_review(), started=started)
        except Exception:
            logging.exception("export failed")
            self._fail(
                "GET /kb/export",
                "{}",
                RequestError(500, "internal_error", "внутренняя ошибка"),
                started,
            )
            return
        filename = "kb-export-review.json" if body["only_review"] else "kb-export.json"
        payload = json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _panel(self) -> None:
        file_path = ROOT / "web" / "index.html"
        if not file_path.is_file():
            self._send(500, {"error": "panel_missing", "message": "панель не найдена"})
            return
        payload = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _only_review(self) -> bool:
        values = parse_qs(urlparse(self.path).query).get("needs_review", [])
        return any(value.lower() in {"1", "true", "yes"} for value in values)

    def _limit(self) -> int:
        raw = parse_qs(urlparse(self.path).query).get("limit", ["100"])
        try:
            value = int(raw[0])
        except (TypeError, ValueError):
            return 100
        return min(100, max(1, value))

    def _create_document(self) -> None:
        started = time.perf_counter()
        raw = ""
        try:
            raw = self._read_json_text()
            title, text = parse_document(raw)
            body = self.server.db.create_document(title, text, raw, started)
        except RequestError as exc:
            self._fail(f"POST {DOCUMENTS_PATH}", raw, exc, started)
            return
        except Exception:
            logging.exception("create document failed")
            self._fail(
                f"POST {DOCUMENTS_PATH}",
                raw,
                RequestError(500, "internal_error", "внутренняя ошибка"),
                started,
            )
            return
        self._send(200, body)

    def _ask(self) -> None:
        started = time.perf_counter()
        raw = ""
        try:
            raw = self._read_json_text()
            question = parse_question(raw)
            body, review_reason = ask(self.server.db, question)
            self.server.db.save_question(
                question=question,
                body=body,
                error=review_reason,
                raw_input=raw,
                started=started,
            )
        except RequestError as exc:
            self._fail(f"POST {ASK_PATH}", raw, exc, started)
            return
        except Exception:
            logging.exception("ask failed")
            self._fail(
                f"POST {ASK_PATH}",
                raw,
                RequestError(500, "internal_error", "внутренняя ошибка"),
                started,
            )
            return
        self._send(200, body)

    def _answer_with_sources(self) -> None:
        started = time.perf_counter()
        raw = ""
        try:
            raw = self._read_json_text()
            question, context = parse_ai_request(raw)
            body, review_reason = answer_with_sources(question, context)
            self.server.db.record_audit(
                action=f"POST {AI_PATH}",
                raw_input=raw,
                output=body,
                status="ok",
                error=review_reason,
                duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
            )
        except RequestError as exc:
            self._fail(f"POST {AI_PATH}", raw, exc, started)
            return
        except Exception:
            logging.exception("answer with sources failed")
            self._fail(
                f"POST {AI_PATH}",
                raw,
                RequestError(500, "internal_error", "внутренняя ошибка"),
                started,
            )
            return
        self._send(200, body)

    def _read_json_text(self) -> str:
        if "Content-Length" not in self.headers:
            raise RequestError(400, "invalid_body", "нужен заголовок Content-Length")
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type.lower():
            raise RequestError(400, "invalid_body", "нужен Content-Type: application/json")
        try:
            length = int(self.headers["Content-Length"])
        except ValueError as exc:
            raise RequestError(400, "invalid_body", "некорректный Content-Length") from exc
        if length < 0 or length > BODY_MAX:
            raise RequestError(400, "body_too_large", "тело запроса слишком большое")
        raw_bytes = self.rfile.read(length)
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RequestError(400, "invalid_json", "тело не в UTF-8") from exc

    def _fail(self, action: str, raw_input: str, exc: RequestError, started: float) -> None:
        body = exc.body()
        try:
            self.server.db.record_audit(
                action=action,
                raw_input=raw_input,
                output=body,
                status="error",
                error=exc.message,
                duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
            )
        except Exception:
            logging.exception("audit failed")
        self._send(exc.status_code, body)

    def _send(self, status_code: int, body: dict | list) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def parse_document(raw: str) -> tuple[str, str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RequestError(400, "invalid_json", "тело не является JSON") from exc
    if not isinstance(data, dict):
        raise RequestError(400, "invalid_body", "ожидается объект JSON")
    extra = sorted(set(data) - {"title", "text"})
    if extra:
        raise RequestError(422, "unknown_field", "лишние поля: " + ", ".join(extra))
    if "title" not in data or "text" not in data:
        raise RequestError(422, "missing_field", "нужны поля title и text")
    title = data["title"]
    text = data["text"]
    if not isinstance(title, str) or not isinstance(text, str):
        raise RequestError(422, "invalid_type", "title и text должны быть строками")
    title = title.strip()
    text = normalize_text(text)
    if not title or not text:
        raise RequestError(422, "empty_field", "title и text не могут быть пустыми")
    if len(title) > TITLE_MAX or len(text) > TEXT_MAX:
        raise RequestError(
            422,
            "too_long",
            f"title до {TITLE_MAX} символов, text до {TEXT_MAX}",
        )
    return title, text


def parse_question(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RequestError(400, "invalid_json", "тело не является JSON") from exc
    if not isinstance(data, dict):
        raise RequestError(400, "invalid_body", "ожидается объект JSON")
    extra = sorted(set(data) - {"question"})
    if extra:
        raise RequestError(422, "unknown_field", "лишние поля: " + ", ".join(extra))
    if "question" not in data:
        raise RequestError(422, "missing_field", "нужно поле question")
    question = data["question"]
    if not isinstance(question, str):
        raise RequestError(422, "invalid_type", "question должен быть строкой")
    question = question.strip()
    if not question:
        raise RequestError(422, "empty_field", "question не может быть пустым")
    if len(question) > QUESTION_MAX:
        raise RequestError(422, "too_long", f"question до {QUESTION_MAX} символов")
    return question


def parse_ai_request(raw: str) -> tuple[str, str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RequestError(400, "invalid_json", "тело не является JSON") from exc
    if not isinstance(data, dict):
        raise RequestError(400, "invalid_body", "ожидается объект JSON")
    extra = sorted(set(data) - {"question", "context"})
    if extra:
        raise RequestError(422, "unknown_field", "лишние поля: " + ", ".join(extra))
    if "question" not in data or "context" not in data:
        raise RequestError(422, "missing_field", "нужны поля question и context")
    question = data["question"]
    context = data["context"]
    if not isinstance(question, str) or not isinstance(context, str):
        raise RequestError(422, "invalid_type", "question и context должны быть строками")
    question = question.strip()
    context = normalize_text(context)
    if not question or not context:
        raise RequestError(422, "empty_field", "question и context не могут быть пустыми")
    if len(question) > QUESTION_MAX or len(context) > CONTEXT_MAX:
        raise RequestError(
            422,
            "too_long",
            f"question до {QUESTION_MAX} символов, context до {CONTEXT_MAX}",
        )
    return question, context


def serve(host: str, port: int, db_path: Path) -> KBServer:
    return KBServer((host, port), Database(db_path))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    host = os.environ.get("KB_HOST", "127.0.0.1")
    port = int(os.environ.get("KB_PORT", "8080"))
    db_path = Path(os.environ.get("KB_DB", str(ROOT / "data" / "kb.sqlite")))
    server = serve(host, port, db_path)
    logging.info("база знаний слушает http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("остановка")
    finally:
        server.server_close()
