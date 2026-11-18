FROM python:3.13-slim

WORKDIR /app

COPY server ./server
COPY web ./web

RUN mkdir -p /app/data

ENV PYTHONUNBUFFERED=1 \
    KB_HOST=0.0.0.0 \
    KB_PORT=8080 \
    KB_DB=/app/data/kb.sqlite

EXPOSE 8080

CMD ["python", "-m", "server"]
