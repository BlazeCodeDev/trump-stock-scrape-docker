FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY db.py monitor.py web.py ollama_client.py main.py ./
COPY templates ./templates

# Persist SQLite DB in a volume
VOLUME ["/data"]
ENV DB_PATH=/data/monitor.db
ENV PORT=1122

EXPOSE 1122

# Non-root user (owns /data so SQLite can write)
RUN useradd -m appuser && mkdir -p /data && chown appuser /data
USER appuser

CMD ["python", "main.py"]
