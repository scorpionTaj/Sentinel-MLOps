FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTINEL_STATE_DIR=/app/data/runtime

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"

RUN useradd --create-home --uid 10001 sentinel \
    && mkdir -p /app/data/runtime \
    && chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 8000
CMD ["uvicorn", "sentinel.api:app", "--host", "0.0.0.0", "--port", "8000"]

