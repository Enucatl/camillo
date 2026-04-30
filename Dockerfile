FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_NO_CACHE=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN groupadd --system camillo \
    && useradd --system --gid camillo --home-dir /app --shell /usr/sbin/nologin camillo \
    && chown camillo:camillo /app

USER camillo

COPY --chown=camillo:camillo pyproject.toml uv.lock README.md ./
COPY --chown=camillo:camillo src ./src
COPY --chown=camillo:camillo alembic.ini ./
COPY --chown=camillo:camillo migrate ./migrate

RUN uv sync --locked --extra dev --extra trace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "camillo.main:app", "--host", "0.0.0.0", "--port", "8000"]
