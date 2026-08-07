FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_NO_CACHE=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

USER root
RUN apt-get update \
    && apt-get install --no-install-recommends --yes git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10000 camillo \
    && useradd --uid 10000 --gid 10000 --home-dir /app --no-create-home --shell /usr/sbin/nologin camillo \
    && chown camillo:camillo /app

USER camillo

COPY --chown=camillo:camillo pyproject.toml uv.lock README.md ./
ARG CAMILLO_DEV=false
RUN if [ "$CAMILLO_DEV" = "true" ]; then uv sync --locked --extra trace --extra dev --no-install-project; else uv sync --locked --no-dev --extra trace --no-install-project; fi

COPY --chown=camillo:camillo src ./src
COPY --chown=camillo:camillo alembic.ini ./
COPY --chown=camillo:camillo migrate ./migrate

RUN if [ "$CAMILLO_DEV" = "true" ]; then uv sync --locked --extra trace --extra dev; else uv sync --locked --no-dev --extra trace; fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "camillo.main:app", "--host", "0.0.0.0", "--port", "8000"]
