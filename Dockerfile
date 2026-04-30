FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --upgrade pip

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY migrate ./migrate

RUN pip install -e ".[dev,trace]"

EXPOSE 8000

CMD ["uvicorn", "camillo.main:app", "--host", "0.0.0.0", "--port", "8000"]
