FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu torch
RUN python -m pip install --no-cache-dir '.[ml]'

EXPOSE 8000
CMD ["uvicorn", "papertrail.api:app", "--host", "0.0.0.0", "--port", "8000"]
