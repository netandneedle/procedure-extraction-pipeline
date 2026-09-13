FROM python:3.12-slim

WORKDIR /app

# libpq-dev: required by psycopg (pulled in by langgraph-checkpoint-postgres)
# gcc: needed to compile psycopg C extension
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq-dev gcc libxcb1 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
