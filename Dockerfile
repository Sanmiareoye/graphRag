FROM python:3.12.1-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY rag-project/graph_retrieval.py .
COPY rag-project/main.py .
COPY rag-project/graph_rag_builder.py .
COPY rag-project/rag_ingest_qdrant.py .
COPY rag-project/text_extraction2.py .
COPY rag-project/pdf_cleaning4.py .

COPY .env .env

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
