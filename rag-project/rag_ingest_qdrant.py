# rag-ingest-qdrant.py

import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from text_extraction2 import load_pdf_as_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from dotenv import load_dotenv
import boto3

load_dotenv()

api_key = os.getenv("QDRANT_APIKEY")


# ========================================
# SECTION 1: DOCUMENT LOADING & CHUNKING
# ========================================
def load_and_chunk_documents():
    documents = []

    s3 = boto3.resource(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

    bucket = s3.Bucket("dodpdfchunking")

    for obj in bucket.objects.filter(Prefix="raw/"):
        if not obj.key.lower().endswith(".pdf"):
            continue

        filename = obj.key.split("/")[-1]
        doc_id = filename.replace(".pdf", "")

        print(f"Loading from S3: {obj.key}")

        pdf_bytes = obj.get()["Body"].read()
        pages = load_pdf_as_text(pdf_bytes)

        for page in pages:
            documents.append(
                {
                    "id": doc_id,
                    "title": doc_id.replace("_", " ").upper(),
                    "content": page["content"],
                    "page": page["page"],
                    "category": "policy",
                    "author": page.get("author", ""),
                    "file_path": obj.key,
                }
            )

    # Chunking (unchanged)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"{doc['id']}_p{doc['page']}_c{i}",
                    "title": doc["title"],
                    "content": chunk,
                    "category": doc["category"],
                    "source_doc": doc["id"],
                    "page": doc["page"],
                    "author": doc.get("author", ""),
                    "file_path": doc.get("file_path", ""),
                }
            )

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


# ========================================
# SECTION 2: PUSH TO QDRANT
# ========================================


def setup_vector_database(chunks: List[Dict]):
    # Connect to local Qdrant Docker container
    client = QdrantClient(url="http://localhost:6333", api_key=api_key)
    COLLECTION_NAME = "dod_docs"

    # Initialize embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    documents = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(documents)

    # Prepare points for Qdrant
    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["id"])),
                vector=embeddings[i].tolist(),
                payload={
                    "content": chunk["content"],
                    "original_id": chunk["id"],
                    "title": chunk["title"],
                    "page": chunk["page"],
                    "source": chunk["source_doc"],
                    "author": chunk.get("author", ""),
                    "file_path": chunk.get("file_path", ""),
                },
            )
        )

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  Uploaded batch {i//batch_size + 1}/{len(points)//batch_size + 1}")

    print(f"✅ Uploaded {len(points)} chunks to Qdrant!")


# ========================================
# RUN
# ========================================

if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    setup_vector_database(chunks)
