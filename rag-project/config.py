import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    NEO4J_URI = os.getenv("NEO4J_URI_LOCAL", "bolt://localhost:7687")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    QDRANT_URI = os.getenv("QDRANT_URI_LOCAL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_APIKEY")

    AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "document-storage")

    COLLECTION_NAME = "documents"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL_CHOICE = "llama3-3-70b"

    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "llama3.2:3b"

    K_NEIGHBORS = 15
    K_SUPER_COMMUNITIES = 5
    SIMILARITY_THRESHOLD = 0.42
    SUPER_COMMUNITY_THRESHOLD = 0.3
    EDGE_SIMILARITY_THRESHOLD = 0.6

    LEIDEN_RESOLUTION_L1 = 1.4
    LEIDEN_RESOLUTION_L2 = 1.2

    CHUNK_SIZE = 900
    CHUNK_OVERLAP = 100

    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")


config = Config()
