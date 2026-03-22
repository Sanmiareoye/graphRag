# Hierarchical Graph-RAG System

An advanced document retrieval system that combines vector search with graph-based community detection for enhanced contextual understanding.

## Overview

This project implements a **hierarchical Graph-RAG (Retrieval-Augmented Generation)** pipeline that goes beyond traditional vector similarity search by leveraging graph structures and community detection to capture document relationships and improve retrieval quality.

### Key Features

- **Hybrid Retrieval**: Combines vector similarity (Qdrant) with graph-based expansion (Neo4j)
- **Hierarchical Communities**: Two-level community structure using Leiden algorithm
- **Semantic Re-ranking**: Context-aware ranking of expanded chunks
- **Production-Ready API**: FastAPI with streaming support
- **AWS Bedrock Integration**: LLM-powered answer generation

## Architecture

```
PDF Documents (S3)
    ↓
Text Extraction (PyMuPDF4LLM)
    ↓
Chunking & Embedding (SentenceTransformers)
    ↓
┌─────────────────┬──────────────────┐
│   Qdrant        │     Neo4j        │
│ Vector Search   │  Graph Structure │
└─────────────────┴──────────────────┘
    ↓
Graph-Enhanced Retrieval
    ↓
LLM Answer Generation (AWS Bedrock)
```

## Technical Stack

- **Vector Database**: Qdrant
- **Graph Database**: Neo4j
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: AWS Bedrock (Llama 3.3 70B)
- **API Framework**: FastAPI
- **PDF Processing**: PyMuPDF4LLM
- **Graph Algorithms**: igraph + Leiden community detection

## How It Works

### 1. Document Ingestion

- PDFs extracted from S3 with layout-aware parsing
- Text cleaned and chunked (900 chars, 100 overlap)
- Embedded using SentenceTransformers
- Stored in Qdrant with metadata

### 2. Graph Construction

- **k-NN Graph**: 15 nearest neighbors for each chunk
- **L1 Communities**: Leiden algorithm detects topical clusters
- **L2 Super-Communities**: Higher-level groupings of related communities
- **Named Communities**: Ollama generates descriptive names for each cluster

### 3. Retrieval Pipeline

1. **Vector Search**: Find top-k similar chunks
2. **Community Expansion**: Retrieve related chunks from same super-communities
3. **Semantic Re-ranking**: Re-score expanded chunks by query relevance
4. **LLM Generation**: Synthesize answer from combined context

## Setup

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- AWS credentials (for Bedrock)
- Qdrant (local or cloud)
- Neo4j database

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GraphRAG
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your credentials
```

### Configuration

Key settings in `rag-project/config.py`:

- `K_NEIGHBORS`: Number of nearest neighbors (default: 15)
- `SIMILARITY_THRESHOLD`: Edge creation threshold (default: 0.42)
- `LEIDEN_RESOLUTION_L1/L2`: Community detection resolution
- `CHUNK_SIZE/OVERLAP`: Text chunking parameters

## Usage

### 1. Ingest Documents

```bash
cd rag-project
python rag_ingest_qdrant.py
```

### 2. Build Graph

```bash
python graph_rag_builder.py
```

### 3. Run API

```bash
uvicorn main:app --reload
```

### 4. Query

```bash
curl -X POST http://localhost:8000/ask/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key procedures?"}'
```

## Docker Deployment

```bash
docker-compose up -d
```

Services:
- FastAPI: `http://localhost:8000`
- Neo4j Browser: `http://localhost:7474`
- Qdrant: `http://localhost:6333`

## Project Structure

```
GraphRAG/
├── rag-project/
│   ├── config.py                  # Centralized configuration
│   ├── graph_rag_builder.py       # Graph construction
│   ├── graph_retrieval.py         # Retrieval pipeline
│   ├── main.py                    # FastAPI application
│   ├── rag_ingest_qdrant.py       # Document ingestion
│   ├── text_extraction2.py        # PDF processing
│   └── pdf_cleaning4.py           # Text cleaning
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Key Algorithms

### Leiden Community Detection

- Identifies clusters of semantically related chunks
- Two-level hierarchy captures both local and global structure
- Resolution parameters tunable via config

### Graph-Enhanced Retrieval

1. Initial vector search provides high-precision seed chunks
2. Graph expansion adds high-recall related context
3. Semantic re-ranking balances precision and recall

## Performance Considerations

- **Embeddings**: Cached in Qdrant for fast retrieval
- **Graph Structure**: Pre-computed communities avoid runtime clustering
- **Configurable Trade-offs**: Adjust k, thresholds, and resolution parameters

## Future Enhancements

- Incremental graph updates (avoid full rebuilds)
- Multi-document reasoning across super-communities
- Query-specific community selection
- Hybrid ranking (BM25 + vector + graph)

## License

MIT

## Acknowledgments

Built with inspiration from Microsoft's GraphRAG and academic research on graph-based retrieval systems.
