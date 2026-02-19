# Document Retrieval & Graph-RAG Pipeline

This repository showcases the evolution of my PDF document processing system, from simple extraction to an AI-powered Graph-RAG pipeline:

- **v1 – Initial PDF Extraction:** Basic text extraction with `pdfplumber`, storing page-based content.  
- **v2 – Layout-Aware Cleaning:** Removes garbage lines, fixes hyphenation, and normalizes paragraphs while preserving tables and headings.  
- **v3 – Light RAG Pipeline:** Uses `PyMuPDF Layout` / `PyMuPDF4LLM` for layout-aware extraction, generates embeddings and semantic chunks for retrieval through ChromaDB.(https://www.loom.com/share/bd902906966248b9a575fea3f0537688)  
- **v4 – Graph-RAG + AI Integration (Here Now!):** Adds hierarchical clustering (Leiden using `Neo4j`) and AI model integration (`Ollama` -> `Bedrock`) for complex PDFs; local storage for fast retrieval.  
- **v5 – Production Ready (Planned):** Remote embedding storage (`Qdrant` + `Neo4j`), S3 PDF ingestion,hybrid retrieval and a full production-ready RAG system.

**Technologies**: **LangChain, Sentence Transformers, PyMuPDF, Python igraph, Leiden Algorithm, Scikit-learn, Neo4j, Qdrant, AWS (Boto3)**

Each version builds on the previous, progressively improving extraction, cleaning, embeddings, and AI-driven retrieval.
# graph-rag
