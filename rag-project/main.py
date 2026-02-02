# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import requests

from graph_retrieval import (
    GraphRetrieval,
)  # <-- your existing file

app = FastAPI(title="Graph RAG + Ollama API")

# -----------------------------
# CORS configuration
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request & Response Models
# -----------------------------
class QueryRequest(BaseModel):
    query: str


# Instantiate the retriever once for the app
retriever = GraphRetrieval()


# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
async def root():
    return {
        "message": "Graph RAG + Ollama API",
        "endpoint": "POST /search/ - send JSON {'query': 'your question'}",
    }


# -----------------------------
# POST endpoint for frontend queries
# -----------------------------
@app.post("/search/")
async def search_query(payload: QueryRequest):
    query = payload.query
    try:
        # Run the complete pipeline (retrieval + Ollama)
        chunks, answer = retriever.run_complete_pipeline(query)

        if answer is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to generate answer from Ollama"},
            )

        # Return answer (and optionally the chunks)
        return JSONResponse(
            content={
                "query": query,
                "answer": answer,
                "num_chunks": len(chunks),
                "status": "success",
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


@app.post("/search/stream")
async def search_query_stream(payload: QueryRequest):
    query = payload.query
    try:
        chunks = retriever.graph_enhanced_search(query)
        generator = retriever.stream_answer_with_llm(query, chunks)

        # StreamingResponse sends token-by-token
        return StreamingResponse(generator, media_type="text/plain")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.get("/health")
def health():
    return {"status": "ok"}
