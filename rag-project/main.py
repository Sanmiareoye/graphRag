# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import requests

from graph_retrieval import (
    GraphRetrieval,
)

app = FastAPI(title="Graph RAG + Bedrock")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


retriever = GraphRetrieval()

@app.get("/")
async def root():
    return {
        "message": "Graph RAG + Bedrock",
        "endpoint": "POST /search/ - send JSON {'query': 'your question'}",
    }


@app.post("/ask/")
async def search_query(payload: QueryRequest):
    query = payload.query
    try:
        chunks, answer = retriever.run_complete_pipeline(query)

        if answer is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to generate answer from Ollama"},
            )

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


@app.post("/ask/stream")
async def search_query_stream(payload: QueryRequest):
    query = payload.query
    try:
        chunks = retriever.graph_enhanced_search(query)
        generator = retriever.stream_answer_with_llm(query, chunks)

        return StreamingResponse(generator, media_type="text/plain")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.get("/health")
def health():
    return {"status": "ok"}
