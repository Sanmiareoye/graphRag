# graph_retrieval.py
"""
Graph-Enhanced Document Retrieval with LLM Integration
Supports:
- Full answer (JSON)
- Streaming answer generation
- Hybrid vector + graph retrieval
"""

from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import boto3
import json
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()


class GraphRetrieval:
    def __init__(self):
        self.neo4j_pw = os.getenv("NEO4J_PASSWORD")
        self.neo4j_uri = os.getenv("NEO4J_URI_LOCAL")
        self.qdrant_uri = os.getenv("QDRANT_URI_LOCAL")
        self.aws_region = os.getenv("AWS_REGION")

        self.qdrant = QdrantClient(url=self.qdrant_uri)
        self.collection_name = "documents"

        self.neo4j = GraphDatabase.driver(self.neo4j_uri, auth=("neo4j", self.neo4j_pw))
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=self.aws_region,
        )

        model_choice = "llama3-3-70b"
        model_map = {
            "llama3-1-8b": "us.meta.llama3-1-8b-instruct-v1:0",
            "llama3-3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
            "llama3-1-70b": "us.meta.llama3-1-70b-instruct-v1:0",
            "ministral-3b": "mistral.ministral-3-3b-instruct",
            "ministral-8b": "mistral.ministral-3-8b-instruct",
            "ministral-14b": "mistral.ministral-3-14b-instruct",
            "mistral-large-3": "mistral.mistral-large-3-675b-instruct",
            "claude-haiku": "us.anthropic.claude-3-5-haiku-20241022-v2:0",
            "claude-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        }
        self.model_id = model_map[model_choice]

        print(f"Connected to Qdrant, Neo4j, and Bedrock. Using model: {model_choice}")

    def graph_enhanced_search(self, query, top_k=4, expand_k=3):
        """
        Perform hybrid retrieval combining vector and graph data.
        """
        query_embedding = self.model.encode(query).tolist()
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        ).points

        if not results:
            return []

        chunk_ids = [str(hit.id) for hit in results]
        initial_chunks = [
            {
                "id": str(hit.id),
                "content": hit.payload.get("content", ""),
                "source": hit.payload.get("source", ""),
                "page": hit.payload.get("page", 0),
                "similarity": hit.score,
                "method": "vector",
            }
            for hit in results
        ]

        with self.neo4j.session() as session:
            super_result = session.run(
                """
                MATCH (c:Chunk)-[:MEMBER_OF]->(comm:Community)-[:MEMBER_OF]->(sc:SuperCommunity)
                WHERE c.id IN $chunk_ids
                RETURN DISTINCT sc.id AS super_id, sc.name AS super_name
                """,
                {"chunk_ids": chunk_ids},
            )
            super_communities = [dict(r) for r in super_result]

        if not super_communities:
            return initial_chunks

        super_ids = [sc["super_id"] for sc in super_communities]

        with self.neo4j.session() as session:
            result = session.run(
                """
                MATCH (sc:SuperCommunity)<-[:MEMBER_OF]-(comm:Community)<-[:MEMBER_OF]-(other:Chunk)
                WHERE sc.id IN $super_ids
                AND NOT other.id IN $original_chunk_ids
                RETURN other.id AS id,
                    other.text AS text,
                    other.source AS source,
                    other.page AS page,
                    other.embedding AS embedding,
                    comm.name AS community
                """,
                {"super_ids": super_ids, "original_chunk_ids": chunk_ids},
            )

            expanded_chunks = []
            embeddings = []
            for record in result:
                expanded_chunks.append(
                    {
                        "id": record["id"],
                        "content": record["text"],
                        "source": record["source"],
                        "page": record["page"],
                        "method": "graph",
                        "community": record["community"],
                    }
                )
                embeddings.append(np.array(record["embedding"]))

        if expanded_chunks:
            query_vec = np.array(query_embedding)
            sims = [
                np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb) + 1e-8)
                for emb in embeddings
            ]
            sorted_expanded = [
                chunk for _, chunk in sorted(zip(sims, expanded_chunks), key=lambda x: x[0], reverse=True)
            ]
            expanded_chunks = sorted_expanded[:expand_k]

        return initial_chunks + expanded_chunks

    def generate_answer_with_llm(self, query, chunks):
        """Generate a non-streaming answer using AWS Bedrock LLM."""
        context_parts = [f"[{c['source']} p.{c['page']}]\n{c['content']}" for c in chunks]
        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant answering questions based on document excerpts.

DOCUMENT EXCERPTS:
{context}

QUESTION: {query}

ANSWER (be comprehensive but concise, do not ask for clarification):"""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "prompt": prompt,
                    "max_gen_len": 500,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stop": ["\n\nPlease", "\n\nThank you"],
                }),
            )
            result = json.loads(response["body"].read())
            return result.get("generation", "").strip()
        except Exception as e:
            print(f"Bedrock error: {e}")
            return None

    def stream_answer_with_llm(self, query, chunks):
        """Generate a streaming answer using AWS Bedrock LLM."""
        context_parts = [f"[{c['source']} p.{c['page']}]\n{c['content']}" for c in chunks]
        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant answering questions based on document excerpts.

DOCUMENT EXCERPTS:
{context}

QUESTION: {query}

ANSWER (be comprehensive but concise, do not ask for clarification):"""

        try:
            response = self.bedrock.invoke_model_with_response_stream(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "prompt": prompt,
                    "max_gen_len": 600,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stop": ["\n\nPlease", "\n\nThank you"],
                }),
            )

            stream = response.get("body")
            if stream:
                for event in stream:
                    chunk = event.get("chunk")
                    if chunk:
                        chunk_data = json.loads(chunk.get("bytes").decode())
                        token = chunk_data.get("generation", "")
                        if token:
                            yield token
        except Exception as e:
            yield f"Error streaming from Bedrock: {e}"

    def run_complete_pipeline(self, query, top_k=4, expand_k=3):
        """Run full retrieval and answer generation pipeline."""
        chunks = self.graph_enhanced_search(query, top_k, expand_k)
        answer = self.generate_answer_with_llm(query, chunks)
        return chunks, answer

    def close(self):
        """Close connections to Qdrant and Neo4j."""
        self.qdrant.close()
        self.neo4j.close()


if __name__ == "__main__":
    retriever = GraphRetrieval()
    test_queries = [
        "What key rules in the II-208-29 Rulebook affect cargo inspection procedures?",
        "How does terrain analysis affect transportation planning?",
        "How are 463L pallets tracked across deployments?",
    ]

    for query in test_queries:
        chunks, answer = retriever.run_complete_pipeline(query)
        print(f"Query: {query}")
        print(f"Answer:\n{answer}\n")

    retriever.close()


    retriever.close()
    print("\n" + "=" * 70)
    print("✅ All tests complete!")
