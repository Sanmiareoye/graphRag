# graph_retrieval.py
"""
Graph-Enhanced Retrieval WITH AWS Bedrock Response Generation
Supports:
- Full answer (JSON)
- Live streaming (generator)
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

neo4j_pw = os.getenv("NEO4J_PASSWORD")
qdrant_uri = os.getenv("QDRANT_URI_LOCAL")
aws_region = os.getenv("AWS_REGION")
neo4j_uri = os.getenv("NEO4J_URI_LOCAL")


class GraphRetrieval:
    def __init__(self):
        self.qdrant = QdrantClient(url=qdrant_uri)
        self.collection_name = "dod_docs"

        self.neo4j = GraphDatabase.driver(neo4j_uri, auth=("neo4j", neo4j_pw))

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=aws_region,
        )

        model_choice = "llama3-3-70b"

        model_map = {
            # Meta Llama Models
            "llama3-1-8b": "us.meta.llama3-1-8b-instruct-v1:0",  # $138/mo - Fast, cheap, but repetition issues
            "llama3-3-70b": "us.meta.llama3-3-70b-instruct-v1:0",  # $167/mo - Best quality/price
            "llama3-1-70b": "us.meta.llama3-1-70b-instruct-v1:0",  # $167/mo - Same price as 3.3, use 3.3 instead
            # Mistral Models 
            "ministral-3b": "mistral.ministral-3-3b-instruct",  # $134/mo - Smallest, cheapest
            "ministral-8b": "mistral.ministral-3-8b-instruct",  # $141/mo - Good alternative to Llama 8B
            "ministral-14b": "mistral.ministral-3-14b-instruct",  # ~$150/mo - More capable Ministral
            "mistral-large-3": "mistral.mistral-large-3-675b-instruct",  # $176/mo - Premium Mistral
            # Claude Models
            "claude-haiku": "us.anthropic.claude-3-5-haiku-20241022-v2:0",  # $231/mo - Best quality, fastest
            "claude-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",  # $640/mo - Overkill for your use case
        }

        self.model_id = model_map[model_choice]

        print(f"✅ Connected to Qdrant, Neo4j, and Bedrock")
        print(f"📊 Using model: {model_choice} ({self.model_id})")

    # ==============================
    # Graph + Vector Retrieval
    # ==============================
    def graph_enhanced_search(self, query, top_k=4, expand_k=3):
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)

        print("\n📍 Step 1: Vector search for initial chunks...")
        query_embedding = self.model.encode(query).tolist()

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        ).points

        if not results:
            print("⚠️ No vector hits found")
            return []

        chunk_ids = []
        initial_chunks = []
        for hit in results:
            chunk_ids.append(str(hit.id))
            initial_chunks.append(
                {
                    "id": str(hit.id),
                    "content": hit.payload.get("content", ""),
                    "source": hit.payload.get("source", ""),
                    "page": hit.payload.get("page", 0),
                    "similarity": hit.score,
                    "method": "vector",
                }
            )
        print(f"Found {len(initial_chunks)} chunks from vector search")

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
            print("⚠️ No super-communities found, using only initial chunks")
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
                np.dot(query_vec, emb)
                / (np.linalg.norm(query_vec) * np.linalg.norm(emb) + 1e-8)
                for emb in embeddings
            ]

            sorted_expanded = [
                chunk
                for _, chunk in sorted(
                    zip(sims, expanded_chunks), key=lambda x: x[0], reverse=True
                )
            ]
            expanded_chunks = sorted_expanded[:expand_k]

        print(f"📈 Expanded with {len(expanded_chunks)} graph chunks (semantic rank)")

        all_chunks = initial_chunks + expanded_chunks
        print(f"📦 Total chunks for LLM: {len(all_chunks)}")

        return all_chunks

    # ==============================
    # NON-STREAMING: Generate Answer
    # ==============================
    def generate_answer_with_llm(self, query, chunks):
        print("\n" + "=" * 60)
        print("🤖 GENERATING ANSWER WITH BEDROCK")
        print("=" * 60)

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_tag = f"[{chunk['source']} p.{chunk['page']}]"
            context_parts.append(f"{source_tag}\n{chunk['content']}")

        context = "\n\n".join(context_parts)
        prompt = f"""You are a helpful assistant answering questions about military and Department of Defense documents.

Based on the following document excerpts, answer the user's question. Be specific and cite sources when possible.

DOCUMENT EXCERPTS:
{context}

QUESTION: {query}

ANSWER (be comprehensive but concise, do not ask if the user needs clarification):"""

        print(f"Sending {len(chunks)} chunks to Bedrock ({self.model_id})...")
        print("This may take 2-5 seconds...\n")

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(
                    {
                        "prompt": prompt,
                        "max_gen_len": 500,
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "stop": [
                            "Please let me know",
                            "Thank you for your question",
                            "\n\nPlease",
                            "\n\nThank you",
                        ],
                    }
                ),
            )

            result = json.loads(response["body"].read())
            answer = result.get("generation", "").strip()

            print("=" * 60)
            print("ANSWER:")
            print("=" * 60)
            print(answer)
            print("=" * 60)

            return answer

        except Exception as e:
            print(f"❌ Bedrock error: {e}")
            return None

    # ==============================
    # STREAMING: Generate Answer
    # ==============================
    def stream_answer_with_llm(self, query, chunks):
        context_parts = []
        for chunk in chunks:
            source_tag = f"[{chunk['source']} p.{chunk['page']}]"
            context_parts.append(f"{source_tag}\n{chunk['content']}")
        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant answering questions about military and Department of Defense documents.

Based on the following document excerpts, answer the user's question. Be very specific and descriptive, cite sources when possible.

DOCUMENT EXCERPTS:
{context}

QUESTION: {query}

ANSWER (be comprehensive but concise, do not ask if the user needs clarification):"""

        try:
            response = self.bedrock.invoke_model_with_response_stream(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(
                    {
                        "prompt": prompt,
                        "max_gen_len": 600,
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "stop": [
                            "Please let me know",
                            "Thank you for your question",
                            "\n\nPlease",
                            "\n\nThank you",
                        ],
                    }
                ),
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

    # ==============================
    # Full pipeline (non-streaming)
    # ==============================
    def run_complete_pipeline(self, query, top_k=4, expand_k=3):
        chunks = self.graph_enhanced_search(query, top_k, expand_k)
        answer = self.generate_answer_with_llm(query, chunks)

        return chunks, answer

    # ==============================
    # Close connections
    # ==============================
    def close(self):
        self.qdrant.close()
        self.neo4j.close()


# ==============================
# TEST
# ==============================
if __name__ == "__main__":
    retriever = GraphRetrieval()

    test_queries = [
        "What key rules in the II-208-29 Rulebook affect cargo inspection procedures?",
        "How does terrain analysis at Buckley Space Force Base affect transportation planning?",
        "How are 463L pallets tracked across multiple deployments?",
        "What systems protect sensitive cargo during air transport?",
        "What procedures are in place for coordinating military shipments during contingency operations?",
        "How are damaged pallets logged and reported in DoD operations?",
        "During amphibious operations, what are the responsibilities of a combat cargo officer in coordinating ship-to-shore movement and managing landing force operational reserve material (LFORM)?",
        "What strategies should the Department of Defense implement to adapt military installations and operations to climate change while maintaining readiness?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#'*70}")
        print(f"TEST {i}/{len(test_queries)}")
        print("#" * 70)

        chunks, answer = retriever.run_complete_pipeline(query, top_k=4, expand_k=3)
        print("\nANSWER:\n", answer)

        if i < len(test_queries):
            input("\n\n>>> Press Enter for next test...")

    retriever.close()
    print("\n" + "=" * 70)
    print("✅ All tests complete!")
