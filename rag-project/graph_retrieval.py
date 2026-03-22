from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import boto3
import json
import numpy as np

from config import config


class GraphRetrieval:
    def __init__(self):
        self.qdrant = QdrantClient(url=config.QDRANT_URI)
        self.collection_name = config.COLLECTION_NAME

        self.neo4j = GraphDatabase.driver(
            config.NEO4J_URI, auth=("neo4j", config.NEO4J_PASSWORD)
        )

        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=config.AWS_REGION,
        )

        model_choice = config.LLM_MODEL_CHOICE

        model_map = {
            "llama3-1-8b": "us.meta.llama3-1-8b-instruct-v1:0",
            "llama3-3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
            "llama3-1-70b": "us.meta.llama3-1-70b-instruct-v1:0",
            "ministral-3b": "mistral.ministral-3-3b-instruct",
            "ministral-8b": "mistral.ministral-3-8b-instruct",
            "ministral-14b": "mistral.ministral-3-14b-instruct",
            "mistral-large-3": "mistral.mistral-large-3-675b-instruct",
            "claude-haiku": "us.anthropic.claude-3-5-haiku-20241022-v2:0",
            "claude-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        }

        self.model_id = model_map[model_choice]

        print(f"✅ Connected to Qdrant, Neo4j, and Bedrock")
        print(f"📊 Using model: {model_choice} ({self.model_id})")

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

    def generate_answer_with_llm(self, query, chunks):
        print("\n" + "=" * 60)
        print("🤖 GENERATING ANSWER WITH BEDROCK")
        print("=" * 60)

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_tag = f"[{chunk['source']} p.{chunk['page']}]"
            context_parts.append(f"{source_tag}\n{chunk['content']}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant answering questions based on document context.

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

    def stream_answer_with_llm(self, query, chunks):
        context_parts = []
        for chunk in chunks:
            source_tag = f"[{chunk['source']} p.{chunk['page']}]"
            context_parts.append(f"{source_tag}\n{chunk['content']}")
        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant answering questions based on document context.

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

    def run_complete_pipeline(self, query, top_k=4, expand_k=3):
        chunks = self.graph_enhanced_search(query, top_k, expand_k)
        answer = self.generate_answer_with_llm(query, chunks)

        return chunks, answer

    def close(self):
        self.qdrant.close()
        self.neo4j.close()


if __name__ == "__main__":
    retriever = GraphRetrieval()

    test_queries = [
        "What are the key procedures outlined in the documentation?",
        "How does the system handle data processing workflows?",
        "What tracking mechanisms are described in the documents?",
        "What security measures are implemented for data protection?",
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
