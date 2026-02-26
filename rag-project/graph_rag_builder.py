"""
Graph RAG Builder
Builds a hierarchical Graph-RAG structure from ChromaDB/Qdrant embeddings
and stores it in Neo4j.
"""

import numpy as np
import igraph
import leidenalg
from neo4j import GraphDatabase
from sklearn.neighbors import NearestNeighbors
import requests
from qdrant_client import QdrantClient
from collections import defaultdict
from dotenv import load_dotenv
import os

load_dotenv()


class GraphRAGBuilder:
    """Builds Graph-RAG from ChromaDB/Qdrant chunks and creates hierarchical communities."""

    def __init__(self):
        self.qdrant_client = QdrantClient(url=os.getenv("QDRANT_URI_LOCAL"))
        self.neo4j = GraphDatabase.driver(
            os.getenv("NEO4J_URI_LOCAL"), auth=("neo4j", os.getenv("NEO4J_PASSWORD"))
        )
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "llama3.2:3b"
        print("Connected to Qdrant and Neo4j")

    def build_graph(self, collection_name="documents"):
        """
        Fetch chunks, build similarity graphs, detect communities,
        generate super-communities, and store all in Neo4j.
        """
        node_ids, embeddings, chunks = self._fetch_from_qdrant(collection_name)
        k = 15
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="cosine")
        nbrs.fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)

        g = igraph.Graph(n=len(node_ids), edges=[
            (i, j_idx) for i in range(len(node_ids)) 
            for j_idx, dist in zip(indices[i], distances[i]) if i != j_idx and 1 - dist > 0.42
        ], directed=False)
        g.es["weight"] = [
            1 - dist for i in range(len(node_ids))
            for j_idx, dist in zip(indices[i], distances[i]) if i != j_idx and 1 - dist > 0.42
        ]

        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, weights=g.es["weight"], resolution_parameter=1.4
        )
        community_labels = partition.membership

        community_to_chunks = defaultdict(list)
        for i, comm_id in enumerate(community_labels):
            community_to_chunks[comm_id].append(i)

        community_centroids = {}
        community_ids = []
        for comm_id, chunk_indices in community_to_chunks.items():
            if len(chunk_indices) < 2:
                continue
            comm_embeddings = [embeddings[i] for i in chunk_indices]
            centroid = np.mean(comm_embeddings, axis=0)
            community_centroids[comm_id] = centroid
            community_ids.append(comm_id)

        centroid_matrix = np.array([community_centroids[cid] for cid in community_ids])
        k_super = min(5, len(community_ids) - 1)
        nbrs_super = NearestNeighbors(n_neighbors=k_super, algorithm="auto", metric="cosine")
        nbrs_super.fit(centroid_matrix)
        distances_super, indices_super = nbrs_super.kneighbors(centroid_matrix)

        edges_super = []
        weights_super = []
        for i in range(len(community_ids)):
            for j_idx, dist in zip(indices_super[i], distances_super[i]):
                if i != j_idx and 1 - dist > 0.3:
                    edges_super.append((i, j_idx))
                    weights_super.append(1 - dist)

        g_super = igraph.Graph(n=len(community_ids), edges=edges_super, directed=False)
        g_super.es["weight"] = weights_super

        partition_super = leidenalg.find_partition(
            g_super, leidenalg.RBConfigurationVertexPartition, weights=g_super.es["weight"], resolution_parameter=1.2
        )
        super_community_labels = partition_super.membership

        chunk_super_communities = [
            super_community_labels[community_ids.index(comm_id)] if comm_id in community_ids else -1
            for comm_id in community_labels
        ]

        self._store_in_neo4j(
            node_ids, chunks, embeddings, indices, distances,
            community_labels, chunk_super_communities
        )

        self._print_stats()

    def _fetch_from_qdrant(self, collection_name):
        """Fetch all embeddings and metadata from Qdrant."""
        node_ids, embeddings, chunks = [], [], []
        offset = None

        while True:
            result = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            points, offset = result
            if not points:
                break
            for point in points:
                node_ids.append(str(point.id))
                embeddings.append(point.vector)
                payload = point.payload
                chunks.append({
                    "id": str(point.id),
                    "text": payload.get("content", payload.get("text", "")),
                    "title": payload.get("title", ""),
                    "source": payload.get("source", ""),
                    "page": payload.get("page", 0),
                })
            if offset is None:
                break

        return node_ids, np.array(embeddings), chunks

    def _generate_community_name_ollama(self, community_texts, comm_id):
        """Generate concise community name using Ollama."""
        sample_texts = community_texts[:3]
        combined = " ".join(sample_texts)[:1500]
        if not combined.strip():
            return f"Community_{comm_id}"

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": f"Summarize this cluster in one concise name:\n{combined}\nName:",
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 30},
                },
                timeout=30,
            )
            if response.status_code == 200:
                name = response.json().get("response", "").strip()
                return name.split("\n")[0] if name else f"Community_{comm_id}"
            return f"Community_{comm_id}"
        except Exception:
            return f"Community_{comm_id}"

    def _store_in_neo4j(self, node_ids, chunks, embeddings, indices, distances, community_labels, super_community_labels):
        """Store hierarchical graph (chunks, communities, super-communities) in Neo4j."""
        with self.neo4j.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

            chunk_data = [{
                "id": nid,
                "text": chunks[i]["text"],
                "title": chunks[i]["title"],
                "source": chunks[i]["source"],
                "page": chunks[i]["page"],
                "community_l1": int(community_labels[i]),
                "community_l2": int(super_community_labels[i]),
                "embedding": embeddings[i].tolist(),
            } for i, nid in enumerate(node_ids)]

            session.run("""
                UNWIND $chunks AS chunk
                CREATE (c:Chunk {
                    id: chunk.id, text: chunk.text, title: chunk.title,
                    source: chunk.source, page: chunk.page,
                    community_l1: chunk.community_l1,
                    community_l2: chunk.community_l2,
                    embedding: chunk.embedding
                })
            """, {"chunks": chunk_data})

            edges_data = [
                {"idA": node_ids[i], "idB": node_ids[j_idx], "sim": float(1 - dist)}
                for i in range(len(node_ids))
                for j_idx, dist in zip(indices[i], distances[i])
                if i != j_idx and 1 - dist > 0.6
            ]
            session.run("""
                UNWIND $edges AS edge
                MATCH (a:Chunk {id: edge.idA})
                MATCH (b:Chunk {id: edge.idB})
                MERGE (a)-[r:SIMILAR {weight: edge.sim}]->(b)
            """, {"edges": edges_data})

    def _print_stats(self):
        """Print basic statistics for chunks and communities."""
        with self.neo4j.session() as session:
            chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) AS count").single()["count"]
            edge_count = session.run("MATCH ()-[r:SIMILAR]->() RETURN count(r) AS count").single()["count"]
            comm_count = session.run("MATCH (c:Community) RETURN count(c) AS count").single()["count"]

            print(f"Chunks: {chunk_count}, Similarity Edges: {edge_count}, Communities: {comm_count}")

    def close(self):
        """Close Neo4j connection."""
        self.neo4j.close()


if __name__ == "__main__":
    builder = GraphRAGBuilder()
    builder.build_graph(collection_name="documents")
    builder.close()
    print("Graph RAG build complete")

