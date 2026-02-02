# graph_rag_builder.py
"""
Build Graph RAG from ChromaDB chunks
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

neo4j_pw = os.getenv("NEO4J_PASSWORD")
qdrant_uri = os.getenv("QDRANT_URI_LOCAL")
aws_region = os.getenv("AWS_REGION")
neo4j_uri = os.getenv("NEO4J_URI_LOCAL")


class GraphRAGBuilder:
    def __init__(self):
        # YOUR ChromaDB path (from rag-ingest.py)
        self.qdrant_client = QdrantClient(
            url=qdrant_uri,  # Qdrant server URL
            api_key=None,  # only needed if using cloud
        )

        # YOUR Neo4j connection
        self.neo4j = GraphDatabase.driver(neo4j_uri, auth=("neo4j", neo4j_pw))

        # Ollama setup (just 3 lines added!)
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "llama3.2:3b"

        print("✅ Connected to Qdrant and Neo4j")

    def build_graph(self, collection_name="dod_docs"):
        """
        Complete pipeline:
        1. Fetch chunks from ChromaDB
        2. Build k-NN similarity graph
        3. Detect communities with Leiden
        4. Store in Neo4j
        """

        print("📥 Fetching chunks from Qdrant...")
        node_ids, embeddings, chunks = self._fetch_from_qdrant(collection_name)

        print(f"✅ Retrieved {len(node_ids)} chunks")

        # ========================================
        # FROM DENIZ: k-NN Similarity
        # ========================================
        print("🔗 Building k-NN similarity graph...")
        k = 15
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="cosine")
        nbrs.fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)

        # ========================================
        # FROM DENIZ: Leiden Community Detection
        # ========================================
        print("🏘️  Detecting communities with Leiden...")
        edges = []
        weights = []

        for i in range(len(node_ids)):
            for j_idx, dist in zip(indices[i], distances[i]):
                if i == j_idx:
                    continue
                similarity = max(0, 1 - dist)
                if similarity > 0.42:
                    edges.append((i, j_idx))
                    weights.append(similarity)

        # Build igraph
        g = igraph.Graph(n=len(node_ids), edges=edges, directed=False)
        g.es["weight"] = weights

        # Run Leiden
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=1.4,
        )
        community_labels = partition.membership

        print(f"✅ Found {len(set(community_labels))} communities")

        # ========================================
        # NEW: LEVEL 2 - SUPER-COMMUNITIES
        # ========================================
        print("🏘️  Building Level 2 super-communities...")

        # Step 1: Calculate community centroids
        community_to_chunks = defaultdict(list)
        for i, comm_id in enumerate(community_labels):
            community_to_chunks[comm_id].append(i)

        community_centroids = {}
        community_ids = []

        for comm_id, chunk_indices in community_to_chunks.items():
            if len(chunk_indices) < 2:  # Skip singletons
                continue

            # Average embeddings of all chunks in this community
            comm_embeddings = [embeddings[i] for i in chunk_indices]
            centroid = np.mean(comm_embeddings, axis=0)

            community_centroids[comm_id] = centroid
            community_ids.append(comm_id)

        print(f"  Calculated {len(community_centroids)} community centroids")

        # Step 2: k-NN on community centroids
        centroid_matrix = np.array([community_centroids[cid] for cid in community_ids])

        k_super = min(
            5, len(community_ids) - 1
        )  # Fewer neighbors for super-communities
        nbrs_super = NearestNeighbors(
            n_neighbors=k_super, algorithm="auto", metric="cosine"
        )
        nbrs_super.fit(centroid_matrix)
        distances_super, indices_super = nbrs_super.kneighbors(centroid_matrix)

        # Step 3: Build graph of communities
        edges_super = []
        weights_super = []

        for i in range(len(community_ids)):
            for j_idx, dist in zip(indices_super[i], distances_super[i]):
                if i == j_idx:
                    continue
                similarity = max(0, 1 - dist)
                if similarity > 0.3:  # Lower threshold for super-communities
                    edges_super.append((i, j_idx))
                    weights_super.append(similarity)

        # Step 4: Leiden on community graph
        g_super = igraph.Graph(n=len(community_ids), edges=edges_super, directed=False)
        g_super.es["weight"] = weights_super

        partition_super = leidenalg.find_partition(
            g_super,
            leidenalg.RBConfigurationVertexPartition,
            weights=g_super.es["weight"],
            resolution_parameter=1.2,  # Slightly higher for super-communities
        )
        super_community_labels = partition_super.membership

        # Step 5: Map chunks to super-communities
        chunk_super_communities = []
        for chunk_comm_l1 in community_labels:
            if chunk_comm_l1 in community_ids:
                comm_idx = community_ids.index(chunk_comm_l1)
                super_comm = super_community_labels[comm_idx]
            else:
                super_comm = -1  # Singleton
            chunk_super_communities.append(super_comm)

        print(f"✅ Found {len(set(super_community_labels))} Level 2 super-communities")

        # ========================================
        # FROM DENIZ: Store in Neo4j
        # ========================================
        print("💾 Storing graph in Neo4j...")
        self._store_in_neo4j(
            node_ids,
            chunks,
            embeddings,
            indices,
            distances,
            community_labels,
            chunk_super_communities,
        )

        print("🎉 Graph RAG built successfully!")
        self._print_stats()

    def _fetch_from_qdrant(self, collection_name):
        """
        Fetch all vectors + metadata from Qdrant
        Returns: node_ids, embeddings, chunks
        """
        print("  Fetching all points from Qdrant...")

        node_ids = []
        embeddings = []
        chunks = []

        # Qdrant scroll returns batches, need to get ALL points
        offset = None

        while True:
            # Scroll through collection
            result = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )

            points, offset = result

            # If no points returned, we're done
            if not points:
                break

            # Process each point
            for point in points:
                node_ids.append(str(point.id))  # Convert to string for consistency
                embeddings.append(point.vector)  # This is now a list/array

                metadata = point.payload
                chunks.append(
                    {
                        "id": str(point.id),
                        "text": metadata.get(
                            "content", metadata.get("text", "")
                        ),  # Try both keys
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", ""),
                        "page": metadata.get("page", 0),
                    }
                )

            # If offset is None, we've retrieved all points
            if offset is None:
                break

        print(f"  Retrieved {len(node_ids)} total points")

        # Convert to numpy array
        embeddings = np.array(embeddings)

        # Verify shape
        if len(embeddings.shape) == 1 or embeddings.shape[0] == 0:
            raise ValueError(
                f"Invalid embeddings shape: {embeddings.shape}. Check if vectors exist in Qdrant."
            )

        print(f"  Embeddings shape: {embeddings.shape}")

        return node_ids, embeddings, chunks

    def _generate_community_name_ollama(self, community_texts, comm_id):
        """Use Ollama to generate community name - 100% FREE"""
        sample_texts = community_texts[:3]
        combined = " ".join(sample_texts)[:1500]

        if not combined.strip():
            return f"Community_{comm_id}"

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": f"""You are labeling a document cluster to summarize its content. Generate **ONE concise, descriptive name** that clearly represents this cluster. 

        Rules:
        - Use 3-7 words maximum.
        - Make it clear what the community is about.
        - Do not output multiple names, instructions, or filler text.

        Document cluster:
        {combined}

    Name:""",
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 30},
                },
                timeout=30,
            )

            if response.status_code == 200:
                name = response.json().get("response", "").strip()
                name = name.replace('"', "").replace("'", "").split("\n")[0]
                return name if name else f"Community_{comm_id}"
            else:
                return f"Community_{comm_id}"

        except Exception as e:
            print(f"  ⚠️  Ollama failed: {e}")
            return f"Community_{comm_id}"

    def _store_in_neo4j(
        self,
        node_ids,
        chunks,
        embeddings,
        indices,
        distances,
        community_labels,
        super_community_labels,
    ):
        """Store graph in Neo4j with hierarchical communities"""

        with self.neo4j.session() as session:
            # ========================================
            # CLEAR OLD DATA
            # ========================================
            print("  Clearing old data...")
            session.run("MATCH (n) DETACH DELETE n")

            # ========================================
            # CREATE CHUNK NODES
            # ========================================
            print("  Creating chunk nodes...")
            chunk_data = []
            for i, nid in enumerate(node_ids):
                chunk_data.append(
                    {
                        "id": nid,
                        "text": chunks[i]["text"],
                        "title": chunks[i]["title"],
                        "source": chunks[i]["source"],
                        "page": chunks[i]["page"],
                        "community_l1": int(community_labels[i]),
                        "community_l2": int(super_community_labels[i]),
                        "embedding": embeddings[i].tolist(),
                    }
                )

            session.run(
                """
                UNWIND $chunks AS chunk
                CREATE (c:Chunk {
                    id: chunk.id,
                    text: chunk.text,
                    title: chunk.title,
                    source: chunk.source,
                    page: chunk.page,
                    community_l1: chunk.community_l1,
                    community_l2: chunk.community_l2,
                    embedding: chunk.embedding
                })
            """,
                {"chunks": chunk_data},
            )

            # ========================================
            # CREATE SIMILAR EDGES
            # ========================================
            print("  Creating similarity edges...")
            edges_data = []
            for i in range(len(node_ids)):
                idA = node_ids[i]
                for j_idx, dist in zip(indices[i], distances[i]):
                    if i == j_idx:
                        continue
                    sim = 1 - dist
                    idB = node_ids[j_idx]
                    if sim > 0.6:
                        edges_data.append({"idA": idA, "idB": idB, "sim": float(sim)})

            session.run(
                """
                UNWIND $edges AS edge
                MATCH (a:Chunk {id: edge.idA})
                MATCH (b:Chunk {id: edge.idB})
                MERGE (a)-[r:SIMILAR {weight: edge.sim}]->(b)
            """,
                {"edges": edges_data},
            )

            # ========================================
            # CREATE LEVEL 1 COMMUNITY NODES
            # ========================================
            print("  Creating Level 1 community nodes...")

            # Group chunks by L1 community
            communities_l1_map = defaultdict(list)
            for i, comm_id in enumerate(community_labels):
                communities_l1_map[comm_id].append(chunks[i]["text"])

            # Get unique L1 communities
            unique_communities_l1 = set(community_labels)

            # Create each L1 Community node
            for comm_id in unique_communities_l1:
                comm_chunks = [
                    c for c, comm in zip(chunks, community_labels) if comm == comm_id
                ]

                # Skip singletons
                if len(comm_chunks) < 2:
                    continue

                # Get super-community for this L1 community
                chunk_idx = [i for i, c in enumerate(community_labels) if c == comm_id][
                    0
                ]
                super_comm = super_community_labels[chunk_idx]

                # Generate L1 community name with Ollama
                print(
                    f"    🤖 Naming L1 community {comm_id} ({len(comm_chunks)} chunks)..."
                )
                community_texts = communities_l1_map[comm_id]
                comm_name = self._generate_community_name_ollama(
                    community_texts, comm_id
                )

                # Create L1 Community node
                session.run(
                    """
                    CREATE (c:Community {
                        id: $comm_id,
                        name: $name,
                        size: $size,
                        level: 1,
                        super_community: $super_comm
                    })
                """,
                    {
                        "comm_id": int(comm_id),
                        "name": comm_name,
                        "size": len(comm_chunks),
                        "super_comm": int(super_comm),
                    },
                )

                # Link chunks to L1 communities
                session.run(
                    """
                    MATCH (chunk:Chunk {community_l1: $comm_id})
                    MATCH (comm:Community {id: $comm_id, level: 1})
                    CREATE (chunk)-[:MEMBER_OF]->(comm)
                """,
                    {"comm_id": int(comm_id)},
                )

            # ========================================
            # CREATE LEVEL 2 SUPER-COMMUNITY NODES
            # ========================================
            print("  Creating Level 2 super-community nodes...")

            # Group L1 communities by super-community
            super_communities_map = defaultdict(list)
            for comm_id in unique_communities_l1:
                # Skip singletons
                comm_chunks = [
                    c for c, comm in zip(chunks, community_labels) if comm == comm_id
                ]
                if len(comm_chunks) < 2:
                    continue

                # Get super-community for this L1 community
                chunk_idx = [i for i, c in enumerate(community_labels) if c == comm_id][
                    0
                ]
                super_comm = super_community_labels[chunk_idx]

                # Add this L1 community's texts to super-community map
                super_communities_map[super_comm].extend(communities_l1_map[comm_id])

            # Get unique super-communities
            unique_super_communities = set(super_community_labels)

            # Create each L2 Super-Community node
            for super_id in unique_super_communities:
                # Skip invalid super-communities
                if super_id == -1:
                    continue

                # Skip if no texts
                if super_id not in super_communities_map:
                    continue

                all_texts = super_communities_map[super_id]

                # Skip if too few texts
                if len(all_texts) < 2:
                    continue

                # Generate L2 super-community name with Ollama
                print(
                    f"    🤖 Naming L2 super-community {super_id} ({len(all_texts)} total texts)..."
                )
                super_name = self._generate_community_name_ollama(
                    all_texts[:10], f"super_{super_id}"
                )

                # Count how many L1 communities in this super-community
                l1_communities_in_super = [
                    c
                    for c in unique_communities_l1
                    if len(
                        [c2 for c2, comm in zip(chunks, community_labels) if comm == c]
                    )
                    >= 2
                    and super_community_labels[
                        [i for i, comm in enumerate(community_labels) if comm == c][0]
                    ]
                    == super_id
                ]
                l1_count = len(l1_communities_in_super)

                # Create L2 SuperCommunity node
                session.run(
                    """
                    CREATE (sc:SuperCommunity {
                        id: $super_id,
                        name: $name,
                        size: $size,
                        level: 2,
                        num_communities: $l1_count
                    })
                """,
                    {
                        "super_id": int(super_id),
                        "name": super_name,
                        "size": len(all_texts),
                        "l1_count": l1_count,
                    },
                )

                # Link L1 communities to super-communities
                session.run(
                    """
                    MATCH (comm:Community {level: 1})
                    WHERE comm.super_community = $super_id
                    MATCH (sc:SuperCommunity {id: $super_id})
                    CREATE (comm)-[:MEMBER_OF]->(sc)
                """,
                    {"super_id": int(super_id)},
                )

        # ✅ CRITICAL: This print is OUTSIDE the with block (correct!)
        print("  ✅ Hierarchical graph stored successfully!")

    def _print_stats(self):
        """Print statistics"""
        with self.neo4j.session() as session:
            result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
            chunk_count = result.single()["count"]

            result = session.run("MATCH ()-[r:SIMILAR]->() RETURN count(r) as count")
            edge_count = result.single()["count"]

            result = session.run("MATCH (c:Community) RETURN count(c) as count")
            comm_count = result.single()["count"]

            print("\n" + "=" * 50)
            print("GRAPH STATISTICS")
            print("=" * 50)
            print(f"Chunks: {chunk_count}")
            print(f"Similarity Edges: {edge_count}")
            print(f"Communities: {comm_count}")
            print("=" * 50)

            print("\nSample Communities:")
            result = session.run(
                """
                MATCH (c:Community)
                RETURN c.name as name, c.size as size
                ORDER BY c.size DESC
            """
            )
            for record in result:
                print(f"  • {record['name']} ({record['size']} chunks)")

    def close(self):
        self.neo4j.close()


if __name__ == "__main__":
    builder = GraphRAGBuilder()
    builder.build_graph(collection_name="dod_docs")
    builder.close()

    print("\n✅ Done! Visualize at: http://localhost:7474")
