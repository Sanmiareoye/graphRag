import numpy as np
import igraph
import leidenalg
from neo4j import GraphDatabase
from sklearn.neighbors import NearestNeighbors
import requests
from qdrant_client import QdrantClient
from collections import defaultdict

from config import config


class GraphRAGBuilder:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=config.QDRANT_URI,
            api_key=config.QDRANT_API_KEY,
        )

        self.neo4j = GraphDatabase.driver(
            config.NEO4J_URI, auth=("neo4j", config.NEO4J_PASSWORD)
        )

        self.ollama_url = config.OLLAMA_URL
        self.ollama_model = config.OLLAMA_MODEL

        print("✅ Connected to Qdrant and Neo4j")

    def build_graph(self, collection_name=None):
        if collection_name is None:
            collection_name = config.COLLECTION_NAME

        print("📥 Fetching chunks from Qdrant...")
        node_ids, embeddings, chunks = self._fetch_from_qdrant(collection_name)

        print(f"✅ Retrieved {len(node_ids)} chunks")

        print("🔗 Building k-NN similarity graph...")
        k = config.K_NEIGHBORS
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="cosine")
        nbrs.fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)

        print("🏘️  Detecting communities with Leiden...")
        edges = []
        weights = []

        for i in range(len(node_ids)):
            for j_idx, dist in zip(indices[i], distances[i]):
                if i == j_idx:
                    continue
                similarity = max(0, 1 - dist)
                if similarity > config.SIMILARITY_THRESHOLD:
                    edges.append((i, j_idx))
                    weights.append(similarity)

        g = igraph.Graph(n=len(node_ids), edges=edges, directed=False)
        g.es["weight"] = weights

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=config.LEIDEN_RESOLUTION_L1,
        )
        community_labels = partition.membership

        print(f"✅ Found {len(set(community_labels))} communities")

        print("🏘️  Building Level 2 super-communities...")

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

        print(f"  Calculated {len(community_centroids)} community centroids")

        centroid_matrix = np.array([community_centroids[cid] for cid in community_ids])

        k_super = min(config.K_SUPER_COMMUNITIES, len(community_ids) - 1)
        nbrs_super = NearestNeighbors(
            n_neighbors=k_super, algorithm="auto", metric="cosine"
        )
        nbrs_super.fit(centroid_matrix)
        distances_super, indices_super = nbrs_super.kneighbors(centroid_matrix)

        edges_super = []
        weights_super = []

        for i in range(len(community_ids)):
            for j_idx, dist in zip(indices_super[i], distances_super[i]):
                if i == j_idx:
                    continue
                similarity = max(0, 1 - dist)
                if similarity > config.SUPER_COMMUNITY_THRESHOLD:
                    edges_super.append((i, j_idx))
                    weights_super.append(similarity)

        g_super = igraph.Graph(n=len(community_ids), edges=edges_super, directed=False)
        g_super.es["weight"] = weights_super

        partition_super = leidenalg.find_partition(
            g_super,
            leidenalg.RBConfigurationVertexPartition,
            weights=g_super.es["weight"],
            resolution_parameter=config.LEIDEN_RESOLUTION_L2,
        )
        super_community_labels = partition_super.membership

        chunk_super_communities = []
        for chunk_comm_l1 in community_labels:
            if chunk_comm_l1 in community_ids:
                comm_idx = community_ids.index(chunk_comm_l1)
                super_comm = super_community_labels[comm_idx]
            else:
                super_comm = -1
            chunk_super_communities.append(super_comm)

        print(f"✅ Found {len(set(super_community_labels))} Level 2 super-communities")

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
        print("  Fetching all points from Qdrant...")

        node_ids = []
        embeddings = []
        chunks = []

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

                metadata = point.payload
                chunks.append(
                    {
                        "id": str(point.id),
                        "text": metadata.get("content", metadata.get("text", "")),
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", ""),
                        "page": metadata.get("page", 0),
                    }
                )

            if offset is None:
                break

        print(f"  Retrieved {len(node_ids)} total points")

        embeddings = np.array(embeddings)

        if len(embeddings.shape) == 1 or embeddings.shape[0] == 0:
            raise ValueError(
                f"Invalid embeddings shape: {embeddings.shape}. Check if vectors exist in Qdrant."
            )

        print(f"  Embeddings shape: {embeddings.shape}")

        return node_ids, embeddings, chunks

    def _generate_community_name_ollama(self, community_texts, comm_id):
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
        with self.neo4j.session() as session:
            print("  Clearing old data...")
            session.run("MATCH (n) DETACH DELETE n")

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

            print("  Creating similarity edges...")
            edges_data = []
            for i in range(len(node_ids)):
                idA = node_ids[i]
                for j_idx, dist in zip(indices[i], distances[i]):
                    if i == j_idx:
                        continue
                    sim = 1 - dist
                    idB = node_ids[j_idx]
                    if sim > config.EDGE_SIMILARITY_THRESHOLD:
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

            print("  Creating Level 1 community nodes...")

            communities_l1_map = defaultdict(list)
            for i, comm_id in enumerate(community_labels):
                communities_l1_map[comm_id].append(chunks[i]["text"])

            unique_communities_l1 = set(community_labels)

            for comm_id in unique_communities_l1:
                comm_chunks = [
                    c for c, comm in zip(chunks, community_labels) if comm == comm_id
                ]

                if len(comm_chunks) < 2:
                    continue

                chunk_idx = [i for i, c in enumerate(community_labels) if c == comm_id][
                    0
                ]
                super_comm = super_community_labels[chunk_idx]

                print(
                    f"    🤖 Naming L1 community {comm_id} ({len(comm_chunks)} chunks)..."
                )
                community_texts = communities_l1_map[comm_id]
                comm_name = self._generate_community_name_ollama(
                    community_texts, comm_id
                )

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

                session.run(
                    """
                    MATCH (chunk:Chunk {community_l1: $comm_id})
                    MATCH (comm:Community {id: $comm_id, level: 1})
                    CREATE (chunk)-[:MEMBER_OF]->(comm)
                """,
                    {"comm_id": int(comm_id)},
                )

            print("  Creating Level 2 super-community nodes...")

            super_communities_map = defaultdict(list)
            for comm_id in unique_communities_l1:
                comm_chunks = [
                    c for c, comm in zip(chunks, community_labels) if comm == comm_id
                ]
                if len(comm_chunks) < 2:
                    continue

                chunk_idx = [i for i, c in enumerate(community_labels) if c == comm_id][
                    0
                ]
                super_comm = super_community_labels[chunk_idx]

                super_communities_map[super_comm].extend(communities_l1_map[comm_id])

            unique_super_communities = set(super_community_labels)

            for super_id in unique_super_communities:
                if super_id == -1:
                    continue

                if super_id not in super_communities_map:
                    continue

                all_texts = super_communities_map[super_id]

                if len(all_texts) < 2:
                    continue

                print(
                    f"    🤖 Naming L2 super-community {super_id} ({len(all_texts)} total texts)..."
                )
                super_name = self._generate_community_name_ollama(
                    all_texts[:10], f"super_{super_id}"
                )

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

                session.run(
                    """
                    MATCH (comm:Community {level: 1})
                    WHERE comm.super_community = $super_id
                    MATCH (sc:SuperCommunity {id: $super_id})
                    CREATE (comm)-[:MEMBER_OF]->(sc)
                """,
                    {"super_id": int(super_id)},
                )

        print("  ✅ Hierarchical graph stored successfully!")

    def _print_stats(self):
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
    builder.build_graph(collection_name="documents")
    builder.close()

    print("\n✅ Done! Visualize at: http://localhost:7474")
