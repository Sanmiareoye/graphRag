from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import os

load_dotenv()
neo4j_uri = os.getenv("NEO4J_URI_LOCAL")
neo4j_pw = os.getenv("NEO4J_PASSWORD")


class GraphStatsPrinter:
    def __init__(self):
        self.neo4j = GraphDatabase.driver(neo4j_uri, auth=("neo4j", neo4j_pw))

    def print_stats(self):
        with self.neo4j.session() as session:
            # Basic counts
            chunk_count = session.run(
                "MATCH (c:Chunk) RETURN count(c) AS count"
            ).single()["count"]

            edge_count = session.run(
                "MATCH ()-[r:SIMILAR]->() RETURN count(r) AS count"
            ).single()["count"]

            comm_count = session.run(
                "MATCH (c:Community) RETURN count(c) AS count"
            ).single()["count"]

            super_count = session.run(
                "MATCH (sc:SuperCommunity) RETURN count(sc) AS count"
            ).single()["count"]

            print("\n" + "=" * 50)
            print("GRAPH STATISTICS")
            print("=" * 50)
            print(f"Chunks: {chunk_count}")
            print(f"Similarity Edges: {edge_count}")
            print(f"L1 Communities: {comm_count}")
            print(f"L2 Super-Communities: {super_count}")

            # Hierarchy print
            print("\n" + "=" * 50)
            print("SUPER-COMMUNITY HIERARCHY")
            print("=" * 50)

            result = session.run(
                """
                MATCH (sc:SuperCommunity)<-[:MEMBER_OF]-(c:Community)
                RETURN sc.id AS super_id,
                    sc.name AS super_name,
                    sc.num_communities AS l1_count,
                    sc.size AS super_size,
                    c.name AS comm_name,
                    c.size AS comm_size
                ORDER BY sc.id, c.size DESC
                """
            )

            current_super_id = None

            for record in result:
                if record["super_id"] != current_super_id:
                    current_super_id = record["super_id"]
                    print(
                        f"\n🟣 Super-Community: {record['super_name']} "
                        f"({record['l1_count']} L1 communities, "
                        f"{record['super_size']} chunks)"
                    )

                print(
                    f"   ├─ 🔵 {record['comm_name']} " f"({record['comm_size']} chunks)"
                )

    def close(self):
        self.neo4j.close()


if __name__ == "__main__":
    printer = GraphStatsPrinter()
    printer.print_stats()
    printer.close()
