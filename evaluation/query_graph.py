import json
from neo4j import GraphDatabase

class GraphQuery:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def export_graph_data(self, output_file="knowledge_graph/graph_data.json"):
      with self.driver.session() as session:
        # Export nodes separately by label
        entity_nodes = []
        text_nodes = []
        image_nodes = []

        nodes_result = session.run("MATCH (n) RETURN id(n) as id, labels(n) as labels, properties(n) as properties")
        for record in nodes_result:
            node_props = dict(record["properties"])
            node_id = record["id"]
            labels = record["labels"]

            # Fallback metadata defaults
            node_props.setdefault("domain", "unknown")
            node_props.setdefault("semantic_type", "unknown")

            node_data = {
                "id": node_id,
                "labels": labels,
                "properties": node_props
            }

            if "Entity" in labels:
                entity_nodes.append(node_data)
            elif "Text" in labels:
                text_nodes.append(node_data)
            elif "Image" in labels:
                image_nodes.append(node_data)

        # Export relationships
        rels_result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN id(r) as id, type(r) as type, startNode(r) as start, endNode(r) as end, properties(r) as properties
        """)
        relationships = []
        for record in rels_result:
            rel_props = dict(record["properties"])
            relationships.append({
                "id": record["id"],
                "type": record["type"],
                "start": record["start"].element_id,  # use element_id for stability
                "end": record["end"].element_id,
                "properties": rel_props
            })

        # Save as structured JSON
        graph_data = {
            "entities": entity_nodes,
            "texts": text_nodes,
            "images": image_nodes,
            "relationships": relationships
        }

        with open(output_file, "w") as f:
            json.dump(graph_data, f, indent=2)

        print(f"✅ Graph data exported to {output_file} with structured JSON")


if __name__ == "__main__":
    gq = GraphQuery("bolt://localhost:7687", "neo4j", "password")  # Update password if needed!
    gq.export_graph_data()
    gq.close()

