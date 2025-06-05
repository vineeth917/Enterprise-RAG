import json
from neo4j import GraphDatabase

class GraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def ingest_links(self, links_file):  # 👈 FIX: moved inside the class
        with open(links_file, "r") as f:
            links = json.load(f)

        with self.driver.session() as session:
            for link in links:
                entity = link["entity"]
                label = link["label"].strip().upper()
                source = link["source_text"]
                source_type = link.get("source_type", "caption")
                caption = link.get("caption", source)
                location = link.get("location", "")
                date = link.get("date", "")
                image_path = link.get("image_path", "")

                domain = link.get("domain", "").strip().lower()
                if not domain or domain == "unknown":
                 if label in ["TIME", "DATE"]:
                   domain = "temporal"
                 elif label == "PERSON":
                   domain = "human"
                 elif label == "PLACE":
                   domain = "location"
                 elif label == "ORG":
                  domain = "organization"
                 elif label == "CARDINAL":
                  domain = "quantity"
                elif label == "WEATHER":
                  domain = "environment"
                else:
                  domain = "general"

                semantic_type = link.get("semantic_type", "").strip().lower()
                if not semantic_type or semantic_type == "unknown":
                  if label in ["TIME", "DATE"]:
                    semantic_type = "time_reference"
                  elif label == "PERSON":
                    semantic_type = "person"
                  elif label == "PLACE":
                    semantic_type = "landmark"
                  elif label == "ORG":
                    semantic_type = "organization"
                  elif label == "CARDINAL":
                    semantic_type = "number_reference"
                  elif label == "WEATHER":
                    semantic_type = "weather_condition"
                  else:
                    semantic_type = "entity" 

                # Check for existing canonical_id
                result = session.run("""
                    MATCH (e:Entity {name: $entity})
                    RETURN e.canonical_id AS canonical_id
                """, entity=entity).single()
                canonical_id = result["canonical_id"] if result and result["canonical_id"] else entity.lower()

# Create or update Entity node
                session.run("""
                     MERGE (e:Entity {name: $entity})
                     SET e.label = $label,
                     e.domain = $domain,
                     e.semantic_type = $semantic_type,
                     e.canonical_id = $canonical_id
                """, entity=entity, label=label, domain=domain, semantic_type=semantic_type, canonical_id=canonical_id)

# 🟩 Explicitly create :ALIAS_OF relationships for cross-document linking
                session.run("""
                    MERGE (canonical:Entity {name: $canonical_id})
                    MERGE (alias:Entity {name: $entity})
                    MERGE (alias)-[:ALIAS_OF]->(canonical)
                """, canonical_id=canonical_id, entity=entity)

# Create or update Text node
                session.run("""
                   MERGE (t:Text {text: $source, type: $source_type})
                   SET t.caption = $caption,
                   t.location = $location,
                   t.date = $date,
                   t.image_path = $image_path
                """, source=source, source_type=source_type,
                caption=caption, location=location, date=date, image_path=image_path)

# Create or update Image node
                if image_path and image_path != "null":
                    session.run("""
                    MERGE (i:Image {path: $image_path})
                    SET i.caption = $caption,
                    i.source_type = $source_type,
                    i.location = $location,
                    i.date = $date
                    MERGE (t:Text {text: $source, type: $source_type})
                    MERGE (t)-[:DESCRIBES]->(i)
                    """, { "image_path": image_path,
                    "caption": source,
                    "source_type": source_type,
                    "location": location,
                    "date": date,
                    "source": source
                     })

# Create relationship between Entity and Text
                session.run("""
                  MATCH (e:Entity {name: $entity})
                  MATCH (t:Text {text: $source})
                  MERGE (e)-[:MENTIONED_IN]->(t)
                """, entity=entity, source=source)

    print("✅ Graph data ingested with robust domain, semantic type & alias tracking!")

if __name__ == "__main__":
    builder = GraphBuilder("bolt://localhost:7687", "neo4j", "password")  # Update password if needed!
    builder.ingest_links("entity_relationship/linked_entities.json")
    builder.close()
