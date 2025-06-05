import json
import os
from datetime import datetime
from neo4j import GraphDatabase
from typing import List, Dict, Optional, Union
import spacy

class GraphQuery:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.nlp = spacy.load("en_core_web_lg")
        # Create output directories if they don't exist
        self.output_dir = "knowledge_graph/query_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def close(self):
        if self.driver:
            self.driver.close()

    def _process_query_text(self, query_text: Optional[str]) -> List[str]:
        """Process query text into a list of query terms."""
        if not query_text:
            return []
        
        return [
            token.lemma_.lower() for token in self.nlp(query_text)
            if not token.is_stop and not token.is_punct and len(token.text) > 2
        ]

    def save_query_results(self, results: Union[List[Dict], Dict], query_type: str, query_params: Dict) -> str:
        """Save query results to a JSON file with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{query_type}_{timestamp}.json"
        output_path = os.path.join(self.output_dir, filename)
        
        output_data = {
            "query_type": query_type,
            "query_params": query_params,
            "timestamp": timestamp,
            "results": results
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Query results saved to: {output_path}")
        return output_path

    def find_related_entities_with_context(self, entity_name: str, query_text: Optional[str] = None) -> List[Dict]:
        """Find related entities with context-aware relevance scoring."""
        query_terms = self._process_query_text(query_text)
        
        with self.driver.session() as session:
            query = """
            MATCH (n:Entity {name: $entity_name})-[r]-(related:Entity)
            WITH related, r, 
                 CASE WHEN any(term IN $query_terms WHERE toLower(related.name) CONTAINS term) THEN 5
                      WHEN related.label IN ['PLACE', 'BUILDING', 'TIME', 'WEATHER'] THEN 2
                      WHEN related.label IN ['PERSON', 'ORG'] THEN 1
                      ELSE 1.5 END as relevance_score
            OPTIONAL MATCH (related)-[:MENTIONED_IN]->(t:Text)
            WITH related, r, relevance_score, collect(DISTINCT t.text)[0..3] as mentions
            RETURN related.name as name,
                   related.label as label,
                   type(r) as relationship_type,
                   related.domain as domain,
                   related.semantic_type as semantic_type,
                   mentions,
                   relevance_score
            ORDER BY relevance_score DESC
            """
            
            result = session.run(query, 
                               entity_name=entity_name,
                               query_terms=query_terms)
            results = [dict(record) for record in result]
            
            # Save results
            self.save_query_results(
                results,
                "related_entities",
                {"entity_name": entity_name, "query_text": query_text}
            )
            return results

    def find_paths_with_relevance(self, entity1: str, entity2: str, query_text: Optional[str] = None, max_length: int = 3) -> List[Dict]:
        """Find paths between entities with relevance scoring."""
        query_terms = self._process_query_text(query_text)
        
        with self.driver.session() as session:
            query = """
            MATCH path = shortestPath((n1:Entity {name: $entity1})-[*1..%d]-(n2:Entity {name: $entity2}))
            WITH path,
                 [node in nodes(path) | node.name] as node_names,
                 [rel in relationships(path) | type(rel)] as relationship_types,
                 CASE WHEN any(term IN $query_terms 
                              WHERE any(node in nodes(path) 
                                      WHERE toLower(node.name) CONTAINS term)) 
                      THEN 5 ELSE 1 END as path_relevance
            RETURN node_names,
                   relationship_types,
                   length(path) as path_length,
                   path_relevance
            ORDER BY path_relevance DESC, path_length ASC
            """ % max_length
            
            result = session.run(query, 
                               entity1=entity1,
                               entity2=entity2,
                               query_terms=query_terms)
            results = [dict(record) for record in result]
            
            # Save results
            self.save_query_results(
                results,
                "entity_paths",
                {"entity1": entity1, "entity2": entity2, "query_text": query_text, "max_length": max_length}
            )
            return results

    def find_entities_by_type_and_relevance(self, label: str, query_text: str) -> List[Dict]:
        """Find entities by type with query-aware relevance scoring."""
        query_terms = self._process_query_text(query_text)
        
        with self.driver.session() as session:
            query = """
            MATCH (n:Entity)-[:MENTIONED_IN]->(t:Text)
            WHERE n.label = $label
            WITH n, t,
                 CASE WHEN any(term IN $query_terms WHERE toLower(n.name) CONTAINS term) THEN 5
                      WHEN any(term IN $query_terms WHERE toLower(t.text) CONTAINS term) THEN 3
                      ELSE 1 END as relevance_score
            WITH n, collect(DISTINCT t.text)[0..3] as mentions, max(relevance_score) as max_relevance
            RETURN n.name as name,
                   n.label as label,
                   n.domain as domain,
                   n.semantic_type as semantic_type,
                   mentions,
                   max_relevance as relevance
            ORDER BY relevance DESC
            """
            
            result = session.run(query, 
                               label=label,
                               query_terms=query_terms)
            results = [dict(record) for record in result]
            
            # Save results
            self.save_query_results(
                results,
                "entities_by_type",
                {"label": label, "query_text": query_text}
            )
            return results

    def get_entity_context_with_relevance(self, entity_name: str, query_text: Optional[str] = None) -> Dict:
        """Get rich context around an entity with query-aware relevance scoring."""
        query_terms = self._process_query_text(query_text)
        
        with self.driver.session() as session:
            query = """
            MATCH (n:Entity {name: $entity_name})
            OPTIONAL MATCH path = (n)-[*1..2]-(related:Entity)
            WITH n, related, path,
                 CASE WHEN any(term IN $query_terms WHERE toLower(related.name) CONTAINS term) THEN 5
                      WHEN related.label IN ['PLACE', 'BUILDING', 'TIME', 'WEATHER'] THEN 2
                      ELSE 1 END as relevance_score
            OPTIONAL MATCH (related)-[:MENTIONED_IN]->(t:Text)
            WITH n, related, path, relevance_score,
                 collect(DISTINCT t.text)[0..3] as mentions
            RETURN n.name as central_entity,
                   n.label as entity_type,
                   n.domain as domain,
                   collect(DISTINCT {
                       name: related.name,
                       label: related.label,
                       relationship: [rel in relationships(path) | type(rel)],
                       distance: length(path),
                       mentions: mentions,
                       relevance: relevance_score
                   }) as related_entities
            """
            
            result = session.run(query, 
                               entity_name=entity_name,
                               query_terms=query_terms)
            results = dict(result.single())
            
            # Save results
            self.save_query_results(
                results,
                "entity_context",
                {"entity_name": entity_name, "query_text": query_text}
            )
            return results

    def get_query_aware_graph_context(self, query_text: str, max_entities: int = 10) -> List[Dict]:
        """Wrapper for the existing advanced query functionality."""
        from knowledge_graph.graph_advanced_queries import get_query_aware_graph_context
        results = get_query_aware_graph_context(self.driver, query_text, self.nlp, max_entities)
        
        # Save results
        self.save_query_results(
            results,
            "query_aware_context",
            {"query_text": query_text, "max_entities": max_entities}
        )
        return results

    def export_graph_data(self, output_file="knowledge_graph/graph_data.json"):
        """Export the entire graph data to JSON."""
        with self.driver.session() as session:
            # Export nodes separately by label
            entity_nodes = []
            text_nodes = []
            image_nodes = []

            nodes_result = session.run(
                "MATCH (n) RETURN id(n) as id, labels(n) as labels, properties(n) as properties"
            )
            for record in nodes_result:
                node_props = dict(record["properties"])
                node_id = record["id"]
                labels = record["labels"]

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
                RETURN id(r) as id, type(r) as type, 
                       startNode(r) as start, endNode(r) as end,
                       properties(r) as properties
            """)
            relationships = []
            for record in rels_result:
                rel_props = dict(record["properties"])
                relationships.append({
                    "id": record["id"],
                    "type": record["type"],
                    "start": record["start"].element_id,
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

            print(f"Graph data exported to {output_file} with structured JSON")


if __name__ == "__main__":
    # Example usage
    try:
        gq = GraphQuery("bolt://localhost:7687", "neo4j", "password")
        
        # Example query text
        query_text = "sunset view at the castle by the beach"
        
        print("\nQuerying and saving results...")
        
        # Run queries and save results
        related = gq.find_related_entities_with_context("castle", query_text)
        print("\nRelated entities:", json.dumps(related, indent=2))
        
        paths = gq.find_paths_with_relevance("castle", "beach", query_text)
        print("\nPaths between entities:", json.dumps(paths, indent=2))
        
        env_entities = gq.find_entities_by_type_and_relevance("ENVIRONMENT", query_text)
        print("\nEnvironment entities:", json.dumps(env_entities, indent=2))
        
        context = gq.get_entity_context_with_relevance("castle", query_text)
        print("\nEntity context:", json.dumps(context, indent=2))
        
        full_context = gq.get_query_aware_graph_context(query_text)
        print("\nFull query-aware context:", json.dumps(full_context, indent=2))
        
        print("\nAll query results have been saved to the knowledge_graph/query_results directory")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'gq' in locals():
            gq.close()