import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os
from datetime import datetime

class GraphVisualizer:
    def __init__(self):
        self.output_dir = "visualizations"
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_full_graph(self, graph_data_file: str = "graph_data.json"):
        """Visualize the full graph from graph_data.json"""
        with open(graph_data_file) as f:
            data = json.load(f)

        G = nx.DiGraph()

        # Add entity nodes with improved labels
        for entity in data["entities"]:
            node_id = entity["id"]
            props = entity["properties"]
            display_label = f"{props.get('name', 'Unknown')}\n({props.get('label', 'Entity')})"
            node_type = props.get("label", "Entity")
            G.add_node(node_id, 
                      label=display_label, 
                      node_type=node_type,
                      domain=props.get("domain", "unknown"),
                      semantic_type=props.get("semantic_type", "unknown"))

        # Add text and image nodes with improved labels
        for node_type in ["texts", "images"]:
            for node in data.get(node_type, []):
                node_id = node["id"]
                props = node["properties"]
                display_text = props.get("text", props.get("path", "Unknown"))
                display_label = display_text[:30] + "..." if len(display_text) > 30 else display_text
                G.add_node(node_id, 
                          label=display_label, 
                          node_type=node_type.upper())

        # Add relationships with improved edge styling
        for rel in data["relationships"]:
            start = rel["start"]
            end = rel["end"]
            rel_type = rel["type"]
            G.add_edge(start, end, 
                      label=rel_type,
                      weight=1.5 if rel_type == "MENTIONED_IN" else 1.0)

        self._draw_graph(G, "Full Knowledge Graph", "full_graph")

    def visualize_query_result(self, result_file: str):
        """Visualize a specific query result"""
        with open(result_file) as f:
            data = json.load(f)

        G = nx.DiGraph()
        query_type = data["query_type"]
        
        if query_type == "related_entities":
            self._visualize_related_entities(G, data["results"])
        elif query_type == "entity_paths":
            self._visualize_entity_paths(G, data["results"])
        elif query_type == "entities_by_type":
            self._visualize_entities_by_type(G, data["results"])
        elif query_type == "entity_context":
            self._visualize_entity_context(G, data["results"])
        elif query_type == "query_aware_context":
            self._visualize_query_aware_context(G, data["results"])

        title = f"Query Result: {query_type}"
        filename = f"query_{query_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._draw_graph(G, title, filename)

    def _visualize_related_entities(self, G: nx.DiGraph, results: List[Dict]):
        """Visualize related entities results"""
        for result in results:
            source = result["name"]
            G.add_node(source, 
                      label=source,
                      node_type=result["label"],
                      relevance=result["relevance_score"])
            
            # Add mention nodes
            for i, mention in enumerate(result.get("mentions", [])):
                mention_id = f"{source}_mention_{i}"
                G.add_node(mention_id, 
                          label=mention[:20] + "..." if len(mention) > 20 else mention,
                          node_type="MENTION")
                G.add_edge(source, mention_id, label="MENTIONED_IN")

    def _visualize_entity_paths(self, G: nx.DiGraph, results: List[Dict]):
        """Visualize entity paths results"""
        for result in results:
            nodes = result["node_names"]
            rels = result["relationship_types"]
            
            # Add all non-null nodes first
            valid_nodes = [n for n in nodes if n]
            for node in valid_nodes:
                G.add_node(node, label=node, node_type="ENTITY")
            
            # Connect nodes, skipping over nulls
            last_valid_node = None
            accumulated_rels = []
            
            for i, node in enumerate(nodes):
                if node:
                    if last_valid_node:
                        # Connect to previous valid node with accumulated relationships
                        rel_label = " -> ".join(accumulated_rels) if accumulated_rels else "RELATED_TO"
                        G.add_edge(last_valid_node, node, 
                                 label=rel_label,
                                 weight=result.get("path_relevance", 1.0))
                    last_valid_node = node
                    accumulated_rels = []
                elif i < len(rels):
                    accumulated_rels.append(rels[i])

    def _visualize_entity_context(self, G: nx.DiGraph, result: Dict):
        """Visualize entity context results"""
        central = result["central_entity"]
        G.add_node(central, 
                  label=central,
                  node_type=result["entity_type"])
        
        for related in result["related_entities"]:
            name = related["name"]
            G.add_node(name, 
                      label=name,
                      node_type=related["label"],
                      relevance=related["relevance"])
            
            for rel in related["relationship"]:
                G.add_edge(central, name, label=rel)

    def _visualize_query_aware_context(self, G: nx.DiGraph, results: List[Dict]):
        """Visualize query-aware context results"""
        for result in results:
            entity = result["entity"]
            G.add_node(entity, 
                      label=entity,
                      node_type=result["label"],
                      relevance=result["relevance"])
            
            # Add related entities
            for related in result.get("related_entities", []):
                name = related["name"]
                G.add_node(name, 
                          label=name,
                          node_type=related["label"],
                          strength=related["strength"])
                G.add_edge(entity, name, weight=related["strength"])

    def _visualize_entities_by_type(self, G: nx.DiGraph, results: List[Dict]):
        """Visualize entities grouped by their types"""
        # Create a central node for each entity type
        type_nodes = {}
        
        for result in results:
            entity_type = result["label"]
            if entity_type not in type_nodes:
                type_node_id = f"TYPE_{entity_type}"
                type_nodes[entity_type] = type_node_id
                G.add_node(type_node_id, 
                          label=entity_type,
                          node_type="TYPE",
                          size=2000)
            
            # Add entity node
            entity_name = result["name"]
            G.add_node(entity_name,
                      label=entity_name,
                      node_type="ENTITY",
                      relevance=result.get("relevance", 1.0))
            
            # Connect entity to its type
            G.add_edge(type_nodes[entity_type], 
                      entity_name,
                      label="IS_TYPE")
            
            # Add mentions if present
            for i, mention in enumerate(result.get("mentions", [])):
                mention_id = f"{entity_name}_mention_{i}"
                G.add_node(mention_id,
                          label=mention[:20] + "..." if len(mention) > 20 else mention,
                          node_type="MENTION")
                G.add_edge(entity_name, mention_id, label="MENTIONED_IN")

    def _draw_graph(self, G: nx.DiGraph, title: str, filename: str):
        """Draw and save the graph visualization"""
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw nodes with different colors based on type
        node_colors = {
            'ENTITY': 'skyblue',
            'TEXT': 'lightgreen',
            'IMAGE': 'lightpink',
            'MENTION': 'lightyellow',
            'TYPE': 'lightcoral'  # Add color for type nodes
        }

        # Draw nodes by type
        for node_type, color in node_colors.items():
            nodes = [n for n, attr in G.nodes(data=True) 
                    if attr.get('node_type', '').upper().startswith(node_type)]
            if nodes:
                nx.draw_networkx_nodes(G, pos, 
                                     nodelist=nodes,
                                     node_color=color, 
                                     node_size=1000, 
                                     alpha=0.7)

        # Draw labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        # Draw edges with weights if available
        edge_weights = nx.get_edge_attributes(G, 'weight')
        if edge_weights:
            edges = [(u, v) for (u, v, d) in G.edges(data=True)]
            weights = [d.get('weight', 1.0) for (u, v, d) in G.edges(data=True)]
            nx.draw_networkx_edges(G, pos, 
                                 edgelist=edges,
                                 width=weights,
                                 edge_color='gray',
                                 arrows=True)
        else:
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, 
                                   edge_labels=edge_labels,
                                   font_size=6,
                                   font_color='red')

        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    visualizer = GraphVisualizer()
    
    # Visualize query results
    query_results_dir = "query_results"
    for result_file in os.listdir(query_results_dir):
        if result_file.endswith('.json'):
            full_path = os.path.join(query_results_dir, result_file)
            visualizer.visualize_query_result(full_path)
            
    # Only visualize full graph if graph_data.json exists
    if os.path.exists("graph_data.json"):
        visualizer.visualize_full_graph()