import json
import networkx as nx
import matplotlib.pyplot as plt

def main():
    # Load exported graph data
    with open("knowledge_graph/graph_data.json") as f:
        data = json.load(f)

    G = nx.DiGraph()

    # Add nodes with meaningful labels
    for node in data["nodes"]:
        node_id = node["id"]
        node_label = node["labels"][0] if node["labels"] else "Unknown"
        props = node["properties"]

        # Determine display label
        display_label = ""
        if "name" in props:
            display_label = props["name"]
        elif "text" in props:
            display_label = props["text"]
        elif "path" in props:
            display_label = props["path"].split("/")[-1]  # show image file name
        else:
            display_label = node_label

        G.add_node(node_id, label=display_label, node_type=node_label)

    # Add relationships
    for rel in data["relationships"]:
        start = rel["start"]
        end = rel["end"]
        rel_type = rel["type"]
        G.add_edge(start, end, label=rel_type)

    # Draw graph
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Draw edges
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrows=True, alpha=0.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=7)

    plt.title("Knowledge Graph Visualisation")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("knowledge_graph/graph_visualization.png", dpi=300, bbox_inches="tight")
    plt.show() 

if __name__ == "__main__":
    main()
