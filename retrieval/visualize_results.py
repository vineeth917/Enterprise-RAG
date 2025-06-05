import os
import json
import matplotlib.pyplot as plt
from PIL import Image

# Output folder for visualizations
output_dir = "retrieval/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Visualization helper
def visualize_matches(matches, title, output_file):
    fig, axes = plt.subplots(1, len(matches), figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    for ax, match in zip(axes, matches):
        image_path = match["image_path"]
        caption = match["caption"]
        try:
            img = Image.open(image_path)
            ax.imshow(img)
            ax.set_title(f"Score: {match['score']:.2f}")
            ax.set_xlabel(caption, fontsize=8, wrap=True)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", fontsize=8, ha="center", va="center")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"✅ Saved visualization to {output_file}")
    plt.close()

# Load and visualize text queries
with open("retrieval/results_text.json", "r") as f:
    text_data = json.load(f)

for i, entry in enumerate(text_data["text_queries"]):
    query = entry["query"]
    matches = entry["results"]
    output_file = os.path.join(output_dir, f"text_query_{i+1}.png")
    visualize_matches(matches, f"Text Query: {query}", output_file)

# Load and visualize single image query (no "image_queries" list, direct fields instead)
with open("retrieval/results_image.json", "r") as f:
    image_data = json.load(f)

image_query_path = image_data["image_query"]
matches = image_data["results"]
output_file = os.path.join(output_dir, "image_query_1.png")
visualize_matches(matches, f"Image Query: {image_query_path}", output_file)

print("\n✅ All visualizations saved in retrieval/visualizations/")
