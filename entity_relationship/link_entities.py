import json
from collections import defaultdict

def build_links(entity_file, output_file="entity_relationship/linked_entities.json"):
    with open(entity_file, "r") as f:
        data = json.load(f)

    links = []
    seen_links = set()
    
    for entry in data:
        for ent in entry.get("entities", []):  # In case entities is empty
            entity_text = ent["text"].strip()
            label = ent["label"].strip()
            domain = ent.get("domain", "unknown").strip()
            semantic_type = ent.get("semantic_type", "unknown").strip()
            source_text = entry["text"].strip()
            source_type = entry["source"]
            image_path = entry.get("image_path", None)

            # Use a unique key to avoid duplicate links
            link_key = f"{entity_text.lower()}|{label}|{source_text.lower()}"
            if link_key not in seen_links:
                seen_links.add(link_key)
                links.append({
                    "entity": entity_text,
                    "label": label,
                    "domain": domain,
                    "semantic_type": semantic_type,
                    "source_text": source_text,
                    "source_type": source_type,
                    "image_path": image_path
                })

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(links, f, indent=2)

    print(f"✅ Linked entities saved to {output_file} with domain & semantic_type fields!")

    # 🟩 Optional: Debug stats
    label_counts = defaultdict(int)
    for l in links:
        label_counts[l["label"]] += 1
    print("\n📊 Entity label counts:")
    for label, count in label_counts.items():
        print(f"  - {label}: {count}")

if __name__ == "__main__":
    build_links("entity_relationship/entities.json")

