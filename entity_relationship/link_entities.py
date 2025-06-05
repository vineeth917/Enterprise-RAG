import json
from collections import defaultdict

def build_links_from_file(entity_file):
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
                    "image_path": image_path,
                    "source_file": entity_file  # Added to track which file the entity came from
                })
    
    return links, seen_links

def build_combined_links(entity_files, output_file="entity_relationship/linked_entities.json"):
    all_links = []
    all_seen_links = set()
    
    # Process each entity file
    for entity_file in entity_files:
        try:
            links, seen_links = build_links_from_file(entity_file)
            all_links.extend(links)
            all_seen_links.update(seen_links)
            print(f"Processed {len(links)} entities from {entity_file}")
        except Exception as e:
            print(f"Error processing {entity_file}: {str(e)}")
            continue

    # Save combined links
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_links, f, indent=2)

    print(f"\nCombined linked entities saved to {output_file}")
    print(f"Total unique entities: {len(all_links)}")

    # Debug stats
    label_counts = defaultdict(int)
    file_counts = defaultdict(int)
    
    for link in all_links:
        label_counts[link["label"]] += 1
        file_counts[link["source_file"]] += 1
    
    print("\nEntity label counts:")
    for label, count in label_counts.items():
        print(f"  - {label}: {count}")
    
    print("\nEntities per source file:")
    for file, count in file_counts.items():
        print(f"  - {file}: {count}")

if __name__ == "__main__":
    entity_files = [
        "entity_relationship/entities_image.json",
        "entity_relationship/entities_test.json"
    ]
    build_combined_links(entity_files)