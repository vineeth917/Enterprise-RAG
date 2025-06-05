import spacy
import json
import re
from tqdm import tqdm

# Load SpaCy model and add EntityRuler
nlp = spacy.load("en_core_web_lg")
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Expanded domain-specific patterns
patterns = [
    {"label": "ARCHITECTURE", "pattern": "coastal castle"},
    {"label": "ARCHITECTURE", "pattern": "seaside fortress"},
    {"label": "ARCHITECTURE", "pattern": "maritime castle"},
    {"label": "ARCHITECTURE", "pattern": "medieval castle"},
    {"label": "ARCHITECTURE", "pattern": "ancient fortress"},
    {"label": "ARCHITECTURE", "pattern": "historic landmark"},
    {"label": "ARCHITECTURE", "pattern": "stone bridge"},
    {"label": "ARCHITECTURE", "pattern": "old harbor"},
    {"label": "SCENIC_VIEW", "pattern": "dramatic ocean view"},
    {"label": "SCENIC_VIEW", "pattern": "urban skyline"},
    {"label": "SCENIC_VIEW", "pattern": "fishing port scene"},
    {"label": "TIME", "pattern": "golden hour"},
    {"label": "TIME", "pattern": "evening light"},
    {"label": "TIME", "pattern": "twilight glow"},
    {"label": "TIME", "pattern": "early evening light"},
    {"label": "TIME", "pattern": "late dusk"},
    {"label": "ENVIRONMENT", "pattern": "rugged coastline"},
    {"label": "ENVIRONMENT", "pattern": "rocky cliff"},
    {"label": "ENVIRONMENT", "pattern": "harbor view"}
]
ruler.add_patterns(patterns)

# Enriched domain keywords
domain_keywords = {
    "ARCHITECTURE": [
        "castle", "fortress", "citadel", "tower", "keep",
        "bridge", "harbor", "maritime castle", "seaside fortress"
    ],
    "TIME": [
        "sunset", "sunrise", "dawn", "dusk", "twilight", "golden hour",
        "evening light", "twilight glow", "early evening light", "late dusk"
    ],
    "SCENIC_VIEW": [
        "view", "scene", "landscape", "panorama", "vista", "scenery",
        "skyline", "harbor view"
    ],
    "ENVIRONMENT": [
        "ocean", "sea", "coastal", "shore", "waterfront", "cliff", "beach",
        "island", "harbor", "village"
    ],
    "ACTIVITY": [
        "walking", "hiking", "photographing", "exploring", "observing"
    ]
}

# Expanded regex patterns (with re.IGNORECASE and new synonyms)
regex_patterns = [
    r"\b(seaside|maritime|waterfront|ocean|riverside|lakeside)\s+(castle|fortress|citadel|keep|harbor)\b",
    r"\b(medieval|ancient|old|historic|ruined|abandoned)\s+(castle|fortress|citadel|tower|keep|harbor)\b",
    r"\b(evening|morning|golden hour|blue hour|dusk|twilight|twilight glow|sunset|sunrise|nightfall|daybreak|early evening light|late dusk)\b",
    r"\b(scenic|breathtaking|dramatic|panoramic|picturesque|romantic)\s+(view|scene|landscape|vista|skyline)\b",
    r"\b(coastal|riverside|lakeside|maritime|waterfront|harbor)\s+(view|scene|landscape|environment)\b"
]

# Merging similar terms
domain_merges = {
    "view": "SCENIC_VIEW",
    "scene": "SCENIC_VIEW",
    "landscape": "SCENIC_VIEW",
    "cityscape": "SCENIC_VIEW",
    "skyline": "SCENIC_VIEW",
    "seascape": "SCENIC_VIEW"
}

# ADDED 'LOC' to handle location-based entities
priority_labels = ["ARCHITECTURE", "SCENIC_VIEW", "ENVIRONMENT", "TIME", "ACTIVITY", "REGEX_MATCH", "GENERAL", "LOC"]

def extract_entities(text):
    doc = nlp(text)
    entities = []

    # SpaCy's NER and EntityRuler
    for ent in doc.ents:
        entities.append({
            "text": ent.text.strip(),
            "label": ent.label_
        })

    # Additional: Split multi-word domain matches
    for ent in entities.copy():
        if " " in ent["text"]:  # If it's a phrase
            for word in ent["text"].split():
                if word.lower() in domain_keywords.get(ent["label"], []):
                    entities.append({
                        "text": word,
                        "label": ent["label"]
                    })

    # Regex-based domain enrichment
    lower_text = text.lower()
    for pattern in regex_patterns:
        matches = re.findall(pattern, lower_text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = next((m for m in match if m), "")
            if match:
                label = "REGEX_MATCH"  # default
                for domain, keywords in domain_keywords.items():
                    if match.lower() in [k.lower() for k in keywords]:
                        label = domain
                        break
                # Context-based disambiguation
                if match.lower() == "view" and "review" in lower_text:
                    continue
                entities.append({
                    "text": match.strip(),
                    "label": label
                })

    #  Keyword-based enrichment
    for label, keywords in domain_keywords.items():
        for kw in keywords:
            if kw in lower_text:
                # Avoid duplicate "view" in review
                if kw == "view" and "review" in lower_text:
                    continue
                entities.append({
                    "text": kw,
                    "label": label
                })

    #  Merge similar synonyms (like view/scene/landscape → SCENIC_VIEW)
    entities = merge_similar_entities(entities)

    #  Deduplicate by priority
    entities = deduplicate_entities(entities)

    return entities

def merge_similar_entities(entities):
    for ent in entities:
        ent_text = ent["text"].lower()
        if ent_text in domain_merges:
            ent["label"] = domain_merges[ent_text]
    return entities

def deduplicate_entities(entities):
    seen = {}
    for ent in entities:
        key = ent["text"].lower()
        current_label = ent["label"]
        if key not in seen:
            seen[key] = ent
        else:
            existing_label = seen[key]["label"]
            if priority_labels.index(current_label) < priority_labels.index(existing_label):
                seen[key] = ent
    return list(seen.values())

def process_results(results_file, output_file="entity_relationship/entities.json"):
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
        
        entity_data = []
        
        # Handle image query format
        if "image_query" in data:
            # Process the image query itself
            if "image_query" in data:
                entity_data.append({
                    "source": "image_query",
                    "text": data["image_query"],
                    "entities": []  # No entities for image path
                })
            
            # Process results
            for match in tqdm(data["results"], desc="Extracting enriched entities"):
                caption = match["caption"]
                caption_entities = extract_entities(caption)
                entity_data.append({
                    "source": "caption",
                    "text": caption,
                    "image_path": match["image_path"],
                    "entities": caption_entities
                })
        
        # Handle text query format
        elif "text_queries" in data:
            for entry in tqdm(data["text_queries"], desc="Extracting enriched entities"):
                query = entry["query"]
                query_entities = extract_entities(query)
                entity_data.append({
                    "source": "query",
                    "text": query,
                    "entities": query_entities
                })

                for match in entry["results"]:
                    caption = match["caption"]
                    caption_entities = extract_entities(caption)
                    entity_data.append({
                        "source": "caption",
                        "text": caption,
                        "image_path": match["image_path"],
                        "entities": caption_entities
                    })
        else:
            raise ValueError("Unknown results format - missing 'image_query' or 'text_queries' key")

        # Save final enriched entity data
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(entity_data, f, indent=2)
        print(f"Final enriched & deduplicated entities saved to {output_file}")
    except Exception as e:
        print(f"Error processing {results_file}: {str(e)}")

if __name__ == "__main__":
    # Process results_image.json
    process_results("retrieval/results_image.json", "entity_relationship/entities_image.json")
    
    # Attempt to process results_test.json if it exists
    process_results("retrieval/results_text.json", "entity_relationship/entities_text.json")