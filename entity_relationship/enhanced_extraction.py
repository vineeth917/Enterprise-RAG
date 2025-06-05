# entity_relationship/enhanced_extraction.py

import json
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import spacy
from PIL import Image

# Load CLIP model & processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
nlp = spacy.load("en_core_web_sm")

def get_text_embedding(text):
    """Generate text embedding using CLIP, respecting 77-token limit to avoid indexing errors."""
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,  # 🔥 fix: ensure safe token length
        max_length=77
    )
    with torch.no_grad():
        embeds = model.get_text_features(**inputs)
        embeds /= embeds.norm(p=2, dim=-1, keepdim=True)
    return embeds[0].tolist()

def filter_relevant_entities(query_text, extracted_entities, threshold=0.3):
    query_embedding = get_text_embedding(query_text)
    relevant_entities = []

    for entity in extracted_entities:
        entity_text = entity.get('text', '').strip()
        if not entity_text:
            continue  # skip empty

        entity_embedding = get_text_embedding(entity_text)
        similarity = torch.cosine_similarity(
            torch.tensor(query_embedding).unsqueeze(0),
            torch.tensor(entity_embedding).unsqueeze(0)
        ).item()

        if similarity > threshold:
            entity['relevance_score'] = similarity
            relevant_entities.append(entity)

    # Deduplicate by (text, label) to avoid repeated entries
    deduped = []
    seen = set()
    for e in relevant_entities:
        key = f"{e.get('text', '').lower()}|{e.get('label', '').lower()}"
        if key not in seen:
            deduped.append(e)
            seen.add(key)

    return sorted(deduped, key=lambda x: x['relevance_score'], reverse=True)

def enhance_entity_extraction(query_text, process_results_fn, temp_results_file, entities_file="entity_relationship/entities.json"):
    # Extract entities from captions/queries
    process_results_fn(temp_results_file, entities_file)

    # Load extracted entities
    with open(entities_file, 'r') as f:
        raw_entities = json.load(f)

    # Filter and deduplicate
    relevant_entities = filter_relevant_entities(query_text, raw_entities)

    # Save only relevant, deduplicated entities
    with open(entities_file, 'w') as f:
        json.dump(relevant_entities, f, indent=2)

    print(f"✅ Enhanced & deduplicated {len(relevant_entities)} entities for query: '{query_text}'")
    return relevant_entities
