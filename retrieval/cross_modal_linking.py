# retrieval/cross_modal_linking.py

import torch

def extract_visual_concepts(query_text, nlp):
    visual_keywords = [
        "castle", "beach", "sunset", "mountain", "forest", "city", "building", "house",
        "car", "person", "animal", "tree", "water", "sky", "cloud", "landscape",
        "portrait", "night", "day", "harbor", "bridge", "valley", "river", "lake", "sea"
    ]
    doc = nlp(query_text.lower())
    return [token.text for token in doc if token.lemma_ in visual_keywords or token.text in visual_keywords]

def enhance_cross_modal_linking(graph_driver, nlp, get_text_embedding, extract_entities, query_text, pinecone_results, query_entities):
    """Enhanced cross-modal linking with improved integration."""
    image_context = []
    visual_terms = extract_visual_concepts(query_text, nlp)

    # Precompute query embedding for efficiency
    query_embedding = get_text_embedding(query_text)

    with graph_driver.session() as session:
        # 1. Direct entity-based linking with confidence scoring
        for entity in query_entities[:5]:
            result = session.run("""
                MATCH (e:Entity {name: $entity})-[:MENTIONED_IN]->(t:Text)-[:DESCRIBES]->(i:Image)
                WITH e, t, i,
                     CASE WHEN e.semantic_type IN ['landmark', 'location', 'object'] THEN 1.2
                          WHEN e.semantic_type IN ['time_reference', 'weather_condition'] THEN 1.1
                          ELSE 1.0 END as type_boost
                RETURN DISTINCT 
                    i.path AS image_path, 
                    i.caption AS caption, 
                    t.content AS context,
                    type_boost,
                    e.semantic_type AS entity_type
                LIMIT 3
            """, entity=entity)

            for record in result:
                confidence_score = record["type_boost"]
                image_context.append({
                    "type": "direct_entity",
                    "entity": entity,
                    "entity_type": record["entity_type"],
                    "image_path": record["image_path"],
                    "caption": record["caption"],
                    "context": record["context"],
                    "confidence": confidence_score
                })

        # 2. Semantic similarity-based linking with improved scoring
        for match in pinecone_results[:3]:
            caption = match['metadata']['caption']
            caption_embedding = get_text_embedding(caption)

            # Calculate semantic similarity
            similarity = torch.cosine_similarity(
                torch.tensor(query_embedding).unsqueeze(0),
                torch.tensor(caption_embedding).unsqueeze(0)
            ).item()

            # Extract and score entities
            if similarity > 0.5:
                caption_entities = extract_entities(caption)
                entity_overlap = len(set(e["text"].lower() for e in caption_entities) & 
                                   set(query_entities))
                
                # Combined score with entity overlap boost
                combined_score = similarity * (1 + 0.2 * entity_overlap)

                image_context.append({
                    "type": "semantic_match",
                    "similarity": similarity,
                    "combined_score": combined_score,
                    "image_path": match['metadata']['image_path'],
                    "caption": caption,
                    "entities": caption_entities,
                    "entity_overlap": entity_overlap
                })

        # 3. Visual term-based linking with contextual relevance
        if visual_terms:
            result = session.run("""
                MATCH (i:Image)
                WHERE any(term IN $visual_terms WHERE toLower(i.caption) CONTAINS toLower(term))
                WITH i, 
                     [term IN $visual_terms WHERE toLower(i.caption) CONTAINS toLower(term)] AS matched_terms,
                     size([term IN $visual_terms WHERE toLower(i.caption) CONTAINS toLower(term)]) as term_count
                RETURN 
                    i.path AS image_path, 
                    i.caption AS caption,
                    matched_terms,
                    term_count,
                    (term_count * 1.0 / size($visual_terms)) as coverage_score
                ORDER BY coverage_score DESC
                LIMIT 5
            """, visual_terms=visual_terms)

            for record in result:
                coverage_score = record["coverage_score"]
                image_context.append({
                    "type": "visual_concept",
                    "image_path": record["image_path"],
                    "caption": record["caption"],
                    "matched_terms": record["matched_terms"],
                    "term_count": record["term_count"],
                    "coverage_score": coverage_score
                })

        # 4. Deduplicate and sort results
        seen_images = set()
        unique_context = []
        
        for item in sorted(image_context, 
                         key=lambda x: x.get("combined_score", 
                                           x.get("confidence",
                                                x.get("coverage_score", 0))), 
                         reverse=True):
            if item["image_path"] not in seen_images:
                seen_images.add(item["image_path"])
                unique_context.append(item)

        return unique_context
