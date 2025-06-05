# knowledge_graph/graph_advanced_queries.py

def get_query_aware_graph_context(graph_driver, query_text, nlp, max_entities=10):
    context = []

    # 🔹 Normalize input
    if isinstance(query_text, list):
        query_text = " ".join(query_text)
    elif not isinstance(query_text, str):
        query_text = str(query_text)

    if not query_text.strip():
        return context

    # 🔹 Refine query terms: skip numeric-only or short stopword-like tokens
    query_terms = [
        token.lemma_.lower() for token in nlp(query_text)
        if not token.is_stop and not token.is_punct and len(token.text) > 2 and not token.text.isdigit()
    ]

    print("DEBUG: query_terms ->", query_terms)

    if not query_terms:
        return context

    with graph_driver.session() as session:
        # Entity & mention extraction
        result = session.run("""
            MATCH (e:Entity)-[:MENTIONED_IN]->(t:Text)
            WITH e, t,
                 CASE WHEN any(term IN $query_terms WHERE toLower(e.name) CONTAINS term) THEN 5
                      WHEN any(term IN $query_terms WHERE toLower(t.text) CONTAINS term) THEN 3
                      WHEN t.source_type = 'query' THEN 2
                      ELSE 1 END as base_relevance,
                 CASE WHEN e.label IN ['PLACE', 'BUILDING', 'TIME', 'WEATHER'] THEN 2
                      WHEN e.label IN ['PERSON', 'ORG'] THEN 1
                      ELSE 1.5 END as type_relevance
            WHERE base_relevance >= 1
            WITH e, t, (base_relevance * type_relevance) as relevance_score
            RETURN e.name AS entity, e.label AS label,
                   collect(DISTINCT t.text)[0..3] AS sample_mentions,
                   collect(DISTINCT t.caption)[0..3] AS sample_captions,
                   collect(DISTINCT t.location)[0..3] AS sample_locations,
                   collect(DISTINCT t.image_path)[0..3] AS sample_images,
                   count(t) AS mention_count,
                   max(relevance_score) AS relevance
            ORDER BY relevance DESC, mention_count DESC
            LIMIT $max_entities
        """, query_terms=query_terms, max_entities=max_entities)

        for record in result:
            related_result = session.run("""
                MATCH (e1:Entity {name: $name})-[:MENTIONED_IN]->(t:Text)<-[:MENTIONED_IN]-(e2:Entity)
                WHERE e1 <> e2
                WITH e2, t,
                     CASE WHEN any(term IN $query_terms WHERE toLower(t.text) CONTAINS term) THEN 3 ELSE 1 END as context_relevance,
                     count(*) as co_occurrence
                RETURN e2.name AS related_entity, e2.label AS related_label,
                       sum(context_relevance * co_occurrence) AS weighted_strength
                ORDER BY weighted_strength DESC
                LIMIT 5
            """, name=record["entity"], query_terms=query_terms)

            related_entities = [
                {"name": r["related_entity"], "label": r["related_label"], "strength": r["weighted_strength"]}
                for r in related_result
            ]

            context.append({
                "entity": record["entity"],
                "label": record["label"],
                "mentions": record["sample_mentions"],
                "captions": record["sample_captions"],
                "locations": record["sample_locations"],
                "images": record["sample_images"],
                "mention_count": record["mention_count"],
                "relevance": record["relevance"],
                "related_entities": related_entities
            })

    return context