import re
import os
import json
import tempfile
from dotenv import load_dotenv
import torch
from tqdm import tqdm
from PIL import Image
import requests
import spacy
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity

from pinecone import Pinecone
from neo4j import GraphDatabase
from transformers import CLIPProcessor, CLIPModel

# Import your existing modules
from entity_relationship.extract_entities import extract_entities, process_results
from entity_relationship.link_entities import build_links
from knowledge_graph.build_graph import GraphBuilder
from knowledge_graph.query_graph import GraphQuery
from entity_relationship.enhanced_extraction import enhance_entity_extraction
from knowledge_graph.graph_advanced_queries import get_query_aware_graph_context
from retrieval.cross_modal_linking import enhance_cross_modal_linking

from sentence_transformers import SentenceTransformer
# Load environment variables
load_dotenv()
PINECONE_API_KEY_TEXT = os.environ["PINECONE_API_KEY_TEXT"]
PINECONE_API_KEY_IMAGE = os.environ["PINECONE_API_KEY_IMAGE"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Gemini endpoint (Gemini 2.0 Flash)
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

pc_text = Pinecone(api_key=PINECONE_API_KEY_TEXT)
pc_image = Pinecone(api_key=PINECONE_API_KEY_IMAGE)
index_text = pc_text.Index("cc3m-new1")         # 1024-dimensional for text
index_image = pc_image.Index("cc3m-crossmodal") # 512-dimensional for image
graph_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Initialize CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer("intfloat/e5-large-v2")

# Initialize Spacy
# Load SpaCy model
nlp = spacy.load("en_core_web_lg")


def enhanced_environmental_query_expansion(query: str) -> str:
    """Comprehensive environment and landscape synonym expansion with contextual combinations"""
    
    environmental_expansions = {
        # Architecture + Environment
        "coastal castle": [
            "seaside fortress", "ocean castle", "maritime castle", "waterfront castle",
            "clifftop castle", "promontory fortress", "headland stronghold", "coastal citadel"
        ],
        "castle": [
            "fortress", "citadel", "stronghold", "keep", "tower", "battlement",
            "medieval structure", "stone fortification", "defensive structure"
        ],
        
        # Coastal & Water Features
        "coastal": [
            "seaside", "waterfront", "oceanfront", "shoreline", "maritime", 
            "littoral", "seaboard", "coast", "beach", "strand"
        ],
        "ocean": [
            "sea", "waters", "marine", "maritime", "nautical", "aquatic",
            "tidal", "pelagic", "oceanic", "seawater"
        ],
        "bay": ["cove", "inlet", "harbor", "harbour", "estuary", "sound", "gulf"],
        "beach": ["shore", "strand", "coastline", "seashore", "waterfront"],
        
        # Terrain & Landscape Features  
        "cliff": ["bluff", "escarpment", "precipice", "promontory", "headland", "crag"],
        "hill": ["elevation", "rise", "mound", "knoll", "summit", "peak", "ridge"],
        "rocks": ["stone", "boulder", "outcrop", "rocky", "craggy", "rugged"],
        "landscape": ["scenery", "terrain", "topography", "vista", "panorama"],
        
        # Lighting & Atmosphere
        "sunset": [
            "dusk", "twilight", "golden hour", "evening glow", "sundown",
            "eventide", "gloaming", "vespers", "nightfall"
        ],
        "golden hour": ["magic hour", "warm light", "soft light", "amber light"],
        "atmospheric": ["moody", "dramatic", "ethereal", "mystical", "romantic"],
        
        # Visual & Perspective Terms
        "view": [
            "vista", "panorama", "scenery", "outlook", "prospect", "perspective",
            "scene", "landscape", "seascape", "tableau"
        ],
        "overlook": ["vantage point", "viewpoint", "lookout", "observation point"],
        
        # Weather & Sky
        "sky": ["heavens", "firmament", "atmosphere", "celestial", "aerial"],
        "clouds": ["cloudscape", "overcast", "cumulus", "atmospheric"],
        "horizon": ["skyline", "distant view", "far vista", "edge"],
        
        # Natural Elements
        "waves": ["surf", "breakers", "whitecaps", "swells", "tide", "spray"],
        "wind": ["breeze", "gale", "maritime wind", "sea breeze", "coastal air"],
        "mist": ["fog", "haze", "marine layer", "atmospheric moisture"],
        
        # Architectural Details
        "tower": ["turret", "keep", "watchtower", "spire", "minaret", "campanile"],
        "stone": ["masonry", "stonework", "ashlar", "granite", "limestone"],
        "medieval": ["historic", "ancient", "period", "heritage", "traditional"]
    }
    
    expanded_query = query.lower()
    expansion_terms = []
    
    # Apply expansions based on detected terms
    for key, synonyms in environmental_expansions.items():
        if key in expanded_query:
            expansion_terms.extend(synonyms[:4])  # Limit to prevent over-expansion
    
    # Add contextual combinations for specific scenarios
    if "castle" in expanded_query and "coastal" in expanded_query:
        expansion_terms.extend([
            "fortress by the sea", "medieval coastal defense", "maritime stronghold",
            "clifftop fortification", "seaside architecture"
        ])
    
    if "sunset" in expanded_query and ("view" in expanded_query or "vista" in expanded_query):
        expansion_terms.extend([
            "evening panorama", "golden light scenery", "atmospheric lighting",
            "dramatic sky", "warm evening glow"
        ])
    
    # Combine original query with expansions
    final_query = f"{query} {' '.join(expansion_terms)}"
    return final_query


def calculate_image_similarity(image_embedding_1: np.ndarray, image_embedding_2: np.ndarray) -> float:
    """Calculate cosine similarity between two image embeddings"""
    if isinstance(image_embedding_1, list):
        image_embedding_1 = np.array(image_embedding_1).reshape(1, -1)
    if isinstance(image_embedding_2, list):
        image_embedding_2 = np.array(image_embedding_2).reshape(1, -1)
    
    return cosine_similarity(image_embedding_1, image_embedding_2)[0][0]


def verify_dimensions(embedding, expected_dim, source):
    """Verify embedding dimensions."""
    actual_dim = len(embedding)
    if actual_dim != expected_dim:
        raise ValueError(f"Dimension mismatch for {source}. Expected {expected_dim}, got {actual_dim}")


def get_text_embedding(query_text: str) -> List[float]:
    """Generate 1024-dim text embedding using SentenceTransformer (not CLIP!)"""
    # Use SentenceTransformer for 1024-dim embeddings
    embedding = text_model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
    if embedding.ndim > 1:
        emb = embedding[0].tolist()
    else:
        emb = embedding.tolist()
    verify_dimensions(emb, 1024, "text embedding")
    return emb


def get_image_embedding(image_path: str) -> List[float]:
    """Generate 512-dim image embedding using CLIP"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt", padding=True)
    with torch.no_grad():
        embeds = model.get_image_features(**inputs)
        embeds /= embeds.norm(p=2, dim=-1, keepdim=True)
    emb = embeds[0].tolist()
    verify_dimensions(emb, 512, "image embedding")
    return emb


def enhanced_multimodal_search(query_text: Optional[str] = None, 
                             query_image_path: Optional[str] = None, 
                             similarity_threshold: float = 0.75) -> List[Dict]:
    """Enhanced search using both text and image similarity"""
    
    results = []
    
    # Text-based search (1024-dim) using e5-large-v2
    if query_text:
        expanded_query = enhanced_environmental_query_expansion(query_text)
        print(f"🌿 Enhanced Environmental Expansion: {expanded_query[:100]}...")
        
        # Get and verify text embedding
        text_embedding = get_text_embedding(expanded_query)
        
        # Query TEXT index
        text_results = index_text.query(vector=text_embedding, top_k=5, include_metadata=True)["matches"]
        
        # Annotate results
        for result in text_results:
            result['search_type'] = 'text_semantic'
            result['query_expansion'] = 'environmental_enhanced'
        
        results.extend(text_results)
    
    # Image-based search (512-dim) using CLIP
    if query_image_path:
        # Get and verify image embedding
        query_image_embedding = get_image_embedding(query_image_path)
        
        # Query IMAGE index
        image_results = index_image.query(vector=query_image_embedding, top_k=5, include_metadata=True)["matches"]
        
        # Enhance with visual similarity
        enhanced_image_results = []
        for result in image_results:
            result['search_type'] = 'image_visual'
            result_image_path = result['metadata'].get('image_path')
            
            if result_image_path:
                try:
                    result_image_embedding = get_image_embedding(result_image_path)
                    visual_similarity = calculate_image_similarity(query_image_embedding, result_image_embedding)
                    result['visual_similarity'] = visual_similarity
                    result['enhanced_score'] = (result.get('score', 0) * 0.6) + (visual_similarity * 0.4)
                except Exception as e:
                    print(f"⚠️ Warning: Could not calculate visual similarity for {result_image_path}: {e}")
                    result['visual_similarity'] = result.get('score', 0)
                    result['enhanced_score'] = result.get('score', 0)
            
            enhanced_image_results.append(result)
        
        results.extend(enhanced_image_results)

    # Deduplicate and sort
    unique_results = {}
    for result in results:
        image_path = result['metadata']['image_path']
        if image_path not in unique_results:
            unique_results[image_path] = result
        else:
            current_score = result.get('enhanced_score', result.get('visual_similarity', result.get('score', 0)))
            existing_score = unique_results[image_path].get('enhanced_score',
                                                             unique_results[image_path].get('visual_similarity',
                                                                                             unique_results[image_path].get('score', 0)))
            if current_score > existing_score:
                unique_results[image_path] = result

    # Sort by enhanced score
    sorted_results = sorted(
        unique_results.values(),
        key=lambda x: x.get('enhanced_score', x.get('visual_similarity', x.get('score', 0))),
        reverse=True
    )

    # Filter by visual similarity threshold for image queries
    if query_image_path:
        sorted_results = [r for r in sorted_results if r.get('visual_similarity', 0) >= similarity_threshold]

    return sorted_results


def cross_modal_image_similarity_analysis(query_text: str, 
                                        query_image_path: str, 
                                        pinecone_results: List[Dict]) -> List[Dict]:
    """Analyze cross-modal connections using image similarity"""
    
    cross_modal_connections = []
    
    if not query_image_path:
        return cross_modal_connections
    
    try:
        query_image_embedding = get_image_embedding(query_image_path)
    except Exception as e:
        print(f"⚠️ Error loading query image {query_image_path}: {e}")
        return cross_modal_connections
    
    for result in pinecone_results:
        result_image_path = result['metadata'].get('image_path')
        caption = result['metadata'].get('caption', '')
        
        if result_image_path:
            try:
                result_image_embedding = get_image_embedding(result_image_path)
                visual_similarity = calculate_image_similarity(
                    query_image_embedding, result_image_embedding
                )
                
                # Text-image cross-modal analysis
                text_embedding = get_text_embedding(query_text)
                text_to_result_image_sim = calculate_image_similarity(
                    text_embedding, result_image_embedding
                )
                
                connection = {
                    'type': 'visual_similarity',
                    'query_image': query_image_path,
                    'result_image': result_image_path,
                    'caption': caption,
                    'visual_similarity': visual_similarity,
                    'text_to_image_similarity': text_to_result_image_sim,
                    'cross_modal_score': (visual_similarity * 0.6) + (text_to_result_image_sim * 0.4),
                    'original_score': result.get('score', 0)
                }
                
                # Only include high-quality cross-modal connections
                if connection['cross_modal_score'] > 0.7:
                    cross_modal_connections.append(connection)
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not process image similarity for {result_image_path}: {e}")
    
    return sorted(cross_modal_connections, key=lambda x: x['cross_modal_score'], reverse=True)


def enhanced_cross_modal_linking_with_images(query_text: str, 
                                             query_image_path: Optional[str], 
                                             pinecone_results: List[Dict], 
                                             query_entities: List[str]) -> List[Dict]:
    """
    Enhanced cross-modal linking that includes:
    - Entity-based matching in captions
    - Image similarity-based matching
    - Nearest-neighbor fallback if image context is too sparse
    """
    cross_modal_connections = []

    # 1️⃣ Entity-based connections (text & entity match in captions)
    for result in pinecone_results:
        caption = result['metadata']['caption']
        image_path = result['metadata']['image_path']

        for entity in query_entities:
            if entity.lower() in caption.lower():
                cross_modal_connections.append({
                    'type': 'direct_entity',
                    'entity': entity,
                    'caption': caption,
                    'image_path': image_path,
                    'confidence': 'high',
                    'score': result.get('score', 0)
                })

    # 2️⃣ Image similarity-based connections (if query image provided)
    if query_image_path:
        image_similarity_connections = cross_modal_image_similarity_analysis(
            query_text, query_image_path, pinecone_results
        )
        cross_modal_connections.extend(image_similarity_connections)

    # 3️⃣ Deduplicate and sort connections
    unique_connections = []
    seen_combinations = set()

    for conn in cross_modal_connections:
        identifier = (
            conn.get('type', ''),
            conn.get('image_path', ''),
            conn.get('entity', conn.get('visual_similarity', ''))
        )

        if identifier not in seen_combinations:
            seen_combinations.add(identifier)
            unique_connections.append(conn)

    # 4️⃣ Fallback to nearest-neighbor image if context is too sparse
    if query_image_path and len(unique_connections) < 3:
        print("⚠️ Sparse image context found, applying nearest-neighbor fallback...")
        # Get the embedding of the query image
        query_image_embedding = get_image_embedding(query_image_path)
        fallback_results = index_image.query(
            vector=query_image_embedding,
            top_k=3,
            include_metadata=True
        )["matches"]

        if fallback_results:
            fallback_item = fallback_results[0]
            fallback_conn = {
                'type': 'nearest_neighbor_fallback',
                'caption': fallback_item['metadata']['caption'],
                'image_path': fallback_item['metadata']['image_path'],
                'similarity': fallback_item.get('score', 0)
            }
            unique_connections.append(fallback_conn)
            print(f"✅ Fallback image found: {fallback_item['metadata']['caption']}")

    # Final sorted list
    return sorted(
        unique_connections,
        key=lambda x: x.get('cross_modal_score', x.get('score', 0)),
        reverse=True
    )


def search_pinecone(query_text: Optional[str] = None, 
                   image_path: Optional[str] = None, 
                   top_k: int = 5) -> List[Dict]:
    """Search Pinecone index with text or image query"""
    if query_text:
        print(f"🔎 Searching Pinecone for TEXT: '{query_text[:50]}...'")
        query_embedding = get_text_embedding(query_text)
        results = index_text.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    elif image_path:
        print(f"🔎 Searching Pinecone for IMAGE: '{image_path}'")
        query_embedding = get_image_embedding(image_path)
        results = index_image.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    else:
        raise ValueError("Provide either query_text or image_path for search!")

    return results["matches"]


def create_temp_results_file(query_text: str, pinecone_results: List[Dict]) -> str:
    """Create a temporary results file in the format expected by existing modules"""
    temp_data = {
        "text_queries": [{
            "query": query_text,
            "results": []
        }]
    }
    
    for match in pinecone_results:
        temp_data["text_queries"][0]["results"].append({
            "caption": match["metadata"]["caption"],
            "image_path": match["metadata"]["image_path"],
            "score": match.get("score", 0.0)
        })
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(temp_data, temp_file, indent=2)
    temp_file.close()
    
    return temp_file.name


def process_entities_pipeline(query_text: str, pinecone_results: List[Dict]) -> List[Dict]:
    """Process entities using the enhanced extraction pipeline"""
    temp_results_file = create_temp_results_file(query_text, pinecone_results)
    try:
        relevant_entities = enhance_entity_extraction(
            query_text, process_results_fn=process_results, temp_results_file=temp_results_file
        )
        return relevant_entities
    finally:
        os.unlink(temp_results_file)  # cleanup


def get_graph_context(graph_driver, query_text: str, nlp) -> List[Dict]:
    """Get context from knowledge graph"""
    return get_query_aware_graph_context(graph_driver, query_text, nlp)


def get_image_context(query_text: str, 
                     pinecone_results: List[Dict], 
                     query_entities: List[str]) -> List[Dict]:
    """Get enhanced cross-modal image context"""
    return enhance_cross_modal_linking(
        graph_driver, nlp, get_text_embedding, extract_entities,
        query_text, pinecone_results, query_entities
    )


def filter_relevant_context(context_items: List[Dict], query: str) -> List[Dict]:
    """Filter context items for relevance to query"""
    query_terms = set(query.lower().split())
    relevant_context = []
    
    for item in context_items:
        content = item.get('content', '').lower()
        overlap = len(query_terms.intersection(set(content.split())))
        if overlap > 0 or any(t in content for t in ['sunset', 'castle', 'coastal', 'view']):
            relevant_context.append(item)
    
    return relevant_context


def filter_entities_by_domain(entities: List[Dict], 
                            target_domain: str = "architecture_landscape") -> List[Dict]:
    """Filter entities by domain relevance"""
    relevant_labels = {'TIME', 'DATE', 'GPE', 'FAC', 'PERSON', 'ORG'}
    domain_keywords = {'castle', 'sunset', 'coastal', 'view', 'architecture', 'landscape'}
    
    filtered = []
    for entity in entities:
        entity_name = entity.get('text', '').lower() or entity.get('entity', '').lower()
        if (entity.get('label') in relevant_labels and 
            any(keyword in entity_name for keyword in domain_keywords)):
            filtered.append(entity)
    
    return filtered


def aggregate_entities(entities: List[Dict]) -> List[Dict]:
    """Deduplicate and aggregate entity mentions with counts"""
    entity_texts = [ent.get("text", ent.get("entity", "")).lower() for ent in entities]
    entity_counts = Counter(entity_texts)
    unique_entities = [{"entity": ent, "count": count} 
                      for ent, count in entity_counts.items() if ent]
    return unique_entities


def aggregate_all_entities(relevant_entities: List[Dict]) -> List[Dict]:
    """Aggregate entities from all processed entries"""
    all_entities = []
    for entry in relevant_entities:
        all_entities.extend(entry.get("entities", []))
    return aggregate_entities(all_entities)


def deduplicate_cross_modal_context(image_context: List[Dict]) -> List[Dict]:
    """Deduplicate cross-modal context entries to avoid repetition"""
    seen_combinations = set()
    deduplicated = []
    
    for item in image_context:
        if item.get('type') == 'direct_entity':
            identifier = (item['type'], item['entity'], item['image_path'])
        elif item.get('type') == 'semantic_match':
            identifier = (item['type'], item['image_path'], item.get('similarity', 0))
        elif item.get('type') == 'visual_concept':
            identifier = (item['type'], item['image_path'], tuple(item.get('matched_terms', [])))
        else:
            identifier = (item.get('type', 'unknown'), item.get('image_path', ''), str(item))
        
        if identifier not in seen_combinations:
            seen_combinations.add(identifier)
            deduplicated.append(item)
    
    return deduplicated


def filter_by_relevance_threshold(pinecone_results: List[Dict], 
                                threshold: float = 0.65) -> List[Dict]:
    """Filter Pinecone results by relevance threshold to reduce noise"""
    return [result for result in pinecone_results 
            if result.get('enhanced_score', result.get('score', 0)) >= threshold]


def format_multimodal_context_for_gemini(query: str, 
                                       results: List[Dict], 
                                       cross_modal_connections: List[Dict]) -> str:
    """Format context including image similarity information for Gemini"""
    
    context_parts = []
    
    # Query Analysis
    context_parts.append("🎯 ENHANCED QUERY ANALYSIS:")
    context_parts.append(f"   Original Query: {query}")
    context_parts.append(f"   Environmental Expansion: Applied ✓")
    has_image_analysis = any('visual_similarity' in str(conn) for conn in cross_modal_connections)
    context_parts.append(f"   Image Similarity Analysis: {'✓' if has_image_analysis else '✗'}")
    context_parts.append("")
    
    # Multimodal Results
    if results:
        context_parts.append("📊 MULTIMODAL SEARCH RESULTS:")
        for i, result in enumerate(results[:3], 1):
            caption = result['metadata']['caption']
            search_type = result.get('search_type', 'unknown')
            score = result.get('enhanced_score', result.get('score', 0))
            visual_sim = result.get('visual_similarity', 'N/A')
            
            context_parts.append(f"   {i}. {caption}")
            context_parts.append(f"      → Search Type: {search_type}")
            context_parts.append(f"      → Enhanced Score: {score:.3f}")
            if visual_sim != 'N/A':
                context_parts.append(f"      → Visual Similarity: {visual_sim:.3f}")
        context_parts.append("")
    
    # Cross-Modal Analysis with Image Similarity
    if cross_modal_connections:
        context_parts.append("🔄 CROSS-MODAL ANALYSIS (Including Image Similarity):")
        
        visual_connections = [c for c in cross_modal_connections 
                            if c.get('type') == 'visual_similarity']
        entity_connections = [c for c in cross_modal_connections 
                            if c.get('type') == 'direct_entity']
        
        if visual_connections:
            context_parts.append("   Visual Similarity Matches:")
            for conn in visual_connections[:3]:
                context_parts.append(f"     • Visual Match: {conn['caption'][:60]}...")
                context_parts.append(f"       Cross-Modal Score: {conn['cross_modal_score']:.3f}")
        
        if entity_connections:
            context_parts.append("   Entity-Based Matches:")
            for conn in entity_connections[:3]:
                context_parts.append(f"     • Entity '{conn['entity']}': {conn['caption'][:60]}...")
        
        context_parts.append("")
    
    return "\n".join(context_parts)


def format_context_for_gemini(query_text: str, pinecone_results: List[Dict], 
                             graph_context: List[Dict], 
                             image_context: List[Dict]) -> str:
    """Format comprehensive context for Gemini, including synonyms for graph context entities."""
    context_parts = []

    # Example synonym mapping for graph context augmentation
    synonym_map = {
        "castle": ["fortress", "citadel", "keep", "stronghold", "bastion"],
        "fortress": ["castle", "citadel", "keep", "stronghold"],
        "ocean": ["sea", "waters", "marine"],
        "sunset": ["dusk", "twilight", "evening glow"],
        "coastal": ["seashore", "shoreline", "waterfront", "beach", "shore", "strand"],
        "view": ["vista", "panorama", "scenery", "landscape", "terrain", "topography"],
        "architecture": ["building", "structure", "design", "style", "construction"],
        "landscape": ["scenery", "terrain", "topography", "vista", "panorama"],
        "temporal": ["time", "date", "season", "year", "day", "night"],
        "location": ["place", "site", "area", "region", "location", "position"],
        "human": ["person", "individual", "figure", "actor", "subject"],
        "organization": ["company", "institution", "group", "agency", "organization"],
        "quantity": ["number", "amount", "size", "quantity", "count"],
        "environment": ["nature", "wildlife", "ecosystem", "environment", "landscape"],
        "general": ["thing", "object", "item", "subject", "topic"],
    }

    # Visual Search Results
    if pinecone_results:
        context_parts.append("📸 VISUAL SEARCH RESULTS:")
        for i, match in enumerate(pinecone_results[:3], 1):
            score = match.get('enhanced_score', match.get('score', 0))
            caption = match['metadata']['caption']
            search_type = match.get('search_type', 'standard')
            visual_sim = match.get('visual_similarity')
            
            context_parts.append(f"  {i}. {caption} (relevance: {score:.3f})")
            if search_type != 'standard':
                context_parts.append(f"      → Search Type: {search_type}")
            if visual_sim is not None:
                context_parts.append(f"      → Visual Similarity: {visual_sim:.3f}")
        context_parts.append("")

    # Domain-aware Knowledge Graph context with synonym augmentation
    if graph_context:
        domain_chunks = {
            "temporal": [], "location": [], "human": [], "organization": [],
            "quantity": [], "environment": [], "general": []
        }
        
        for item in graph_context:
            domain = item.get("domain", "general")
            
            # Add synonyms to entity if applicable
            entity_lower = item.get("entity", "").lower()
            synonyms = synonym_map.get(entity_lower)
            if synonyms:
                item["synonyms"] = synonyms
            
            domain_chunks.setdefault(domain, []).append(item)

        for domain, items in domain_chunks.items():
            if items:
                context_parts.append(f"🟦 DOMAIN: {domain.upper()}")
                for item in items:
                    line = f"  • {item['entity']} ({item['label']})"
                    if item.get('mention_count', 0) > 1:
                        line += f" - mentioned {item['mention_count']} times"
                    if item.get('related_entities'):
                        related = [r['name'] for r in item['related_entities'][:3]]
                        if related:
                            line += f" - related to: {', '.join(related)}"
                    if item.get("synonyms"):
                        line += f" - synonyms: {', '.join(item['synonyms'])}"
                    context_parts.append(line)
                context_parts.append("")

    # Enhanced Cross-Modal Connections
    if image_context:
        context_parts.append(" ENHANCED CROSS-MODAL CONNECTIONS:")
        
        # Group by connection type
        visual_matches = [c for c in image_context if c.get('type') == 'visual_similarity']
        entity_matches = [c for c in image_context if c.get('type') == 'direct_entity']
        semantic_matches = [c for c in image_context if c.get('type') == 'semantic_match']
        
        if visual_matches:
            context_parts.append("   Visual Similarity Matches:")
            for item in visual_matches[:2]:
                context_parts.append(f"     • {item['caption'][:50]}... "
                                   f"(cross-modal score: {item.get('cross_modal_score', 0):.2f})")
        
        if entity_matches:
            context_parts.append("   Entity-Based Matches:")
            for item in entity_matches[:2]:
                context_parts.append(f"     • Entity '{item['entity']}': {item['caption'][:50]}...")
        
        if semantic_matches:
            context_parts.append("   Semantic Matches:")
            for item in semantic_matches[:2]:
                context_parts.append(f"     • {item['caption'][:50]}... "
                                   f"(similarity: {item.get('similarity', 0):.2f})")
        
        context_parts.append("")

    if not context_parts:
        context_parts.append("No specific context available from visual search or knowledge graph.")

    return "\n".join(context_parts)


def query_gemini_rag(query_text: str, 
                    pinecone_results: List[Dict], 
                    graph_context: List[Dict], 
                    image_context: List[Dict]) -> str:
    """Query Gemini with enhanced RAG context - Anti-hallucination version"""
    
    formatted_context = format_context_for_gemini(query_text, pinecone_results, graph_context, image_context)
    
    # Count available context items
    context_items_count = len(pinecone_results) + len(graph_context) + len(image_context)
    
    # If very limited context, return conservative response immediately
    if context_items_count < 2:
        return f"Limited context available for query '{query_text}'. Found {len(pinecone_results)} vector matches, {len(graph_context)} graph items, and {len(image_context)} image connections. Cannot provide detailed response without more relevant context."
    
    # Create anti-hallucination prompt
    prompt = f"""You are a precise AI assistant that provides accurate, context-based responses.

USER QUERY: "{query_text}"

AVAILABLE CONTEXT:
- Vector search results: {len(pinecone_results)} items
- Knowledge graph context: {len(graph_context)} items  
- Cross-modal connections: {len(image_context)} items

CONTEXT DETAILS:
{formatted_context}

STRICT INSTRUCTIONS:
1. **Base your response on the explicitly provided context above. If none exists, cautiously suggest plausible connections based on visual similarity.**
2. **Start with**: "Based on the available context..."
3. **For each domain present in context**, provide 1-2 factual sentences with hashtags
4. **If information is missing**, state: "No specific information available about [aspect]"
5. **Use descriptive but factual language** - avoid creative metaphors
6. **End with**: "This response is based on the {context_items_count} available context items"

CONTEXT VALIDATION:
- If context seems insufficient for a domain, acknowledge this limitation
- Do not invent details not present in the provided context
- Focus on what the context explicitly contains

RESPONSE FORMAT:
"Based on the available context: [factual statement]. #Domain
[Additional factual statements for other domains present in context]
No specific information available about [missing aspects].
This response is based on the {context_items_count} available context items."

RESPONSE:"""

    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    params = {"key": GEMINI_API_KEY}

    try:
        response = requests.post(GEMINI_URL, headers=headers, params=params, json=data)
        response.raise_for_status()
        answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        
        # Validate response doesn't contain obvious hallucination patterns
        if _contains_hallucination_indicators(answer):
            return f"Context insufficient for detailed response to '{query_text}'. Available: {len(pinecone_results)} search results, {len(graph_context)} graph items, {len(image_context)} image connections. More context needed for comprehensive answer."
        
        # Enhanced fallback with context awareness
        if not answer.strip() or "no specific context" in answer.lower():
            return f"Unable to generate detailed response for '{query_text}' with current context. Available: {len(pinecone_results)} vector matches, {len(graph_context)} knowledge graph items, {len(image_context)} cross-modal connections. Context may be insufficient or not directly relevant to the query."
        
        return answer
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

def _contains_hallucination_indicators(response: str) -> bool:
    """Quick check for obvious hallucination patterns"""
    
    hallucination_phrases = [
        "reminiscent of", "like.*poured", "honey.*heaven", 
        "celestial.*gold", "mystical", "ethereal", "as if",
        "seems to suggest", "one might imagine", "evokes"
    ]
    
    response_lower = response.lower()
    
    # Check for creative language that's likely hallucinated
    for phrase in hallucination_phrases:
        if re.search(phrase, response_lower):
            return True
    
    # Check if response starts appropriately
    if not response_lower.startswith("based on"):
        return True
    
    return False


def enhanced_rag_pipeline(query_text=None, image_path=None):
    print("🚀 Starting Enhanced RAG Pipeline with Knowledge Graph Integration...")
    
    try:
        # Step 1️⃣ Expand the query first
        expanded_query = enhanced_environmental_query_expansion(query_text) if query_text else ""
        print(f"🔍 Expanded Query: {expanded_query}")
        
        # Step 2️⃣ Use the expanded query in Pinecone search with dimension verification
        pinecone_results = enhanced_multimodal_search(
            query_text=expanded_query if expanded_query else None,
            query_image_path=image_path,
            similarity_threshold=0.65
        )
        print(f"📊 Found {len(pinecone_results)} relevant matches in Pinecone")
        
        # Apply relevance threshold filtering
        filtered_results = filter_by_relevance_threshold(pinecone_results, threshold=0.65)
        print(f"🎯 Filtered to {len(filtered_results)} high-relevance matches (≥0.65)")
        pinecone_results = filtered_results
        
        if not expanded_query and image_path:
            expanded_query = f"Image analysis: {image_path}"
        
        # Process entities and build context
        relevant_entities = []
        unique_entities = []
        query_entities = []
        
        if pinecone_results:
            print("🔄 Enhanced entity extraction and linking...")
            relevant_entities = process_entities_pipeline(expanded_query, pinecone_results)
            
            # Extract query entities
            for entry in relevant_entities:
                for entity in entry.get("entities", []):
                    query_entities.append(entity["text"].lower())
            
            print(f"🔍 Identified {len(query_entities)} unique entities")
            
            # Aggregate entities
            unique_entities = aggregate_entities(relevant_entities)
            print(f"✨ Deduplicated to {len(unique_entities)} unique entity types")
        
        # Get graph context
        graph_context = get_graph_context(graph_driver, expanded_query, nlp)
        print(f"📊 Retrieved context for {len(graph_context)} entities from knowledge graph")
        
        if not graph_context:
            print("\n⚠️ No relevant graph context found. Falling back to image context only.")
        
        # Get image context with improved cross-modal linking
        image_context = enhanced_cross_modal_linking_with_images(
            expanded_query, 
            image_path,
            pinecone_results,
            query_entities
        )
        print(f"🖼️ Found {len(image_context)} cross-modal image connections")
        
        # Generate final answer with all context
        final_answer = query_gemini_rag(
            expanded_query,
            pinecone_results,
            graph_context,
            image_context
        )
        
        return {
            "query": expanded_query,
            "pinecone_matches": len(pinecone_results),
            "entities_processed": len(relevant_entities),
            "raw_entities": len(query_entities),
            "unique_entities": unique_entities,
            "graph_context_items": len(graph_context),
            "graph_context": graph_context,
            "cross_modal_connections": len(image_context),
            "image_context": image_context,
            "pinecone_results": pinecone_results,
            "answer": final_answer
        }
        
    except Exception as e:
        print(f"⚠️ Error in RAG pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # ✅ Use the new enhanced expansion function instead of expand_coastal_castle_query
    query_text = enhanced_environmental_query_expansion("a sunset view of a coastal castle")
    result = enhanced_rag_pipeline(query_text=query_text)

    # ✅ 3️⃣ Prettify Final Summary
    print("\n" + "="*60)
    print("🤖 ENHANCED RAG PIPELINE RESULTS")
    print("="*60)
    print(f"Query: {result['query']}")
    print(f"Pinecone Matches: {result['pinecone_matches']}")
    print(f"Entities Processed: {result['entities_processed']}")
    print(f"Raw Entity Mentions: {result['raw_entities']}")
    print(f"Unique Entity Types: {len(result['unique_entities'])}")
    print(f"Graph Context Items: {result['graph_context_items']}")
    print(f"Cross-Modal Connections: {result['cross_modal_connections']}")
    if 'original_cross_modal_count' in result:
        print(f"Original Cross-Modal Count: {result['original_cross_modal_count']} → Deduplicated: {result['cross_modal_connections']}")
    
    # Show unique entities with counts
    if result['unique_entities']:
        print("\n🌟 Unique Entities Mentioned:")
        for item in result['unique_entities']:
            print(f"  - {item['entity']} (mentioned {item['count']} times)")
    
    print("\n📝 Final Answer:")
    print(result['answer'])
    
    print("\n📊 Summary Highlights:")
    print("- Integrates visual search results with knowledge graph context")
    print("- Deduplicates entity mentions for cleaner analysis")
    print("- Provides fallback handling when graph context is sparse")
    print("- Generates domain-aware responses with hashtag categorization")
    print("- Enhanced query expansion with architectural and scenic terms")
    print("- Relevance threshold filtering (≥0.65) to reduce noise")
    print("- Cross-modal deduplication for cleaner image connections")
