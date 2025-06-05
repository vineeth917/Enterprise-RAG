import streamlit as st
import json
from retrieval.rag_pipeline_gemini import enhanced_rag_pipeline, get_text_embedding, get_image_embedding
from pyvis.network import Network
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import torch

def verify_embedding_dimensions(embedding, expected_dim, embedding_type="text"):
    """Verify embedding dimensions match expected dimensions."""
    actual_dim = len(embedding)
    if actual_dim != expected_dim:
        raise ValueError(f"{embedding_type.capitalize()} embedding dimension mismatch. Expected {expected_dim}, got {actual_dim}")
    return True

def render_pyvis_graph(graph_data):
    """Render interactive graph visualization using Pyvis."""
    net = Network(height="500px", width="100%", notebook=False, bgcolor="#ffffff", font_color="#000000")
    
    # Add nodes with improved styling and error handling
    for node in graph_data:
        try:
            # Ensure node has required fields
            if not isinstance(node, dict):
                continue
                
            # Get node ID with fallback
            node_id = node.get("id") or node.get("entity")
            if not node_id:
                continue
                
            # Get other node properties with fallbacks
            label = node.get("entity") or node_id
            title = f"Label: {node.get('label', 'Unknown')}\nType: {node.get('semantic_type', 'Unknown')}"
            color = "#1f77b4" if node.get("type") == "entity" else "#2ca02c"
            
            net.add_node(node_id, label=label, title=title, color=color)

            # Add edges with error handling
            for rel in node.get("related_entities", []):
                if not isinstance(rel, dict):
                    continue
                    
                rel_id = rel.get("id") or rel.get("name")
                if rel_id:
                    weight = float(rel.get("strength", 1.0))
                    net.add_edge(node_id, rel_id, width=weight)
                    
        except Exception as e:
            print(f"Warning: Error processing node {node}: {str(e)}")
            continue

    net.show("graph.html")
    with open("graph.html", "r") as f:
        html = f.read()
        components.html(html, height=550)

def display_image_results(result):
    """Display image results in a grid with captions and scores."""
    if "image_context" in result and result["image_context"]:
        st.subheader("🖼️ Visual Results")
        
        # Custom CSS for better caption styling
        st.markdown("""
        <style>
        .image-caption {
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 8px;
            border-radius: 4px;
            margin-top: -10px;
            font-size: 0.9em;
        }
        .match-type {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            margin-right: 5px;
            font-size: 0.8em;
        }
        .type-direct {
            background-color: #28a745;
            color: white;
        }
        .type-entity {
            background-color: #007bff;
            color: white;
        }
        .type-semantic {
            background-color: #6f42c1;
            color: white;
        }
        .score-badge {
            background-color: rgba(255, 255, 255, 0.9);
            color: #333;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Track displayed images to prevent duplicates
        displayed_images = set()
        filtered_results = []
        
        # Filter out duplicates while preserving order
        for img_result in result["image_context"]:
            image_path = img_result.get("image_path")
            if not image_path or image_path in displayed_images:
                continue
            displayed_images.add(image_path)
            filtered_results.append(img_result)
        
        # Create columns for displaying images
        cols = st.columns(3)
        for idx, img_result in enumerate(filtered_results):
            with cols[idx % 3]:
                try:
                    # Get match type and corresponding icon
                    match_type = img_result.get("type", "unknown")
                    type_icons = {
                        "direct": "🎯",
                        "direct_entity": "🎯",
                        "entity": "📌",
                        "semantic": "🔍",
                        "visual_concept": "👁️",
                        "unknown": "❓"
                    }
                    type_icon = type_icons.get(match_type, "❓")
                    
                    # Format match type for display
                    type_display = {
                        "direct": "Direct Match",
                        "direct_entity": "Direct Match",
                        "entity": "Entity Match",
                        "semantic": "Semantic Match",
                        "visual_concept": "Visual Concept",
                        "unknown": "Unknown Match"
                    }
                    
                    # Load and display image
                    img = Image.open(img_result["image_path"])
                    st.image(img)
                    
                    # Format caption with styling
                    caption = img_result.get("caption", "")[:100] + ("..." if len(img_result.get("caption", "")) > 100 else "")
                    
                    # Format score consistently
                    score = img_result.get("similarity", img_result.get("score", 0))
                    formatted_score = f"{score:.3f}"
                    
                    # Create styled metadata HTML
                    metadata_html = f"""
                    <div class="image-caption">
                        <div>
                            <span class="match-type type-{match_type}">{type_icon} {type_display.get(match_type, "Unknown")}</span>
                            <span class="score-badge">Score: {formatted_score}</span>
                        </div>
                        <div style="margin-top: 5px;">{caption}</div>
                    </div>
                    """
                    
                    # Display metadata with styling
                    st.markdown(metadata_html, unsafe_allow_html=True)
                    
                    # Display additional metadata if available
                    if img_result.get("matched_terms"):
                        terms = ", ".join(img_result["matched_terms"][:3])
                        st.markdown(f"<div style='font-size: 0.8em; color: #666;'>🏷️ {terms}</div>", unsafe_allow_html=True)
                        
                except Exception as e:
                    st.warning(f"Could not load image: {img_result.get('image_path', 'Unknown path')}")

# Streamlit UI with improved layout and feedback
st.set_page_config(page_title="Enterprise RAG: Query & Graph UI", layout="wide")

# Sidebar configuration
st.sidebar.title("📋 System Status")
text_dim_indicator = st.sidebar.empty()
image_dim_indicator = st.sidebar.empty()

# Main content
st.title("🌐 Enterprise RAG - Demo UI")
st.markdown("""
This system combines text and image understanding with knowledge graph integration.
- Text queries use E5-large-v2 (1024d)
- Image analysis uses CLIP (512d)
- Knowledge graph provides entity context
""")

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    query = st.text_input("🔎 Enter your natural language query:", 
                         placeholder="E.g., sunset landscape view...")
    
with col2:
    uploaded_file = st.file_uploader("🖼️ Upload an image", type=["jpg", "png", "jpeg"])

# Progress indicators
progress_placeholder = st.empty()
error_placeholder = st.empty()

if st.button("Run Query", type="primary"):
    if not query and not uploaded_file:
        error_placeholder.error("⚠️ Please enter a query or upload an image!")
    else:
        try:
            with st.spinner("🚀 Processing query..."):
                # Initialize progress bar
                progress_bar = progress_placeholder.progress(0)
                
                # Verify text embedding dimensions if query exists
                if query:
                    progress_bar.progress(20)
                    text_embedding = get_text_embedding(query)
                    verify_embedding_dimensions(text_embedding, 1024, "text")
                    text_dim_indicator.success("✅ Text embedding: 1024d")
                
                # Handle image if uploaded
                image_path = None
                if uploaded_file:
                    progress_bar.progress(40)
                    with open("temp_uploaded_image.png", "wb") as f:
                        f.write(uploaded_file.read())
                    image_path = "temp_uploaded_image.png"
                    image_embedding = get_image_embedding(image_path)
                    verify_embedding_dimensions(image_embedding, 512, "image")
                    image_dim_indicator.success("✅ Image embedding: 512d")
                
                # Run RAG pipeline
                progress_bar.progress(60)
                result = enhanced_rag_pipeline(query_text=query, image_path=image_path)
                progress_bar.progress(100)
                progress_placeholder.empty()
                
                # Display results in tabs
                tab1, tab2 = st.tabs(["📝 Answer", "📊 Analysis"])
                
                with tab1:
                    st.markdown("### Final Answer")
                    st.write(result.get("answer", "No answer generated."))
                    if "image_context" in result:
                        display_image_results(result)
                
                with tab2:
                    st.markdown("### 📊 Query Analysis")
                    
                    # Enhanced metrics visualization
                    metrics_container = st.container()
                    with metrics_container:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "📌 Vector Matches",
                                result['pinecone_matches'],
                                delta=None,
                                help="Number of relevant matches found in vector database"
                            )
                        with col2:
                            st.metric(
                                "🎯 Entities",
                                result['entities_processed'],
                                delta=result['raw_entities'] - result['entities_processed'],
                                help="Number of processed entities (delta shows filtered entities)"
                            )
                        with col3:
                            st.metric(
                                "🔗 Graph Items",
                                result['graph_context_items'],
                                delta=None,
                                help="Number of related items from knowledge graph"
                            )
                        with col4:
                            st.metric(
                                "🤝 Cross-Modal",
                                result['cross_modal_connections'],
                                delta=None,
                                help="Number of text-image connections found"
                            )
                    
                    # Entity Analysis with grouping and confidence scores
                    if result.get('unique_entities'):
                        st.markdown("### 🎯 Entity Analysis")
                        
                        # Group entities by type/category
                        entity_groups = {}
                        for entity in result['unique_entities']:
                            # Extract or infer entity type
                            entity_type = entity.get('type', 'Other')
                            if not entity_type:
                                # Infer type from entity name/properties
                                if any(word in entity['entity'].lower() for word in ['time', 'date', 'year']):
                                    entity_type = 'Temporal'
                                elif any(word in entity['entity'].lower() for word in ['place', 'city', 'country']):
                                    entity_type = 'Location'
                                else:
                                    entity_type = 'General'
                            
                            if entity_type not in entity_groups:
                                entity_groups[entity_type] = []
                            entity_groups[entity_type].append(entity)
                        
                        # Display grouped entities with expandable sections
                        for group_name, entities in entity_groups.items():
                            with st.expander(f"{group_name} Entities ({len(entities)})", expanded=True):
                                # Create columns for entity display
                                cols = st.columns(2)
                                for idx, entity in enumerate(entities):
                                    with cols[idx % 2]:
                                        # Enhanced entity card
                                        st.markdown(f"""
                                        **{entity['entity']}**
                                        - Mentions: {entity['count']}
                                        - Confidence: {entity.get('confidence', 0.0):.2f}
                                        """)
                                        
                                        # Show related entities if available
                                        if entity.get('related_entities'):
                                            related = [rel['name'] for rel in entity['related_entities'][:3]]
                                            st.markdown("*Related*: " + ", ".join(related))
                        
                        # Add confidence score distribution if available
                        if any('confidence' in entity for entity in result['unique_entities']):
                            st.markdown("### 📈 Confidence Distribution")
                            confidence_scores = [entity.get('confidence', 0) for entity in result['unique_entities']]
                            st.bar_chart(confidence_scores)
                    
                    # Display any additional analysis metrics
                    if result.get('analysis_metrics'):
                        st.markdown("### 📊 Additional Metrics")
                        st.json(result['analysis_metrics'])
                
        except ValueError as ve:
            error_placeholder.error(f"⚠️ Dimension Error: {str(ve)}")
        except Exception as e:
            error_placeholder.error(f"⚠️ Error: {str(e)}")

# Footer with system information
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📋 System Checklist
✅ Query Processing  
✅ Pinecone Integration  
✅ Entity Extraction  
✅ Graph Integration  
✅ Cross-modal Linking  
✅ Visual Analysis  
✅ Interactive Graph  
✅ Anti-Hallucination
""")
