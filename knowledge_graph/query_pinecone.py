from dotenv import load_dotenv
import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# Load environment
load_dotenv()

# Initialize Pinecone for text index
pc_text = Pinecone(api_key=os.environ["PINECONE_API_KEY_TEXT"])
index_text = pc_text.Index("cc3m-new1")  # text embeddings index

# Initialize Pinecone for image index
pc_image = Pinecone(api_key=os.environ["PINECONE_API_KEY_IMAGE"])
index_image = pc_image.Index("cc3m-crossmodal")  # image embeddings index

# Initialize models
text_model = SentenceTransformer("intfloat/e5-large-v2")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embedding(query_text):
    emb = text_model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
    return emb[0].tolist() if emb.ndim > 1 else emb.tolist()

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=[image], return_tensors="pt", padding=True)
    with torch.no_grad():
        embeds = clip_model.get_image_features(**inputs)
        embeds /= embeds.norm(p=2, dim=-1, keepdim=True)
    return embeds[0].tolist()

def search_pinecone(index, query_embedding, top_k=5):
    return index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

def save_results(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results to {output_file}")

if __name__ == "__main__":
    batch_queries = [
        "A castle by the beach at sunset",
        "Man walking on a rainy day",
        "People at a fancy dinner",
        "The view of mountains and sunset",
        "Skateboarder doing tricks in a park",
        "A man walks beside a woman on a rainy day"
    ]

    results_text = []
    for query in tqdm(batch_queries, desc="Running text queries"):
        text_embedding = get_text_embedding(query)
        pinecone_results = search_pinecone(index_text, text_embedding, top_k=5)
        matches = [{
            "score": m["score"],
            "type": m["metadata"]["type"],
            "caption": m["metadata"]["caption"],
            "image_path": m["metadata"]["image_path"]
        } for m in pinecone_results["matches"]]
        results_text.append({
            "query": query,
            "results": matches
        })

    image_query_path = "data/cc3m/validation/4087_94184201.jpg"
    image_embedding = get_image_embedding(image_query_path)
    pinecone_results_image = search_pinecone(index_image, image_embedding, top_k=5)
    results_image = [{
        "score": m["score"],
        "type": m["metadata"]["type"],
        "caption": m["metadata"]["caption"],
        "image_path": m["metadata"]["image_path"]
    } for m in pinecone_results_image["matches"]]

    # Save both result sets
    save_results({"text_queries": results_text}, "retrieval/results_text.json")
    save_results({"image_query": image_query_path, "results": results_image}, "retrieval/results_image.json")
