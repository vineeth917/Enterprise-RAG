from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone
import torch
import os
import json
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# Load env
load_dotenv()

# Initialize separate Pinecone clients
pc_text = Pinecone(api_key=os.environ["PINECONE_API_KEY_TEXT"])
pc_image = Pinecone(api_key=os.environ["PINECONE_API_KEY_IMAGE"])

# Index references
index_text = pc_text.Index("cc3m-new1")  # 1024-dimensional for e5-large-v2 text
index_image = pc_image.Index("cc3m-crossmodal")  # 512-dimensional for CLIP image

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer("intfloat/e5-large-v2")  # 1024-dimensional

# Load data
with open("data/processed/cc3m_data.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

batch_size = 16

for i in tqdm(range(0, len(data), batch_size)):
    batch = data[i:i+batch_size]
    image_inputs = []
    text_inputs = []

    for item in batch:
        img = Image.open(item["image_path"]).convert("RGB")
        image_inputs.append(img)
        text_inputs.append(item["caption"])

    # CLIP image embeddings
    image_inputs_enc = clip_processor(images=image_inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_embeds = clip_model.get_image_features(**image_inputs_enc)
    image_embeds /= image_embeds.norm(p=2, dim=-1, keepdim=True)

    # e5 text embeddings
    text_embeds = text_model.encode(text_inputs, convert_to_tensor=True, normalize_embeddings=True)

    # Upsert CLIP image embeddings (512)
    vectors_img = []
    for j, item in enumerate(batch):
        vectors_img.append({
            "id": f"{item['id']}-img",
            "values": image_embeds[j].tolist(),
            "metadata": {
                "type": "image",
                "caption": item["caption"],
                "image_path": item["image_path"]
            }
        })

    # Upsert e5 text embeddings (1024)
    vectors_txt = []
    for j, item in enumerate(batch):
        vectors_txt.append({
            "id": f"{item['id']}-txt",
            "values": text_embeds[j].tolist(),
            "metadata": {
                "type": "caption",
                "caption": item["caption"],
                "image_path": item["image_path"]
            }
        })

    # Upsert separately to their respective indexes
    index_image.upsert(vectors=vectors_img)
    index_text.upsert(vectors=vectors_txt)

print("✅ Done upserting both image and text embeddings separately!")