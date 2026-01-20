from flask import Flask, request, jsonify
from PIL import Image
import io

import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

# Defining constants
COLLECTION_NAME = "TextileProductRec"

QDRANT_DB_URL = os.getenv("QDRANT_DB_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize Flask app
app = Flask(__name__)


# Initialize connection to Qdrant
qclient = QdrantClient(
    url= QDRANT_DB_URL,
    api_key= QDRANT_API_KEY
)

# Set up image processor and model (DINOv2 as of 7–MAR–2025)
# Matches the model used to generate embeddings onto Qdrant
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')
model.eval()

# Functions for UPLOAD endpoint specifically
# Generate matching embeddings for input images using matching model
# Processing reflects embeddings for catalog images to inspire consistency
def load_image_for_embedding(source, max_size=224):
    img = Image.open(source).convert("RGB")
    img.thumbnail((max_size, max_size))
    return img

def get_avg_emb(image):
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    
    patches = outputs.last_hidden_state[:, 1:, :]
    # print(f"Patches shape: {patches.shape}")
    emb = torch.mean(patches, dim=1)

    # Normalize embeddings
    emb = emb / emb.norm(dim=-1, keepdim=True)
    # print(f"Embedding shape (before squeeze): {emb.shape}")
    return emb.squeeze(0).numpy()

def get_avg_rgb(image):
    return np.array(image).mean(axis=(0, 1)).astype(int)



# Rerank search results by average RGB color distance of textiles
# Embeddings prioritize searching on textile texture, so this boosts
# color similarity
def rerank_by_color(results, query_rgb, alpha=0.85):
    query_rgb = np.array(query_rgb)

    rescored = []
    for hit in results:
        r, g, b = hit.payload["r"], hit.payload["g"], hit.payload["b"]
        item_rgb = np.array([r, g, b])

        color_dist = np.linalg.norm(query_rgb - item_rgb)
        color_sim = 1 / (1 + color_dist / 255)

        final_score = alpha * hit.score + (1 - alpha) * color_sim
        rescored.append((final_score, hit))

    rescored.sort(key=lambda x: x[0], reverse=True)
    return [hit for _, hit in rescored]


# Search for similar products in Qdrant database
# Using embeddings generated from input image
def search(input_embedding):
    response = qclient.query_points(
        collection_name= COLLECTION_NAME,
        query= input_embedding,
        limit= 10,
        with_payload=True
    )

    return response.points if response.points else []




# Route for homepage
# Used in setup of API only
# General use implementation will not use homepage route
@app.route('/')
def index():

    return "Homepage"


# Route 1: User uploads non-catalog image to find similar products
# Returns similar products based on input image embeddings
@app.route('/upload', methods=['POST'])
def upload():

    if "file" not in request.files:
        response = jsonify({"message": "No file part in the request"})
        response.status_code = 400

        return response
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "Empty file uploaded"}), 400
    

    image = load_image_for_embedding(io.BytesIO(request.files['file'].read()))
    embedding = get_avg_emb(image).tolist()
    results = search(embedding)

    print("Embedding norm:", np.linalg.norm(np.array(embedding)))
    print("Top scores:", [hit.score for hit in results[:3]])

    if not results:
        response = jsonify({
            "similar_products": [], 
            "message": "No similar products found"
            })
        response.status_code = 200
        return response

    query_rgb = get_avg_rgb(image)
    results = rerank_by_color(results, query_rgb)

    return jsonify({
        "similar_products": [
            {
                "id": hit.id,
                "score": hit.score,
                "image path": hit.payload["image path"],
                "color": hit.payload["color label"],
                "material": hit.payload["material"]
            }
            for hit in results
        ]
    })


# Route 2: User selects product from catalog to find similar products
# Returns similar products based on selected product embeddings
@app.route('/get_similar/<int:id>', methods=['GET'])
def get_similar(id):

    points = qclient.retrieve(
        collection_name=COLLECTION_NAME, 
        ids=[id],
        with_vectors=True,
        with_payload=True
    )

    # Invalid product ID returns error message
    if not points:
        response = jsonify({"message": "Product not found"})
        response.status_code = 404

        return response
    
    embedding = points[0].vector
    payload = points[0].payload

    query_rgb = (payload["r"], payload["g"], payload["b"])

    results = search(embedding)
    results = [hit for hit in results if hit.id != id]

    print("Query ID:", id)
    print("Top scores:", [hit.score for hit in results[:3]])


    if not results:
        response = jsonify({
            "similar_products": [], 
            "message": "No similar products found"
            })
        response.status_code = 200
        return response
    
    results = rerank_by_color(results, query_rgb)

    return jsonify({
        "similar_products": [
            {
                "id": hit.id,
                "score": hit.score,
                "image path": hit.payload["image path"],
                "color": hit.payload["color label"],
                "material": hit.payload["material"]
            }
            for hit in results
        ]
    })



if __name__ == "__main__":
    app.run(debug=True)


# two paths: either you can
    # upload a picture and get similar products
        # require image embeddings on new image
        # query vector db for top three similar embeddings
        # return similar products
    # or you can select a product and view similar products
        # identify product in db
        # query vector db for top three similar embeddings
        # return similar products