"""
REST API for Image Recognition and Product Matching
@File    : app.py
@Date    : 2025-03-04
@Author  : Nandini Bohra
@Contact : nbohra@ucsd.edu

@References : 
"""

from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from transformers import AutoImageProcessor, AutoModel
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os

load_dotenv()

# Defining constants
SIMILARITY_THRESHOLD = 0.7
COLLECTION_NAME = "sample_images_2"
MATCHED_WIDTH = 1700
MATCHED_HEIGHT = 920
QDRANT_DB_URL = os.getenv("QDRANT_DB_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize Flask app
app = Flask(__name__)


# Initialize connection to Qdrant
qclient = QdrantClient(
    url= os.getenv("QDRANT_DB_URL"),
    api_key= os.getenv("QDRANT_API_KEY")
)

# Set up image processor and model (DINOv2 as of 7–MAR–2025)
# Matches the model used to generate embeddings in get_img_embeddings.ipynb
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

# Generate matching embeddings for input images using matching model
# Processing reflects embeddings for catalog images to inspire consistency
def get_embeddings(image):
    input = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model(**input)
    
    # Remove CLS token embedding which is used for classification
    # Taking average of patch embeddings
    patch_embedding = output.last_hidden_state[:, 1:, :]
    avg_patch_embedding = torch.mean(patch_embedding, dim=1)

    return avg_patch_embedding


# Search for similar products in Qdrant database
# Using embeddings generated from input image
def search(input_embedding):
    search_results = qclient.query_points(
        collection_name= COLLECTION_NAME,
        query= input_embedding,
        limit= 5,
        # query_filter=models.Filter(
        #     with_payload=True,
        #     score_threshold= SIMILARITY_THRESHOLD
        # )
    ).points


    # filtered_results = [
    #     {"id": item.id, "score": item.score}
    #     for item in search_results if item.score >= SIMILARITY_THRESHOLD
    # ]

    # If no similar products found above threshold
    # Return not found message to user
    if not search_results:
        return jsonify({"message": "No similar products found"})
    
    payloads = [hit.payload for hit in search_results]
    return payloads


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
    
    # Load image from upload and convert to PIL image
    # Resize image to match size of catalog samples in DB
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    resized_image = image.resize((MATCHED_WIDTH, MATCHED_HEIGHT))

    # Generate embeddings for input image
    # Search for similar products in Qdrant database
    input_embedding = get_embeddings(resized_image).squeeze(0).tolist()
    similar_products = search(input_embedding)

    return jsonify({"similar_products": similar_products})


# Route 2: User selects product from catalog to find similar products
# Returns similar products based on selected product embeddings
@app.route('/get_similar/<int:id>', methods=['GET'])
def get_similar(id):

    catalog_img = qclient.retrieve(
        collection_name=COLLECTION_NAME, 
        ids=[id],
        with_vectors=True 
    )
    print(catalog_img)

    # Invalid product ID returns error message
    if not catalog_img:
        return jsonify({"error": "Product not found"}), 404
    
    embedding = catalog_img[0].vector
    print(embedding)
    search_results = search(embedding)
    

    # search_results = qdrant_client.search(
    #     collection_name= COLLECTION_NAME,
    #     query_vector= id,
    #     limit= 5
    # )


    # Filter out results with similarity score below threshold
    # filtered_results = [
    #     {"id": item.id, "score": item.score}
    #     for item in search_results if item.score >= SIMILARITY_THRESHOLD
    # ]

    # If no similar products found above threshold
    # Return not found message to user
    # if not filtered_results:
    #     return jsonify({"message": "No similar products found"})
    
        # Potential to suggest new products to user at this stage

    return search_results



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