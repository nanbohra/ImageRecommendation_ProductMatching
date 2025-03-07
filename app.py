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
import qdrant_client

# Defining constants
SIMILARITY_THRESHOLD = 0.7
COLLECTION_NAME = "sample_images_2"
MATCHED_WIDTH = 1700
MATCHED_HEIGHT = 920

# Initialize Flask app
app = Flask(__name__)


# Initialize connection to Qdrant
qdrant_client = qdrant_client.QdrantClient("http://localhost:6333")

# Set up image processor and model (DINOv2 as of 07/03/2025)
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
    search_results = qdrant_client.search(
        collection_name= COLLECTION_NAME,
        query_vector= input_embedding,
        limit= 10
    )

    # Filter out results with similarity score below threshold
    filtered_results = [
        {"id": item.id, "score": item.score}
        for item in search_results if item.score >= SIMILARITY_THRESHOLD
    ]

    # If no similar products found above threshold
    # Return not found message to user
    if not filtered_results:
        return jsonify({"message": "No similar products found"})

    return filtered_results


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

    if "files" not in request.files:
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
    input_embedding = get_embeddings(resized_image)
    similar_products = search(input_embedding)

    return jsonify({"similar_products": similar_products})


# Route 2: User selects product from catalog to find similar products
# Returns similar products based on selected product embeddings
@app.route('/get_similar/int:id', methods=['GET'])
def get_similar(id):

    search_results = qdrant_client.search(
        collection_name= COLLECTION_NAME,
        query_vector= id,
        limit= 5
    )

    # Invalid product ID returns error message
    if not search_results:
        return jsonify({"error": "Product not found"}), 404

    # Filter out results with similarity score below threshold
    filtered_results = [
        {"id": item.id, "score": item.score}
        for item in search_results if item.score >= SIMILARITY_THRESHOLD
    ]

    # If no similar products found above threshold
    # Return not found message to user
    if not filtered_results:
        return jsonify({"message": "No similar products found"})
    
        # Potential to suggest new products to user at this stage

    return filtered_results



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