"""
REST API for Image Recognition and Product Matching
@File    : app.py
@Date    : 2025-03-04
@Author  : Nandini Bohra
@Contact : nbohra@ucsd.edu

"""

from flask import Flask, request, jsonify
# from catalog import search 
# from embeddings import get_embeddings

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello World"

# @app.route('/search', methods=['POST'])
# def search():

#     if request.method == 'POST':
#         embeddings = get_embeddings(request.files['image'])
#         similar_products = search(embeddings)

#     return jsonify({"similar_products": similar_products})



if __name__ == "__main__":
    app.run(debug=True)