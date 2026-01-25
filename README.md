# Vision-Based Textile Recommendation System

This project was developed as part of an internship assignment to convert a small, static textile catalog into a searchable, image-based recommendation system. Given a limited dataset of textile samples and swatches, the goal was to explore representation learning strategies for fine-grained visual similarity, develop a searchable vector database, and deploy a usable retrieval API.


## Project Overview

The pipeline converts textile images into vector embeddings, stores them in a vector database, and exposes a Flask-based API for image-to-image similarity search. The system supports both:
- **Catalog-to-catalog search** (finding similar textiles within the database) with the ```/get_similar/<int:id>``` endpoint
- **Query-by-image search** using externally uploaded samples or swatches with the ```/upload``` endpoint

Due to the limited dataset size, emphasis was placed on **representation quality, pooling strategy, and qualitative retrieval behavior**, rather than large-scale training.


## Representation & Modeling Experiments

Multiple embedding approaches were explored to capture fine-grained textile characteristics:

- **Vision models**: DINOv2, CLIP, ResNet-50
- **Embedding strategies**:
  - CLS token vs. patch embeddings
  - Different intermediate layer depths
  - Global pooling vs. average pooling
- **Multimodal experiments**:
  - Incorporating textual attributes (material, color, texture, weave) alongside image embeddings

### Final Embedding Choice
The final system uses **average pooling over patch embeddings** to better preserve fine-grained texture details. This bakes in a richer and finer understanding of the material texture into the final embeddings.  

However, this introduced an overemphasis on material similarity alone. To address this, a **post-retrieval color re-ranking step** was added using average RGB values of top candidates (a valid heuristic for textile color consistency).


## Retrieval System

- **Vector database**: Qdrant
- **Similarity metric**: Cosine similarity
- **Indexing**: Image embeddings stored with associated metadata payloads
- **Re-ranking**: Visual similarity → color-based refinement


## Containerization

The entire application is containerized using **Docker** (first-time use!), enabling:
- Reproducible setup across environments
- Clear separation of dependencies
- Easier deployment and future scaling


## Next Steps

- Build and containerize a lightweight frontend for interactive querying
- Combine services using **Docker Compose** (API + vector DB + frontend)
- Convert the API into a synchronous production-ready service
- Explore improvements with larger and more diverse catalogs

---

## Tech Stack

`DINOv2 · CLIP · ResNet-50 · Qdrant · Flask · Docker · PyTorch`
