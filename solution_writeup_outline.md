# Glance ML Internship — Solution Writeup

## 1. Approaches & Tradeoffs (2 pages)
Discuss the methods considered and why the chosen architecture is superior.

- **Baseline CLIP / Vanilla ViT**:
  - *Pros*: Simple, zero-shot, readily available.
  - *Cons*: Not tuned for fashion imagery, struggles with compositionality ("red tie" vs "white shirt").
- **FashionCLIP**:
  - *Pros*: Tuned specifically on fashion datasets.
  - *Cons*: Still suffers from bag-of-words limitations common to bi-encoders, failing on complex stylistic queries.
- **VLM (LLaVA / Qwen-VL)**:
  - *Pros*: Excellent visual reasoning and capability to answer complex compositional queries.
  - *Cons*: Computationally expensive (slow inference), hard to scale to millions of images.
- **Chosen Architecture (Bi-Encoder + Cross-Encoder with Hybrid Metadata)**:
  - *Pros*: Best of both worlds. The bi-encoder (Marqo-FashionSigLIP) combined with a metadata `WHERE` filter ensures fast retrieval with high recall. The cross-encoder (BLIP-ITM) performs heavy compositional reasoning only on the top-50 candidates, keeping latency low while drastically improving precision. Synthetic captions (Gemini) bridge the gap between structured ontology and natural language.

## 2. Chosen Architecture (1 page)
Explain the pipeline implemented.

- **Data Pipeline**: Explain how 3,000 images were subsampled, prioritizing natural backgrounds (`isstatic=0`), and how Gemini 2.0 Flash was used with 5 few-shot examples to convert COCO-style tags into natural captions + structured metadata (garments, colors, vibe, env).
- **Hybrid Fusion Embeddings**: $v = \alpha \cdot v_{img} + (1-\alpha) \cdot v_{caption}$. Explain that this fuses visual evidence with semantic understanding.
- **Retrieval Pipeline**:
  - **Query Parsing**: Extracting garment/env/vibe keywords to build a ChromaDB `WHERE` clause.
  - **Stage 1 (Bi-Encoder)**: Fast vector search using the fused embeddings.
  - **Stage 2 (Cross-Encoder)**: BLIP ITM scores each of the top-50 candidates against the raw query.

## 3. Results & Evaluation (1 page)
Present the results of the 5 evaluation queries.

- Include the visual Top-5 grid for each query (using `display_results` from `utils.py`).
- Include the **Ablation Study Table**:
  - Compare "FashionSigLIP Only" vs "Full Pipeline (SigLIP + BLIP)".
  - Show how precision improves on the compositional query ("A red tie and a white shirt...").
- Include the **Alpha Grid Search Table**:
  - Show how tuning $\alpha$ between 0.3 and 0.8 affects the mean average precision / mean BLIP score across the queries, justifying the chosen $\alpha = 0.6$.

## 4. Future Work (1 page)
How to scale and improve this system for production.

- **Entity Extraction**: Use a lightweight NLP model (e.g., SpaCy) to extract weather and geographic locations from queries, mapped to a predefined schema for expanded `WHERE` filtering.
- **Late Interaction (ColBERT)**: Replace the cross-encoder with a ColBERT-style late interaction model for faster reranking without sacrificing precision.
- **Hard Negative Mining**: Fine-tune the FashionSigLIP bi-encoder on hard negatives (e.g., images with a white tie and red shirt) to push the model to learn compositionality natively.
- **Scalability**: Migrate from ChromaDB to Qdrant or Milvus for horizontal scaling to 1M+ images.
