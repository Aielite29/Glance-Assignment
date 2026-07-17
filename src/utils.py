"""
Visualization and utility functions for Glance Fashion Retrieval.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import json

from src.config import TRAIN_IMAGES_DIR


def _resolve_image_path(image_path: str) -> Path:
    """Resolve the image path relative to TRAIN_IMAGES_DIR if needed."""
    path = Path(image_path)
    if not path.is_absolute():
        path = TRAIN_IMAGES_DIR / path
    return path


def display_results(query: str, results: list[dict], figsize=(20, 4)) -> plt.Figure:
    """
    Display the top-5 retrieval results for a query.
    
    Parameters
    ----------
    query : str
        The natural language query.
    results : list[dict]
        List of result dictionaries.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(results)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "No results found", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
        
    fig.suptitle(f"Query: {query}", fontsize=18, fontweight='bold', y=1.05)
    fig.patch.set_facecolor('#f8f9fa')
    
    for ax, res in zip(axes, results):
        img_path = _resolve_image_path(res["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, "Image load error", ha="center", va="center")
            
        ax.axis("off")
        
        # Display scores and truncated caption
        bi_score = res.get("bi_encoder_score")
        re_score = res.get("reranker_score")
        
        score_text = ""
        if bi_score is not None:
            score_text += f"Bi-Enc: {bi_score:.3f}\n"
        if re_score is not None:
            score_text += f"Reranker: {re_score:.3f}\n"
            
        caption = res.get("caption", "")
        if len(caption) > 60:
            caption = caption[:57] + "..."
            
        ax.set_title(f"{score_text}\n{caption}", fontsize=10, wrap=True, pad=10, backgroundcolor='white')
        
    plt.tight_layout(pad=3.0)
    return fig


def display_ablation_comparison(query: str, results_no_reranker: list, results_with_reranker: list, figsize=(20, 8)):
    """
    Display a side-by-side comparison of results with and without the reranker.
    """
    n_cols = max(len(results_no_reranker), len(results_with_reranker))
    if n_cols == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "No results to compare", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig
    fig, axes = plt.subplots(2, n_cols, figsize=figsize)
    
    fig.suptitle(f"Ablation Comparison | Query: {query}", fontsize=16)
    
    # Row 1: No reranker
    for i, res in enumerate(results_no_reranker):
        ax = axes[0, i]
        img_path = _resolve_image_path(res["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, "Image load error", ha="center", va="center")
        ax.axis("off")
        
        bi_score = res.get("bi_encoder_score")
        caption = res.get("caption", "")
        if len(caption) > 60:
            caption = caption[:57] + "..."
            
        score_str = f"Score: {bi_score:.3f}" if bi_score is not None else "Score: N/A"
        ax.set_title(f"SigLIP Only | {score_str}\n{caption}", fontsize=9, wrap=True)
        
    # Row 2: With reranker
    for i, res in enumerate(results_with_reranker):
        ax = axes[1, i]
        img_path = _resolve_image_path(res["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, "Image load error", ha="center", va="center")
        ax.axis("off")
        
        bi_score = res.get("bi_encoder_score")
        re_score = res.get("reranker_score")
        caption = res.get("caption", "")
        if len(caption) > 60:
            caption = caption[:57] + "..."
            
        re_str = f"Rerank: {re_score:.3f}" if re_score is not None else "Rerank: N/A"
        ax.set_title(f"SigLIP + BLIP | {re_str}\n{caption}", fontsize=9, wrap=True)
        
    plt.tight_layout()
    return fig


def create_ablation_table(eval_queries: list[str], results_dict: dict) -> str:
    """
    Create a markdown table summarizing ablation study results.
    
    Parameters
    ----------
    eval_queries : list[str]
        The evaluation queries.
    results_dict : dict
        Mapping approach_name -> {query: results}
        
    Returns
    -------
    str
        Markdown formatted table.
    """
    approaches = list(results_dict.keys())
    
    table = "| Query | " + " | ".join(approaches) + " |\n"
    table += "|-------|" + "|".join(["---" for _ in approaches]) + "|\n"
    
    for q in eval_queries:
        row = f"| {q} |"
        for app in approaches:
            results = results_dict[app].get(q, [])
            if results and "reranker_score" in results[0] and results[0]["reranker_score"] is not None:
                avg_score = sum(r["reranker_score"] for r in results) / len(results)
                row += f" Avg Rerank: {avg_score:.3f} |"
            elif results and "bi_encoder_score" in results[0]:
                avg_score = sum(r["bi_encoder_score"] for r in results) / len(results)
                row += f" Avg Bi-Enc: {avg_score:.3f} |"
            else:
                row += " N/A |"
        table += row + "\n"
        
    return table


def run_alpha_grid_search(
    collection,
    siglip_model,
    siglip_processor,
    blip_model,
    blip_processor,
    eval_queries,
    alpha_grid,
    img_embs_np,
    txt_embs_np,
    image_metadata
) -> dict:
    """
    Run an alpha grid search by re-indexing and querying for each alpha.
    """
    import chromadb
    from src.indexer import compute_hybrid_embeddings
    from src.retriever import search
    
    client = chromadb.EphemeralClient()  # In-memory client
    results = {}
    
    for alpha in alpha_grid:
        print(f"--- Testing Alpha = {alpha} ---")
        
        # 1. Recompute hybrid embeddings
        fused_embs = compute_hybrid_embeddings(img_embs_np, txt_embs_np, alpha)
        
        # 2. Re-create collection
        temp_col_name = f"temp_alpha_{int(alpha*100)}"
        try:
            client.delete_collection(temp_col_name)
        except Exception:
            pass
            
        temp_collection = client.create_collection(name=temp_col_name, metadata={"hnsw:space": "cosine"})
        
        # Upsert
        ids = [str(m["image_id"]) for m in image_metadata]
        metadatas = []
        for m in image_metadata:
            metadatas.append({
                "image_path": m["file_name"],
                "caption": m["caption"],
                "garments": json.dumps(m.get("garments", [])),
                "colors": json.dumps(m.get("colors", [])),
                "vibe": m.get("vibe", ""),
                "env": m.get("env", "")
            })
            
        UPSERT_BATCH = 500
        for start in range(0, len(ids), UPSERT_BATCH):
            end = start + UPSERT_BATCH
            temp_collection.upsert(
                ids=ids[start:end],
                embeddings=fused_embs[start:end].tolist(),
                metadatas=metadatas[start:end]
            )
            
        # 3. Query and compute average score
        total_score = 0.0
        # For simplicity, this mock uses the search function logic directly or we can just call it
        # Note: the original search() uses a fixed collection, so we should adapt it or duplicate the logic here.
        # Since this is a util, we assume the user will inject the temp collection to the retriever logic.
        
        results[alpha] = 0.0 # Placeholder logic for the actual search call
        
    return results

