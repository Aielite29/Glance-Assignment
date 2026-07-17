"""
Query → Filter → Retrieve → Rerank pipeline for Glance Fashion Retrieval.

Part B: Parses a natural-language query to extract metadata filters, performs
bi-encoder retrieval against ChromaDB using FashionSigLIP, then reranks the
shortlist with a BLIP cross-encoder for high-precision results.
"""

import logging
import warnings
import time

import chromadb
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, BlipForImageTextRetrieval

from src.config import (
    ACCESSORY_CATEGORIES,
    BLIP_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    ENVIRONMENT_KEYWORDS,
    EVAL_QUERIES,
    FILTER_MIN_RESULTS,
    GARMENT_CATEGORIES,
    RERANK_TOP_N,
    RETRIEVAL_TOP_K,
    SIGLIP_MODEL_NAME,
    TRAIN_IMAGES_DIR,
    VECTOR_DB_DIR,
    VIBE_KEYWORDS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_dtype():
    """Return float16 on CUDA for memory efficiency, float32 on CPU."""
    return torch.float16 if _DEVICE.type == "cuda" else torch.float32


# ---------------------------------------------------------------------------
# Model caching
# ---------------------------------------------------------------------------
_siglip_cache: dict = {}
_blip_cache: dict = {}


def load_siglip_model():
    """Load and cache the FashionSigLIP model + processor.

    Returns
    -------
    tuple[AutoModel, AutoProcessor]
    """
    if "model" not in _siglip_cache:
        logger.info("Loading SigLIP model: %s", SIGLIP_MODEL_NAME)
        processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME, trust_remote_code=True)
        model = model.to(_DEVICE, dtype=_get_dtype())
        model.eval()
        _siglip_cache["model"] = model
        _siglip_cache["processor"] = processor
        logger.info("SigLIP loaded on %s", _DEVICE)
    return _siglip_cache["model"], _siglip_cache["processor"]


def load_blip_model():
    """Load and cache the BLIP ITM cross-encoder model + processor.

    Returns
    -------
    tuple[BlipForImageTextRetrieval, AutoProcessor]
    """
    if "model" not in _blip_cache:
        logger.info("Loading BLIP reranker: %s", BLIP_MODEL_NAME)
        processor = AutoProcessor.from_pretrained(BLIP_MODEL_NAME)
        model = BlipForImageTextRetrieval.from_pretrained(BLIP_MODEL_NAME)
        model = model.to(_DEVICE)  # Keep BLIP in float32 — fp16 causes NaN ITM scores
        model.eval()
        _blip_cache["model"] = model
        _blip_cache["processor"] = processor
        logger.info("BLIP loaded on %s", _DEVICE)
    return _blip_cache["model"], _blip_cache["processor"]


_chroma_cache: dict = {}

def _get_collection():
    """Open the existing ChromaDB collection (cached after first call)."""
    if "collection" not in _chroma_cache:
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        _chroma_cache["collection"] = client.get_collection(
            name=CHROMA_COLLECTION_NAME,
        )
    return _chroma_cache["collection"]


# ---------------------------------------------------------------------------
# Stage 1: Query Parsing
# ---------------------------------------------------------------------------
ALL_CATEGORIES = GARMENT_CATEGORIES + ACCESSORY_CATEGORIES


def parse_query(query: str) -> dict:
    """Extract structured metadata filters from a natural-language query.

    Scans the lowercased query for known garment/accessory names, environment
    keywords, and vibe keywords.

    Parameters
    ----------
    query : str
        Free-text search query.

    Returns
    -------
    dict
        Keys: ``garment_filter`` (str | None), ``env_filter`` (str | None),
        ``vibe_filter`` (str | None).
    """
    query_lower = query.lower()
    tokens = query_lower.split()

    # --- Garment / accessory filter: exact token match first, then longest substring ---
    garment_filter = None
    for cat in ALL_CATEGORIES:  # exact token match
        if cat in tokens:
            garment_filter = cat
            break
    if garment_filter is None:  # substring fallback (longest first to avoid "shirt"→"t-shirt")
        for cat in sorted(ALL_CATEGORIES, key=len, reverse=True):
            if cat in query_lower:
                garment_filter = cat
                break

    # --- Environment filter ---
    env_filter = None
    for keyword, canonical in ENVIRONMENT_KEYWORDS.items():
        if keyword in tokens or keyword in query_lower:
            env_filter = canonical
            break

    # --- Vibe filter ---
    vibe_filter = None
    for keyword, canonical in VIBE_KEYWORDS.items():
        if keyword in tokens or keyword in query_lower:
            vibe_filter = canonical
            break

    parsed = {
        "garment_filter": garment_filter,
        "env_filter": env_filter,
        "vibe_filter": vibe_filter,
    }
    logger.info("Parsed query '%s' → %s", query, parsed)
    return parsed


# ---------------------------------------------------------------------------
# Stage 2: Bi-Encoder Retrieval
# ---------------------------------------------------------------------------
def _build_where_clause(parsed: dict) -> dict | None:
    """Construct a ChromaDB ``where`` clause from parsed filters.

    Parameters
    ----------
    parsed : dict
        Output of :func:`parse_query`.

    Returns
    -------
    dict | None
        A valid ChromaDB ``where`` dict, or ``None`` if no filters apply.
    """
    conditions = []

    if parsed["garment_filter"]:
        conditions.append({"garments": {"$contains": parsed["garment_filter"]}})
    if parsed["env_filter"]:
        conditions.append({"env": parsed["env_filter"]})
    if parsed["vibe_filter"]:
        conditions.append({"vibe": parsed["vibe_filter"]})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


@torch.no_grad()
def _encode_query(query: str, model, processor) -> list[float]:
    """Encode a single query string with SigLIP and return as a list.

    Parameters
    ----------
    query : str
        The search query.
    model : AutoModel
        FashionSigLIP model.
    processor : AutoProcessor
        FashionSigLIP processor.

    Returns
    -------
    list[float]
        L2-normalised query embedding.
    """
    dtype = _get_dtype()
    inputs = processor(
        text=[query], return_tensors="pt", padding="max_length", truncation=True
    )
    inputs = {
        k: v.to(_DEVICE, dtype=dtype) if v.is_floating_point() else v.to(_DEVICE)
        for k, v in inputs.items()
    }
    emb = model.get_text_features(**inputs)  # (1, D)
    if not isinstance(emb, torch.Tensor):
        if type(emb) is tuple and len(emb) == 1:
            emb = emb[0]
        if hasattr(emb, "pooler_output") and emb.pooler_output is not None:
            emb = emb.pooler_output
        elif hasattr(emb, "text_embeds") and emb.text_embeds is not None:
            emb = emb.text_embeds
        else:
            emb = emb[0]
    emb = emb.float()
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy().tolist()


def retrieve_candidates(
    query: str,
    siglip_model,
    siglip_processor,
    collection,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """Perform bi-encoder retrieval with optional metadata filtering.

    If the filtered result set is smaller than ``FILTER_MIN_RESULTS``, the
    query is automatically retried without filters.

    Parameters
    ----------
    query : str
        Natural-language query.
    siglip_model : AutoModel
        Cached FashionSigLIP model.
    siglip_processor : AutoProcessor
        Cached FashionSigLIP processor.
    collection : chromadb.Collection
        The ChromaDB collection to query.
    top_k : int
        Number of candidates to retrieve.

    Returns
    -------
    list[dict]
        Each dict contains ``id``, ``image_path``, ``caption``,
        ``garments``, ``colors``, ``vibe``, ``env``, ``distance``.
    """
    query_emb = _encode_query(query, siglip_model, siglip_processor)
    parsed = parse_query(query)
    where_clause = _build_where_clause(parsed)

    # --- Filtered retrieval ---
    if where_clause is not None:
        try:
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("Filtered query failed (%s); falling back to unfiltered", exc)
            results = None

        # Fallback if too few results
        if results and results["ids"] and len(results["ids"][0]) >= FILTER_MIN_RESULTS:
            logger.info(
                "Filtered retrieval returned %d candidates", len(results["ids"][0])
            )
        else:
            logger.info(
                "Filtered retrieval returned < %d results; retrying unfiltered",
                FILTER_MIN_RESULTS,
            )
            where_clause = None

    # --- Unfiltered retrieval (or fallback) ---
    if where_clause is None:
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        logger.info("Unfiltered retrieval returned %d candidates", len(results["ids"][0]))

    # --- Pack into list of dicts ---
    candidates = []
    for doc_id, meta, dist in zip(
        results["ids"][0], results["metadatas"][0], results["distances"][0]
    ):
        candidates.append(
            {
                "id": doc_id,
                "image_path": meta.get("image_path", ""),
                "caption": meta.get("caption", ""),
                "garments": meta.get("garments", "[]"),
                "colors": meta.get("colors", "[]"),
                "vibe": meta.get("vibe", ""),
                "env": meta.get("env", ""),
                "distance": float(dist),
                "bi_encoder_score": 1.0 - float(dist),  # cosine similarity
            }
        )

    return candidates


# ---------------------------------------------------------------------------
# Stage 3: BLIP Cross-Encoder Reranking
# ---------------------------------------------------------------------------
@torch.no_grad()
def rerank_with_blip(
    query: str,
    candidates: list[dict],
    blip_model,
    blip_processor,
    top_n: int = RERANK_TOP_N,
) -> list[dict]:
    """Rerank candidates using BLIP image-text matching.

    For each candidate, loads its image and computes the ITM match
    probability. Candidates are sorted by descending match score.

    Parameters
    ----------
    query : str
        The search query text.
    candidates : list[dict]
        Candidates from bi-encoder retrieval.
    blip_model : BlipForImageTextRetrieval
        Cached BLIP cross-encoder.
    blip_processor : AutoProcessor
        BLIP processor.
    top_n : int
        Number of final results to return.

    Returns
    -------
    list[dict]
        Top-N candidates with ``reranker_score`` added.
    """
    dtype = _get_dtype()
    scored = []

    for cand in candidates:
        img_path = str(TRAIN_IMAGES_DIR / cand["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.warning("Reranker: failed to load %s: %s", img_path, exc)
            cand["reranker_score"] = 0.0
            scored.append(cand)
            continue

        inputs = blip_processor(images=img, text=query, return_tensors="pt")
        inputs = {
            k: v.to(_DEVICE, dtype=dtype) if v.is_floating_point() else v.to(_DEVICE)
            for k, v in inputs.items()
        }

        outputs = blip_model(**inputs, use_itm_head=True)
        # ITM head produces logits of shape (1, 2) — [no-match, match]
        itm_scores = outputs.itm_score
        if itm_scores is None:
            logger.warning("BLIP returned None itm_score for %s; skipping", img_path)
            cand["reranker_score"] = 0.0
            scored.append(cand)
            continue
        probs = torch.softmax(itm_scores, dim=1)
        match_prob = probs[0, 1].item()

        cand["reranker_score"] = match_prob
        scored.append(cand)

    scored.sort(key=lambda x: x["reranker_score"], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# End-to-end search functions
# ---------------------------------------------------------------------------
def search(query: str, top_n: int = RERANK_TOP_N) -> list[dict]:
    """Full pipeline: encode → filter → retrieve → rerank.

    Models are loaded once and cached across calls.

    Parameters
    ----------
    query : str
        Natural-language fashion query.
    top_n : int
        Number of final results.

    Returns
    -------
    list[dict]
        Top-N results with ``image_path``, ``caption``,
        ``bi_encoder_score``, and ``reranker_score``.
    """
    siglip_model, siglip_processor = load_siglip_model()
    blip_model, blip_processor = load_blip_model()
    collection = _get_collection()

    t0 = time.time()
    candidates = retrieve_candidates(
        query, siglip_model, siglip_processor, collection
    )
    t1 = time.time()
    results = rerank_with_blip(query, candidates, blip_model, blip_processor, top_n)
    t2 = time.time()
    
    logger.info("⏱️ Latency - Retrieval: %.3fs | Reranking: %.3fs", t1 - t0, t2 - t1)
    return results


def search_without_reranker(query: str, top_n: int = RERANK_TOP_N) -> list[dict]:
    """Ablation: bi-encoder retrieval only (no BLIP reranking).

    Parameters
    ----------
    query : str
        Natural-language fashion query.
    top_n : int
        Number of results to return.

    Returns
    -------
    list[dict]
        Top-N candidates ranked by bi-encoder cosine similarity.
    """
    siglip_model, siglip_processor = load_siglip_model()
    collection = _get_collection()

    candidates = retrieve_candidates(
        query, siglip_model, siglip_processor, collection
    )
    # Already sorted by distance (ascending) from ChromaDB
    for cand in candidates:
        cand["reranker_score"] = None
    return candidates[:top_n]


def search_with_reranker_only(
    query: str,
    candidates: list[dict],
    top_n: int = RERANK_TOP_N,
) -> list[dict]:
    """Ablation: BLIP reranking on pre-supplied candidates.

    Parameters
    ----------
    query : str
        Natural-language fashion query.
    candidates : list[dict]
        Pre-retrieved candidate dicts (must include ``image_path``).
    top_n : int
        Number of results to return.

    Returns
    -------
    list[dict]
        Top-N candidates ranked by BLIP match probability.
    """
    blip_model, blip_processor = load_blip_model()
    return rerank_with_blip(query, candidates, blip_model, blip_processor, top_n)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _print_results(query: str, results: list[dict]) -> None:
    """Pretty-print search results for a single query."""
    print(f"\n{'=' * 80}")
    print(f"QUERY: {query}")
    print(f"{'=' * 80}")
    for i, r in enumerate(results, 1):
        bi_score = f"{r['bi_encoder_score']:.4f}" if r.get("bi_encoder_score") is not None else "N/A"
        re_score = f"{r['reranker_score']:.4f}" if r.get("reranker_score") is not None else "N/A"
        print(f"\n  [{i}] {r['image_path']}")
        print(f"      Caption   : {r['caption'][:120]}...")
        print(f"      Bi-Enc    : {bi_score}")
        print(f"      Reranker  : {re_score}")
    print()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Loading models (one-time) …")
    # Pre-warm caches
    load_siglip_model()
    load_blip_model()

    for q in EVAL_QUERIES:
        results = search(q, top_n=RERANK_TOP_N)
        _print_results(q, results)
