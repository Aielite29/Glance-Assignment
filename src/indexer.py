"""
Feature extraction and vector storage for the Glance Fashion Retrieval pipeline.

Part A: Loads images and captions, encodes them with Marqo FashionSigLIP,
computes hybrid (image + text) fusion embeddings, and stores them in a
ChromaDB persistent collection with rich metadata for downstream filtering.
"""

import json
import logging
import warnings

import chromadb
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from src.config import (
    CAPTIONS_CSV_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    FUSION_ALPHA,
    SIGLIP_MODEL_NAME,
    TEXT_BATCH_SIZE,
    TRAIN_IMAGES_DIR,
    VECTOR_DB_DIR,
    VISION_BATCH_SIZE,
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


def _to_tensor(output, kind: str) -> torch.Tensor:
    """Robustly extract a plain embedding tensor from a model output.

    Marqo's custom FashionSigLIP ``get_image_features``/``get_text_features``
    return a plain tensor *only* when called with ``normalize=True``
    (see the official model card usage). If they're ever called without it,
    or a differently-wrapped model returns a HF output object instead, this
    unwraps it instead of blowing up on ``.float()``.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return _to_tensor(output[0], kind)
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        val = getattr(output, attr, None)
        if val is not None:
            return val
    last_hidden = getattr(output, "last_hidden_state", None)
    if last_hidden is not None:
        logger.warning(
            "%s: no pooled output found on %s, mean-pooling last_hidden_state",
            kind,
            type(output).__name__,
        )
        return last_hidden.mean(dim=1)
    raise TypeError(
        f"{kind}: cannot extract an embedding tensor from output of type "
        f"{type(output).__name__}"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """Load the Marqo FashionSigLIP model and processor.

    Returns
    -------
    tuple[AutoModel, AutoProcessor]
        The SigLIP vision-language model and its processor, both moved to
        the appropriate device and dtype.
    """
    logger.info("Loading SigLIP model: %s", SIGLIP_MODEL_NAME)
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME, trust_remote_code=True)
    # low_cpu_mem_usage=False: Marqo's custom wrapper builds its submodel via
    # open_clip.create_model() inside __init__, which isn't compatible with
    # transformers' newer meta-device fast-init path (causes "Cannot copy
    # out of meta tensor; no data!"). Forcing eager loading avoids that.
    model = AutoModel.from_pretrained(
        SIGLIP_MODEL_NAME,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model = model.to(_DEVICE, dtype=_get_dtype())
    model.eval()
    logger.info("Model loaded on %s (%s)", _DEVICE, _get_dtype())
    return model, processor


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def _probe_image_embedding_dim(model, processor) -> int:
    """Determine the model's output embedding width by running one dummy
    image through it.

    Can't read this off ``model.config.projection_dim`` — Marqo's custom
    ``MarqoFashionSigLIPConfig`` (loaded via trust_remote_code) only stores
    ``auto_map``, ``open_clip_model_name`` and ``model_type`` in its
    config.json; there's no dimension field anywhere in it. The real
    embedding width only exists inside the open_clip submodel built at
    load time, so probing the model directly is the one approach
    guaranteed to match whatever get_image_features() actually returns.
    """
    dtype = _get_dtype()
    dummy = Image.new("RGB", (224, 224), color=(127, 127, 127))
    inputs = processor(images=[dummy], return_tensors="pt")
    inputs = {
        k: v.to(_DEVICE, dtype=dtype) if v.is_floating_point() else v.to(_DEVICE)
        for k, v in inputs.items()
    }
    embs = model.get_image_features(inputs["pixel_values"], normalize=True)
    return _to_tensor(embs, "image").shape[-1]


@torch.no_grad()
def extract_image_embeddings(
    model,
    processor,
    image_paths: list[str],
    batch_size: int = VISION_BATCH_SIZE,
) -> np.ndarray:
    """Encode a list of images into L2-normalised embeddings.

    Parameters
    ----------
    model : AutoModel
        The SigLIP model.
    processor : AutoProcessor
        The corresponding processor.
    image_paths : list[str]
        Absolute or relative paths to the image files.
    batch_size : int
        Number of images per forward pass.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, EMBEDDING_DIM)`` with L2-normalised vectors.
        Rows corresponding to images that failed to load are zero-vectors.
    """
    all_embeddings = []
    dtype = _get_dtype()
    embed_dim = _probe_image_embedding_dim(model, processor)

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Image embeddings"):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        valid_indices = []

        for i, path in enumerate(batch_paths):
            try:
                with Image.open(path) as raw:
                    images.append(raw.convert("RGB"))
                valid_indices.append(i)
            except Exception as exc:
                logger.warning("Failed to load image %s: %s", path, exc)

        if not images:
            # All images in this batch failed — fill with zeros.
            # (embed_dim is probed once above the loop; model.config has no
            # projection_dim on this custom config class.)
            zeros = np.zeros((len(batch_paths), embed_dim), dtype=np.float32)
            all_embeddings.append(zeros)
            continue

        inputs = processor(images=images, return_tensors="pt")
        inputs = {
            k: v.to(_DEVICE, dtype=dtype) if v.is_floating_point() else v.to(_DEVICE)
            for k, v in inputs.items()
        }

        # Marqo's custom get_image_features requires the positional
        # pixel_values arg + normalize=True to return a plain tensor
        # (per the official Marqo/marqo-fashionSigLIP model card). Calling
        # it as get_image_features(**inputs) without normalize=True returns
        # a raw BaseModelOutputWithPooling instead, which broke this before.
        embs = model.get_image_features(inputs["pixel_values"], normalize=True)
        embs = _to_tensor(embs, "image")
        embs = embs.float()  # back to fp32 for normalisation
        embs = embs / embs.norm(dim=-1, keepdim=True)
        embs_np = embs.cpu().numpy()

        # Place embeddings back into a full-batch-sized array
        batch_result = np.zeros(
            (len(batch_paths), embs_np.shape[1]), dtype=np.float32
        )
        for idx, valid_idx in enumerate(valid_indices):
            batch_result[valid_idx] = embs_np[idx]

        all_embeddings.append(batch_result)

    return np.concatenate(all_embeddings, axis=0)


@torch.no_grad()
def extract_text_embeddings(
    model,
    processor,
    texts: list[str],
    batch_size: int = TEXT_BATCH_SIZE,
) -> np.ndarray:
    """Encode a list of text strings into L2-normalised embeddings.

    Parameters
    ----------
    model : AutoModel
        The SigLIP model.
    processor : AutoProcessor
        The corresponding processor.
    texts : list[str]
        Caption strings to encode.
    batch_size : int
        Number of texts per forward pass.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, EMBEDDING_DIM)`` with L2-normalised vectors.
    """
    all_embeddings = []
    dtype = _get_dtype()

    for start in tqdm(range(0, len(texts), batch_size), desc="Text embeddings"):
        batch_texts = texts[start : start + batch_size]
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inputs = {
            k: v.to(_DEVICE, dtype=dtype) if v.is_floating_point() else v.to(_DEVICE)
            for k, v in inputs.items()
        }

        # Same fix as extract_image_embeddings: positional input_ids +
        # normalize=True per Marqo's documented usage.
        embs = model.get_text_features(inputs["input_ids"], normalize=True)
        embs = _to_tensor(embs, "text")
        embs = embs.float()
        embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embeddings.append(embs.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Hybrid fusion
# ---------------------------------------------------------------------------
def compute_hybrid_embeddings(
    img_embs: np.ndarray,
    txt_embs: np.ndarray,
    alpha: float = FUSION_ALPHA,
) -> np.ndarray:
    """Compute hybrid fusion and re-normalise.

    ``v = alpha * v_img + (1 - alpha) * v_text``

    Parameters
    ----------
    img_embs : np.ndarray
        Image embeddings, shape ``(N, D)``.
    txt_embs : np.ndarray
        Text embeddings, shape ``(N, D)``.
    alpha : float
        Fusion weight for image embeddings.

    Returns
    -------
    np.ndarray
        Fused, L2-normalised embeddings of shape ``(N, D)``.
    """
    fused = alpha * img_embs + (1 - alpha) * txt_embs
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    # Avoid division by zero for any all-zero rows
    norms = np.where(norms == 0, 1, norms)
    return fused / norms


# ---------------------------------------------------------------------------
# ChromaDB indexing
# ---------------------------------------------------------------------------
def _init_chromadb():
    """Create (or open) a persistent ChromaDB client and collection.

    Returns
    -------
    tuple[chromadb.PersistentClient, chromadb.Collection]
    """
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": CHROMA_DISTANCE_METRIC},
    )
    logger.info(
        "ChromaDB collection '%s' ready (%d existing items)",
        CHROMA_COLLECTION_NAME,
        collection.count(),
    )
    return client, collection


def build_index(alpha: float = FUSION_ALPHA) -> None:
    """Run the full indexing pipeline.

    1. Load ``captions.csv``.
    2. Extract image embeddings.
    3. Extract text embeddings.
    4. Compute hybrid fusion.
    5. Upsert into ChromaDB with metadata.

    Parameters
    ----------
    alpha : float
        Fusion weight for image embeddings (default from config).
    """
    # ------------------------------------------------------------------
    # 1. Load captions data
    # ------------------------------------------------------------------
    logger.info("Loading captions from %s", CAPTIONS_CSV_PATH)
    df = pd.read_csv(CAPTIONS_CSV_PATH)
    logger.info("Loaded %d records", len(df))

    image_paths = [str(TRAIN_IMAGES_DIR / fn) for fn in df["file_name"]]
    captions = df["caption"].tolist()

    # ------------------------------------------------------------------
    # 2–3. Extract embeddings
    # ------------------------------------------------------------------
    model, processor = load_model()

    logger.info("Extracting image embeddings …")
    img_embs = extract_image_embeddings(model, processor, image_paths)

    logger.info("Extracting text embeddings …")
    txt_embs = extract_text_embeddings(model, processor, captions)

    # ------------------------------------------------------------------
    # 4. Hybrid fusion
    # ------------------------------------------------------------------
    assert img_embs.shape == txt_embs.shape, (
        f"Shape mismatch: img_embs {img_embs.shape} vs txt_embs {txt_embs.shape}"
    )
    assert img_embs.shape[0] == len(df), (
        f"Embedding count {img_embs.shape[0]} != DataFrame rows {len(df)}"
    )

    logger.info("Computing hybrid embeddings (alpha=%.2f) …", alpha)
    hybrid_embs = compute_hybrid_embeddings(img_embs, txt_embs, alpha)

    # ------------------------------------------------------------------
    # Free GPU memory before ChromaDB operations
    # ------------------------------------------------------------------
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Upsert into ChromaDB
    # ------------------------------------------------------------------
    _, collection = _init_chromadb()

    # Prepare metadata rows
    ids = []
    metadatas = []
    embeddings_list = []

    for pos, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing metadata")):
        # Skip rows whose image embedding is all zeros (failed to load).
        # Check img_embs (the raw image embedding), not hybrid_embs (the
        # fused image+text vector) — a failed image load only produces an
        # all-zero *hybrid* row when alpha == 1.0. Otherwise it silently
        # passes through here as a (1-alpha)-scaled, text-only row.
        if not np.any(img_embs[pos]):  # strictly all-zero
            logger.warning(
                "Skipping image_id %s — zero embedding", row["image_id"]
            )
            continue

        doc_id = str(row["image_id"])
        ids.append(doc_id)
        embeddings_list.append(hybrid_embs[pos].tolist())

        # Ensure garments/colors are stored as JSON strings
        garments_raw = row.get("garments", "[]")
        colors_raw = row.get("colors", "[]")
        # If they're already JSON strings, keep them; otherwise serialise
        if isinstance(garments_raw, str):
            garments_str = garments_raw
        else:
            garments_str = json.dumps(garments_raw)
        if isinstance(colors_raw, str):
            colors_str = colors_raw
        else:
            colors_str = json.dumps(colors_raw)

        metadatas.append(
            {
                "image_path": row["file_name"],
                "caption": str(row["caption"]),
                "garments": garments_str,
                "colors": colors_str,
                "vibe": str(row.get("vibe", "")),
                "env": str(row.get("env", "")),
            }
        )

    # ChromaDB supports batch upsert — chunk to avoid oversized payloads
    UPSERT_BATCH = 500
    for start in tqdm(
        range(0, len(ids), UPSERT_BATCH), desc="Upserting to ChromaDB"
    ):
        end = start + UPSERT_BATCH
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings_list[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info(
        "Indexing complete — %d vectors stored in collection '%s'",
        collection.count(),
        CHROMA_COLLECTION_NAME,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    build_index()