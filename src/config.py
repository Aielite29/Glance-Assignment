"""
Central configuration for the Glance Fashion Retrieval pipeline.

All paths, model names, hyperparameters, and constants are defined here.
Adjust FASHIONPEDIA_ROOT and PROJECT_ROOT for your environment (local vs Kaggle).
"""

import os
from pathlib import Path

# =============================================================================
# Environment Detection
# =============================================================================
IS_KAGGLE = os.path.exists("/kaggle/input")

# =============================================================================
# Path Configuration
# =============================================================================
if IS_KAGGLE:
    # Kaggle paths - automatically find the dataset folder wherever it is mounted
    import glob
    _json_paths = glob.glob("/kaggle/input/**/instances_attributes_train2020.json", recursive=True)
    if _json_paths:
        FASHIONPEDIA_ROOT = Path(_json_paths[0]).parent
    else:
        FASHIONPEDIA_ROOT = Path("/kaggle/input/fashionpedia-dataset")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # works for any copy folder name
else:
    # Local development paths
    FASHIONPEDIA_ROOT = Path(r"C:\Users\Abhinav\Glance-project")
    PROJECT_ROOT = Path(r"C:\Users\Abhinav\Glance-project\glance-fashion-retrieval")

# Fashionpedia data paths
TRAIN_IMAGES_DIR = FASHIONPEDIA_ROOT / "train"
TEST_IMAGES_DIR = FASHIONPEDIA_ROOT / "test"
TRAIN_JSON_PATH = FASHIONPEDIA_ROOT / "instances_attributes_train2020.json"
VAL_JSON_PATH = FASHIONPEDIA_ROOT / "instances_attributes_val2020.json"

# Project output paths
DATA_DIR = PROJECT_ROOT / "data"
SAMPLED_IMAGES_DIR = DATA_DIR / "images"
CAPTIONS_CSV_PATH = DATA_DIR / "captions.csv"
CAPTIONS_CHECKPOINT_PATH = DATA_DIR / "captions_checkpoint.json"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

# =============================================================================
# Sub-sampling Configuration
# =============================================================================
TARGET_SAMPLE_SIZE = 3000
# Take all non-static (isstatic=0) and unknown (isstatic=-1), fill rest with static
NON_STATIC_VALUES = [0, -1]
RANDOM_SEED = 42

# =============================================================================
# Model Configuration
# =============================================================================
# Bi-encoder for retrieval (Stage 1)
SIGLIP_MODEL_NAME = "Marqo/marqo-fashionSigLIP"
EMBEDDING_DIM = 768

# Cross-encoder reranker (Stage 2)
BLIP_MODEL_NAME = "Salesforce/blip-itm-base-coco"

# =============================================================================
# Indexing Configuration
# =============================================================================
# Hybrid fusion weight: v = alpha * v_img + (1 - alpha) * v_text
FUSION_ALPHA = 0.6
ALPHA_GRID = [0.3, 0.5, 0.6, 0.7, 0.8]

# ChromaDB collection settings
CHROMA_COLLECTION_NAME = "fashion_retrieval"
CHROMA_DISTANCE_METRIC = "cosine"

# Batch size for embedding extraction
VISION_BATCH_SIZE = 32
TEXT_BATCH_SIZE = 64

# =============================================================================
# Retrieval Configuration
# =============================================================================
# Number of candidates from bi-encoder
RETRIEVAL_TOP_K = 50
# Number of final results after reranking
RERANK_TOP_N = 5
# Minimum results from filtered search before falling back to unfiltered
FILTER_MIN_RESULTS = 10

# =============================================================================
# Gemini Caption Generation
# =============================================================================
GEMINI_MODEL_NAME = "gemini-2.0-flash"
GEMINI_RATE_LIMIT_DELAY = 1.1  # seconds between requests (60 req/min)
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_BASE_DELAY = 2  # seconds, exponential backoff base

# =============================================================================
# Known Categories & Attributes for Query Parsing
# =============================================================================
# Whole-body garment categories (for metadata filtering)
GARMENT_CATEGORIES = [
    "shirt", "blouse", "top", "t-shirt", "sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "pants", "shorts", "skirt", "coat", "dress", "jumpsuit",
    "cape", "hoodie", "blazer", "jeans", "leggings",
]

# Accessories
ACCESSORY_CATEGORIES = [
    "glasses", "hat", "headband", "tie", "glove", "watch", "belt",
    "tights", "stockings", "sock", "shoe", "bag", "wallet", "scarf", "umbrella",
]

# Environment keywords for query parsing
ENVIRONMENT_KEYWORDS = {
    "office": "office",
    "work": "office",
    "business": "office",
    "professional": "office",
    "corporate": "office",
    "street": "street",
    "urban": "street",
    "city": "street",
    "downtown": "street",
    "park": "park",
    "garden": "park",
    "outdoor": "park",
    "nature": "park",
    "home": "home",
    "house": "home",
    "indoor": "home",
    "cozy": "home",
    "event": "event",
    "party": "event",
    "formal": "event",
    "gala": "event",
    "wedding": "event",
    "beach": "beach",
    "seaside": "beach",
    "tropical": "beach",
    "gym": "gym",
    "workout": "gym",
    "athletic": "gym",
    "sport": "gym",
}

# Vibe keywords for query parsing
VIBE_KEYWORDS = {
    "casual": "casual",
    "relaxed": "casual",
    "laid-back": "casual",
    "weekend": "casual",
    "formal": "formal",
    "elegant": "elegant",
    "classy": "elegant",
    "sophisticated": "elegant",
    "sporty": "sporty",
    "athletic": "sporty",
    "streetwear": "streetwear",
    "bohemian": "bohemian",
    "boho": "bohemian",
}

# =============================================================================
# Evaluation Queries
# =============================================================================
EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]

# =============================================================================
# Fashionpedia Annotation Constants
# =============================================================================
# Whole-body category IDs (garments, not parts)
WHOLEBODY_CATEGORY_IDS = list(range(13)) + [23]  # 0-12 + shoe(23)

# Garment part category IDs (collar, sleeve, pocket, etc.)
GARMENT_PART_IDS = [27, 28, 29, 30, 31, 32, 33]  # hood, collar, lapel, epaulette, sleeve, pocket, neckline

# Decoration category IDs
DECORATION_IDS = [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

# Uninformative attribute IDs to filter out during tag extraction
BORING_ATTRIBUTE_IDS = [
    295,  # "no non-textile material"
    316,  # "no special manufacturing technique"
]

# Attribute names to exclude from captions (too generic)
BORING_ATTRIBUTE_NAMES = {
    "no non-textile material",
    "symmetrical",
    "no special manufacturing technique",
    "no waistline",
}
