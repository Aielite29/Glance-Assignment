"""
Sub-sample 3,000 images from the Fashionpedia train set.
Prioritizes images with natural backgrounds (isstatic=0 or -1).
"""
import json
import random
import logging
from tqdm.auto import tqdm
from src.config import (
    TRAIN_JSON_PATH, 
    DATA_DIR, 
    TARGET_SAMPLE_SIZE, 
    NON_STATIC_VALUES, 
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def subsample_images() -> list[dict]:
    """Select 3,000 images from Fashionpedia."""
    if not TRAIN_JSON_PATH.exists():
        raise FileNotFoundError(f"Train JSON not found at: {TRAIN_JSON_PATH}. Is the dataset mounted?")
    with open(TRAIN_JSON_PATH, "r") as f:
        data = json.load(f)
        
    images = data["images"]
    logging.info(f"Loaded {len(images)} total images from {TRAIN_JSON_PATH}")
    
    non_static = []
    static = []
    
    for img in tqdm(images, desc="Filtering images"):
        if img.get("isstatic") in NON_STATIC_VALUES:
            non_static.append(img)
        else:
            static.append(img)
            
    logging.info(f"Found {len(non_static)} non-static/unknown and {len(static)} static images.")
    
    selected = []
    random.seed(RANDOM_SEED)
    random.shuffle(non_static)  # avoid order bias
    selected.extend(non_static)
    
    remaining = TARGET_SAMPLE_SIZE - len(selected)
    if remaining > 0:
        # Clamp to available static images to avoid ValueError
        k = min(remaining, len(static))
        selected.extend(random.sample(static, k))
        
    # Cap if non_static alone exceeded the target
    selected = selected[:TARGET_SAMPLE_SIZE]
    
    logging.info(f"Selected {len(selected)} total images for the subset.")
    
    out_path = DATA_DIR / "sampled_images.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(selected, f)
        
    logging.info(f"Saved selected image metadata to {out_path}")
    return selected

if __name__ == "__main__":
    subsample_images()
