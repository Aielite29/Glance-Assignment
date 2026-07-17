"""
Parse COCO annotations into structured tag strings for the sampled images.
"""
import json
import logging
from collections import defaultdict
from tqdm.auto import tqdm
from src.config import (
    TRAIN_JSON_PATH,
    DATA_DIR,
    WHOLEBODY_CATEGORY_IDS,
    GARMENT_PART_IDS,
    BORING_ATTRIBUTE_IDS,
    BORING_ATTRIBUTE_NAMES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def parse_annotations() -> list[dict]:
    sampled_path = DATA_DIR / "sampled_images.json"
    if not sampled_path.exists():
        raise FileNotFoundError(f"sampled_images.json not found. Run subsample_images() first.")
    with open(sampled_path, "r") as f:
        sampled = json.load(f)
        
    sampled_ids = {img["id"]: img for img in sampled}
    
    with open(TRAIN_JSON_PATH, "r") as f:
        data = json.load(f)
        
    cat_map = {c["id"]: c["name"].split(',')[0].strip() for c in data["categories"]}
    attr_map = {a["id"]: a["name"].lower() for a in data.get("attributes", data.get("attribute_categories", []))}
    
    # Convert to sets for O(1) membership lookup
    wholebody_ids = set(WHOLEBODY_CATEGORY_IDS)
    garment_part_ids = set(GARMENT_PART_IDS)
    boring_attr_ids = set(BORING_ATTRIBUTE_IDS)
    
    image_anns = defaultdict(list)
    for ann in data["annotations"]:
        if ann["image_id"] in sampled_ids:
            image_anns[ann["image_id"]].append(ann)
            
    results = []
    
    for img_id, img_info in tqdm(sampled_ids.items(), desc="Parsing annotations"):
        anns = image_anns.get(img_id, [])
        
        garment_strings = []
        part_strings = []
        
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id in wholebody_ids:
                cat_name = cat_map.get(cat_id, "unknown")
                attrs = []
                for a_id in ann.get("attribute_ids", []):
                    if a_id not in boring_attr_ids:
                        name = attr_map.get(a_id, "")
                        if name and name not in BORING_ATTRIBUTE_NAMES:
                            attrs.append(name)
                            
                tag_str = f"{cat_name}: {', '.join(attrs)}" if attrs else cat_name
                garment_strings.append(tag_str)
                
            elif cat_id in garment_part_ids:
                cat_name = cat_map.get(cat_id, "unknown")
                attrs = []
                for a_id in ann.get("attribute_ids", []):
                    if a_id not in boring_attr_ids:
                        name = attr_map.get(a_id, "")
                        if name and name not in BORING_ATTRIBUTE_NAMES:
                            attrs.append(name)
                            
                if attrs:
                    tag_str = f"{cat_name}: {', '.join(attrs)}"
                    part_strings.append(tag_str)
                    
        all_tags = garment_strings + part_strings
        tags_string = " | ".join(all_tags)
        
        results.append({
            "image_id": img_id,
            "file_name": img_info["file_name"],
            "isstatic": img_info.get("isstatic", 1),
            "tags_string": tags_string
        })
        
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "image_tags.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
        
    logging.info(f"Parsed annotations for {len(results)} images and saved to {out_path}")
    return results

if __name__ == "__main__":
    parse_annotations()
