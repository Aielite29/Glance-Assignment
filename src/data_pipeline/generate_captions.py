"""
Rule-based caption and metadata generator for Glance Fashion Retrieval.

No API key required. Derives captions and structured metadata entirely from
the rich Fashionpedia tag strings produced by parse_annotations.py.
Runs in ~2 minutes on CPU with zero external calls.
"""
import json
import logging
import random
import re
import pandas as pd
from tqdm.auto import tqdm

from src.config import (
    DATA_DIR,
    CAPTIONS_CSV_PATH,
    CAPTIONS_CHECKPOINT_PATH,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# NOTE: do not call random.seed() globally — env inference uses per-image seeding for reproducibility

# ---------------------------------------------------------------------------
# Colour vocabulary — matched against attribute words
# ---------------------------------------------------------------------------
COLORS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "grey", "gray", "navy", "beige", "cream", "ivory",
    "maroon", "teal", "cyan", "magenta", "gold", "silver", "khaki",
    "olive", "coral", "turquoise", "lilac", "lavender", "burgundy",
    "charcoal", "tan", "camel", "mustard", "mint", "rose",
}

# ---------------------------------------------------------------------------
# Vibe inference — scored by garment keywords in tags
# ---------------------------------------------------------------------------
_VIBE_SCORES: dict[str, list[str]] = {
    "formal":     ["blazer", "suit", "tie", "oxford", "dress shirt", "trousers",
                   "waistcoat", "tuxedo", "gown", "evening", "formal"],
    "elegant":    ["gown", "silk", "velvet", "heels", "pearl", "lace",
                   "chiffon", "satin", "maxi dress", "cocktail"],
    "sporty":     ["track", "sweatpants", "jersey", "shorts", "running", "gym",
                   "athletic", "sport", "leggings", "hoodie", "sneakers",
                   "yoga", "cycling"],
    "streetwear": ["hoodie", "cargo", "beanie", "cap", "sneakers", "oversized",
                   "graphic", "bomber", "joggers", "denim jacket"],
    "bohemian":   ["floral", "boho", "fringe", "peasant", "maxi", "wrap",
                   "crochet", "embroidery", "printed"],
    "casual":     ["t-shirt", "jeans", "denim", "chinos", "polo", "sweatshirt",
                   "sweater", "cardigan", "loafers", "sandals", "shorts"],
}

# ---------------------------------------------------------------------------
# Environment inference — derived from vibe + isstatic
# ---------------------------------------------------------------------------
_VIBE_TO_ENV: dict[str, list[str]] = {
    "formal":     ["office", "office", "event"],
    "elegant":    ["event", "event", "office"],
    "sporty":     ["gym", "park", "street"],
    "streetwear": ["street", "street", "park"],
    "bohemian":   ["park", "street", "beach"],
    "casual":     ["street", "park", "home"],
}
_STATIC_ENV_BIAS: dict[str, str] = {
    "formal":     "office",
    "elegant":    "event",
    "sporty":     "gym",
    "streetwear": "street",
    "bohemian":   "park",
    "casual":     "home",
}

# ---------------------------------------------------------------------------
# Caption templates per vibe  (use {garments} and {env_phrase} placeholders)
# ---------------------------------------------------------------------------
_ENV_PHRASES: dict[str, list[str]] = {
    "office":  ["in a modern office", "heading into a business meeting",
                "at a professional workspace", "in a corporate setting"],
    "street":  ["on a busy city street", "strolling through an urban neighbourhood",
                "walking downtown", "on a city sidewalk"],
    "park":    ["relaxing in a sunny park", "taking a walk outdoors",
                "on a casual afternoon outside", "in a green outdoor setting"],
    "home":    ["at home on a relaxed day", "in a cosy indoor setting",
                "lounging at home", "in a comfortable home environment"],
    "event":   ["at a formal event", "attending a gala or party",
                "at an upscale social gathering", "at a special occasion"],
    "beach":   ["at the beach", "on a sunny seaside day",
                "enjoying a coastal outing", "by the ocean"],
    "gym":     ["at the gym", "during a workout session",
                "at an athletic training session", "in a fitness setting"],
}

_CAPTION_TEMPLATES: list[str] = [
    "A person wearing {garments}, {env_phrase}.",
    "Someone dressed in {garments}, {env_phrase}.",
    "An outfit featuring {garments}, {env_phrase}.",
    "A stylish look with {garments}, seen {env_phrase}.",
    "Wearing {garments} — a great choice {env_phrase}.",
]


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

def _parse_tags(tags_string: str) -> tuple[list[str], list[str]]:
    """Extract garment names and color words from a raw tags string."""
    garments: list[str] = []
    colors: list[str] = []

    if not tags_string.strip():
        return garments, colors

    # Each segment looks like "category: attr1, attr2, ..."
    for segment in tags_string.split("|"):
        segment = segment.strip()
        if not segment:
            continue
        if ":" in segment:
            cat, _, attrs_raw = segment.partition(":")
            garments.append(cat.strip())
            for word in re.split(r"[,\s]+", attrs_raw.lower()):
                if word in COLORS:
                    colors.append(word)
        else:
            garments.append(segment)

    return garments, list(dict.fromkeys(colors))  # deduplicate, preserve order


def _infer_vibe(tags_string: str) -> str:
    """Score each vibe by keyword hits; return the highest-scoring vibe."""
    tags_lower = tags_string.lower()
    scores: dict[str, int] = {v: 0 for v in _VIBE_SCORES}
    for vibe, keywords in _VIBE_SCORES.items():
        for kw in keywords:
            if kw in tags_lower:
                scores[vibe] += 1
    best = max(scores, key=lambda v: (scores[v], list(_VIBE_SCORES).index(v)))
    # Default to casual if nothing matched
    return best if scores[best] > 0 else "casual"


def _infer_env(vibe: str, isstatic: int, image_id: int = 0) -> str:
    """Infer environment from vibe and whether the photo is a studio shot.
    Uses a per-image random seed so results are reproducible across resumed runs.
    """
    if isstatic == 1:
        return _STATIC_ENV_BIAS.get(vibe, "street")
    options = _VIBE_TO_ENV.get(vibe, ["street"])
    return random.Random(image_id).choice(options)


def _build_caption(garments: list[str], env: str, colors: list[str]) -> str:
    """Compose a natural-language caption from structured fields."""
    if not garments:
        garment_str = "a stylish outfit"
    elif len(garments) == 1:
        garment_str = f"a {garments[0]}"
    elif len(garments) == 2:
        garment_str = f"a {garments[0]} and {garments[1]}"
    else:
        garment_str = (
            ", ".join(f"a {g}" for g in garments[:-1])
            + f", and {garments[-1]}"
        )

    if colors:
        color_prefix = " and ".join(colors[:2])  # max 2 colours in caption
        garment_str = f"{color_prefix} {garment_str}"

    env_phrase = random.choice(_ENV_PHRASES.get(env, ["in an everyday setting"]))
    template = random.choice(_CAPTION_TEMPLATES)
    return template.format(garments=garment_str, env_phrase=env_phrase)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_single(item: dict) -> dict:
    """Convert one image_tags record into a caption + metadata record."""
    tags_string = item.get("tags_string", "")
    isstatic = item.get("isstatic", 1)

    garments, colors = _parse_tags(tags_string)
    vibe = _infer_vibe(tags_string)
    env = _infer_env(vibe, isstatic, image_id=item["image_id"])
    caption = _build_caption(garments, env, colors)

    return {
        "image_id": item["image_id"],
        "file_name": item["file_name"],
        "caption": caption,
        "garments": json.dumps(garments),
        "colors": json.dumps(colors),
        "vibe": vibe,
        "env": env,
    }


def generate_all_captions(resume: bool = True) -> None:
    """
    Generate captions for all 3,000 sampled images using rule-based inference.

    No API key required. Runs entirely on CPU in ~2 minutes.

    Parameters
    ----------
    resume : bool
        If True, skip images already present in the checkpoint file.
    """
    tags_path = DATA_DIR / "image_tags.json"
    if not tags_path.exists():
        raise FileNotFoundError(
            f"image_tags.json not found at {tags_path}. Run parse_annotations() first."
        )

    with open(tags_path, "r") as f:
        image_tags = json.load(f)

    # Resume from checkpoint if requested
    done: dict[str, dict] = {}
    if resume and CAPTIONS_CHECKPOINT_PATH.exists():
        with open(CAPTIONS_CHECKPOINT_PATH, "r") as f:
            done = json.load(f)
        logging.info("Resumed from checkpoint: %d captions already done.", len(done))

    results: list[dict] = list(done.values())
    done_ids: set[str] = set(done.keys())

    pending = [item for item in image_tags if str(item["image_id"]) not in done_ids]
    logging.info("Processing %d remaining images…", len(pending))

    for item in tqdm(pending, desc="Generating captions"):
        record = process_single(item)
        results.append(record)
        done_ids.add(str(item["image_id"]))

        # Save checkpoint every 500 images
        if len(results) % 500 == 0:
            _save_checkpoint(results)

    _save_checkpoint(results)

    # Write final CSV
    df = pd.DataFrame(results)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CAPTIONS_CSV_PATH, index=False)
    logging.info(
        "Finished! %d captions saved to %s", len(df), CAPTIONS_CSV_PATH
    )


def _save_checkpoint(records: list[dict]) -> None:
    checkpoint = {str(r["image_id"]): r for r in records}
    CAPTIONS_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CAPTIONS_CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f)


if __name__ == "__main__":
    generate_all_captions()
