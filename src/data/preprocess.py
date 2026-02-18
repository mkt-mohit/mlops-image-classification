"""
src/data/preprocess.py
─────────────────────────────────────────────────────────────────────────────
Reads raw images from  data/raw/cats/  and  data/raw/dogs/
Resizes to 224×224, splits 80/10/10, saves to data/processed/{train,val,test}
Augmentation is applied only to the training split at runtime (via dataset.py).
"""

import os
import shutil
import random
import logging
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]   # project root
RAW_DIR     = ROOT / "data" / "raw"
PROCESSED   = ROOT / "data" / "processed"
PARAMS_FILE = ROOT / "params.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers (also used by unit tests)
# ─────────────────────────────────────────────────────────────────────────────

def load_params() -> dict:
    """Load params.yaml and return the data section."""
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)["data"]


def resize_image(img: Image.Image, size: int = 224) -> Image.Image:
    """
    Resize a PIL image to (size × size) in RGB mode.
    Returns the resized image – does NOT save it.
    """
    img = img.convert("RGB")
    return img.resize((size, size), Image.LANCZOS)


def get_image_paths(class_dir: Path) -> list[Path]:
    """Return sorted list of valid image paths inside *class_dir*."""
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [
        p for p in sorted(class_dir.iterdir())
        if p.suffix.lower() in valid_ext
    ]
    return paths


def split_paths(
    paths: list,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Shuffle *paths* and split into (train, val, test) lists.
    test_ratio = 1 - train_ratio - val_ratio
    """
    random.seed(seed)
    shuffled = paths.copy()
    random.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]
    return train, val, test


def save_images(
    paths: list[Path],
    dest_dir: Path,
    img_size: int,
) -> int:
    """
    Resize each image in *paths* and save to *dest_dir*.
    Returns count of successfully saved images.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for src in tqdm(paths, desc=f"→ {dest_dir.relative_to(ROOT)}", leave=False):
        try:
            img = Image.open(src)
            img = resize_image(img, img_size)
            img.save(dest_dir / src.name)
            saved += 1
        except Exception as exc:
            log.warning("Skipping %s: %s", src.name, exc)
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED,
    params: dict | None = None,
) -> None:
    """
    Full preprocessing pipeline:
      raw/cats + raw/dogs  →  processed/{train,val,test}/{cats,dogs}
    """
    if params is None:
        params = load_params()

    img_size    = params["img_size"]
    train_ratio = params["train_split"]
    val_ratio   = params["val_split"]
    seed        = params["seed"]

    class_map = {"cats": raw_dir / "cats", "dogs": raw_dir / "dogs"}

    total_saved = 0

    for class_name, class_dir in class_map.items():
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected raw class folder not found: {class_dir}\n"
                "Please place your images under data/raw/cats/ and data/raw/dogs/"
            )

        log.info("Processing class: %s  (source: %s)", class_name, class_dir)
        paths = get_image_paths(class_dir)
        log.info("  Found %d images", len(paths))

        train_p, val_p, test_p = split_paths(paths, train_ratio, val_ratio, seed)
        log.info(
            "  Split → train=%d  val=%d  test=%d",
            len(train_p), len(val_p), len(test_p),
        )

        for split_name, paths_in_split in [
            ("train", train_p),
            ("val",   val_p),
            ("test",  test_p),
        ]:
            dest = processed_dir / split_name / class_name
            n = save_images(paths_in_split, dest, img_size)
            total_saved += n

    log.info("✅ Preprocessing complete. Total images saved: %d", total_saved)
    log.info("   Output directory: %s", processed_dir)


if __name__ == "__main__":
    preprocess()