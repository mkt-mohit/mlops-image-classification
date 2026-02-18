"""
tests/test_preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for data preprocessing functions.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import get_image_paths, resize_image, split_paths


class TestSplitPaths:
    """Test split_paths function."""

    def test_split_paths_basic(self):
        """Test basic split with 100 items."""
        paths = list(range(100))
        train, val, test = split_paths(paths, train_ratio=0.8, val_ratio=0.1, seed=42)

        assert len(train) == 80, f"Expected 80 train, got {len(train)}"
        assert len(val) == 10, f"Expected 10 val, got {len(val)}"
        assert len(test) == 10, f"Expected 10 test, got {len(test)}"

    def test_split_paths_ratios(self):
        """Test different ratios."""
        paths = list(range(1000))
        train, val, test = split_paths(paths, train_ratio=0.7, val_ratio=0.15, seed=42)

        assert len(train) == 700
        assert len(val) == 150
        assert len(test) == 150

    def test_split_paths_reproducibility(self):
        """Test that same seed produces same split."""
        paths = list(range(100))

        train1, val1, test1 = split_paths(paths, 0.8, 0.1, seed=42)
        train2, val2, test2 = split_paths(paths, 0.8, 0.1, seed=42)

        assert train1 == train2, "Same seed should produce same train split"
        assert val1 == val2, "Same seed should produce same val split"
        assert test1 == test2, "Same seed should produce same test split"

    def test_split_paths_no_overlap(self):
        """Test that splits don't overlap."""
        paths = list(range(100))
        train, val, test = split_paths(paths, 0.8, 0.1, seed=42)

        combined = set(train + val + test)
        assert len(combined) == len(paths), "Splits should not overlap"
        assert combined == set(paths), "All items should be in one of the splits"

    def test_split_paths_small_dataset(self):
        """Test with very small dataset."""
        paths = list(range(10))
        train, val, test = split_paths(paths, train_ratio=0.5, val_ratio=0.2, seed=42)

        assert len(train) + len(val) + len(test) == 10
        assert len(train) >= 5
        assert len(val) >= 2

    def test_split_paths_single_item(self):
        """Test with single item."""
        paths = [0]
        train, val, test = split_paths(paths, 0.8, 0.1, seed=42)

        assert len(train) + len(val) + len(test) == 1


class TestResizeImage:
    """Test image resizing function."""

    def test_resize_image_size(self):
        """Test that image is resized to correct dimensions."""
        from PIL import Image
        import io

        # Create a small test image
        img = Image.new("RGB", (100, 100), color="red")

        # Resize to 224x224
        resized = resize_image(img, size=224)

        assert resized.size == (224, 224), f"Expected (224, 224), got {resized.size}"

    def test_resize_image_mode(self):
        """Test that image is converted to RGB."""
        from PIL import Image

        # Create a grayscale image
        img = Image.new("L", (100, 100), color=128)

        resized = resize_image(img, size=224)

        assert resized.mode == "RGB", f"Expected RGB, got {resized.mode}"

    def test_resize_image_aspect_ratio(self):
        """Test resizing non-square image."""
        from PIL import Image

        # Create rectangular image
        img = Image.new("RGB", (400, 200), color="blue")
        resized = resize_image(img, size=224)

        # Should fit into 224x224
        assert resized.size == (224, 224)


class TestGetImagePaths:
    """Test image path retrieval function."""

    def test_get_image_paths_with_valid_extensions(self, tmp_path):
        """Test getting image paths with valid extensions."""
        # Create test image files
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        (img_dir / "image1.jpg").touch()
        (img_dir / "image2.png").touch()
        (img_dir / "image3.jpeg").touch()
        (img_dir / "notimage.txt").touch()  # Should be ignored

        paths = get_image_paths(img_dir)

        assert len(paths) == 3
        assert all(p.suffix.lower() in {".jpg", ".png", ".jpeg"} for p in paths)

    def test_get_image_paths_sorted(self, tmp_path):
        """Test that paths are returned sorted."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        (img_dir / "zebra.jpg").touch()
        (img_dir / "apple.jpg").touch()
        (img_dir / "banana.jpg").touch()

        paths = get_image_paths(img_dir)
        path_names = [p.name for p in paths]

        assert path_names == sorted(path_names), "Paths should be sorted"

    def test_get_image_paths_empty_dir(self, tmp_path):
        """Test with empty directory."""
        img_dir = tmp_path / "empty"
        img_dir.mkdir()

        paths = get_image_paths(img_dir)

        assert len(paths) == 0, "Empty directory should return empty list"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
