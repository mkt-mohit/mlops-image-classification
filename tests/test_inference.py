"""
tests/test_inference.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for model inference functions.
"""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.app import preprocess_image
from src.models.cnn import BaselineCNN, build_model


class TestModelLoading:
    """Test model loading and initialization."""

    def test_build_model_baseline(self):
        """Test building baseline CNN model."""
        model = build_model(architecture="baseline_cnn", dropout=0.5)

        assert isinstance(model, BaselineCNN)
        assert isinstance(model, torch.nn.Module)

    def test_build_model_output_shape(self):
        """Test model output shape."""
        model = build_model(architecture="baseline_cnn", dropout=0.5)
        model.eval()

        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, 1), f"Expected (1, 1), got {output.shape}"

    def test_baseline_cnn_layers(self):
        """Test that model has expected layers."""
        model = BaselineCNN(dropout=0.5)

        # Check for feature extractor
        assert hasattr(model, "features")
        assert hasattr(model, "pool")
        assert hasattr(model, "classifier")

    def test_baseline_cnn_forward(self):
        """Test forward pass."""
        model = BaselineCNN(dropout=0.5)
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 1)


class TestPreprocessing:
    """Test image preprocessing for inference."""

    def test_preprocess_image_shape(self):
        """Test that preprocessed image has correct shape."""
        # Create a test image
        img = Image.new("RGB", (100, 100), color="red")

        # Convert to bytes
        import io
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        # Preprocess
        tensor = preprocess_image(img_bytes)

        assert tensor.shape == (1, 3, 224, 224), f"Expected (1, 3, 224, 224), got {tensor.shape}"

    def test_preprocess_image_normalized(self):
        """Test that image is normalized (values in reasonable range)."""
        # Create a test image
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))

        import io
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        tensor = preprocess_image(img_bytes)

        # After normalization, values should be in approximate range [-2, 2]
        assert tensor.min() > -3 and tensor.max() < 3, "Image not properly normalized"

    def test_preprocess_image_dtype(self):
        """Test that output tensor has correct dtype."""
        img = Image.new("RGB", (100, 100), color="blue")

        import io
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        tensor = preprocess_image(img_bytes)

        assert tensor.dtype == torch.float32


class TestModelInference:
    """Test end-to-end inference."""

    def test_inference_produces_valid_output(self):
        """Test that model produces valid predictions."""
        model = build_model(architecture="baseline_cnn", dropout=0.5)
        model.eval()

        # Create dummy input
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            logit = model(x)
            prob = torch.sigmoid(logit)

        # Check output is valid probability
        assert 0 <= prob.item() <= 1, "Output should be probability between 0 and 1"

    def test_inference_batch(self):
        """Test inference with batch of images."""
        model = build_model(architecture="baseline_cnn", dropout=0.5)
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (batch_size, 1)

    def test_model_eval_mode(self):
        """Test that model is in eval mode for inference."""
        model = build_model(architecture="baseline_cnn", dropout=0.5)
        model.eval()

        # Dropout should be disabled
        assert not model.training


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
