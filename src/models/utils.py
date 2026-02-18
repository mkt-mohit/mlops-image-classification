"""
src/models/utils.py
─────────────────────────────────────────────────────────────────────────────
Utility functions for saving/loading the model and running inference.
These helpers are also used by the FastAPI service (M2) and unit tests (M3).
"""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.models.cnn import BaselineCNN, build_model


# ── Label map ─────────────────────────────────────────────────────────────────
IDX_TO_CLASS = {0: "cat", 1: "dog"}

# ── ImageNet normalisation (must match training) ───────────────────────────────
_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: nn.Module, path: Path) -> None:
    """Save model state_dict to *path* (.pt)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(
    path: Path,
    architecture: str = "baseline_cnn",
    dropout: float = 0.5,
    device: str | None = None,
) -> nn.Module:
    """
    Load a saved state_dict into a freshly built model.
    Returns the model in eval mode on *device*.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(architecture, dropout)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image → normalised tensor of shape (1, 3, 224, 224).
    Works with any input size.
    """
    img = img.convert("RGB")
    tensor = _INFER_TRANSFORM(img)
    return tensor.unsqueeze(0)   # add batch dimension


def predict(
    model: nn.Module,
    img: Image.Image,
    device: str | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Run a single-image inference.

    Returns:
        {
            "label":       "cat" | "dog",
            "probability": float,   # P(dog)
            "class_idx":   int,     # 0=cat, 1=dog
        }
    """
    if device is None:
        device = next(model.parameters()).device

    tensor = preprocess_image(img).to(device)

    with torch.no_grad():
        logit = model(tensor)            # (1, 1)
        prob  = torch.sigmoid(logit).item()

    class_idx = int(prob >= threshold)
    return {
        "label":       IDX_TO_CLASS[class_idx],
        "probability": round(prob, 4),
        "class_idx":   class_idx,
    }