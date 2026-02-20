"""
src/api/app.py
─────────────────────────────────────────────────────────────────────────────
FastAPI inference service for Cats vs Dogs classification model.

Endpoints:
  GET  /health          - Health check
  POST /predict         - Predict class from uploaded image
  GET  /docs           - Interactive API documentation (Swagger UI)
"""

import io
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

# ── GCP Logging setup ──────────────────────────────────────────────────────────
try:
    from google.cloud import logging as cloud_logging
    gcp_logging_client = cloud_logging.Client()
    gcp_logger = gcp_logging_client.logger("image-classification-api")
    use_gcp_logging = True
except ImportError:
    gcp_logger = None
    use_gcp_logging = False

# ── Setup path for src imports ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.cnn import build_model

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
PARAMS_FILE = ROOT / "params.yaml"
MODEL_PATH = ROOT / "models" / "baseline_cnn.pt"
IMG_SIZE = 224
CLASS_NAMES = ["cat", "dog"]

# ImageNet normalization (standard for PyTorch models)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ── Global variables ──────────────────────────────────────────────────────────
model: Optional[nn.Module] = None
device: str = "cpu"


def load_model() -> nn.Module:
    """Load trained model from disk."""
    global model, device

    if not MODEL_PATH.exists():
        log.error("Model file not found: %s", MODEL_PATH)
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    log.info("Loading model from: %s", MODEL_PATH)

    # Load params to get model config
    with open(PARAMS_FILE) as f:
        params = yaml.safe_load(f)
    model_cfg = params["model"]

    # Build model architecture
    model = build_model(
        architecture=model_cfg["architecture"],
        dropout=model_cfg["dropout"],
    )

    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    log.info("✅ Model loaded successfully (device: %s)", device)
    return model


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Load image from bytes, resize to 224×224, normalize, and return tensor.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        torch.Tensor of shape (1, 3, 224, 224)
    """
    # Open image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    img_tensor = transform(img)

    # Add batch dimension
    return img_tensor.unsqueeze(0)


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="REST API for binary image classification (cats vs dogs)",
    version="1.0.0",
)


# ── Middleware for request/response logging ─────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log HTTP requests and responses to GCP Cloud Logging."""
    start_time = time.time()
    
    # Only log for non-health endpoints to reduce noise
    if request.url.path != "/health":
        if use_gcp_logging and gcp_logger:
            gcp_logger.log_struct({
                "severity": "INFO",
                "event": "request_received",
                "method": request.method,
                "path": request.url.path,
                "timestamp": datetime.now().isoformat()
            })
    
    response = await call_next(request)
    
    # Log response with latency
    if request.url.path != "/health":
        process_time = time.time() - start_time
        if use_gcp_logging and gcp_logger:
            gcp_logger.log_struct({
                "severity": "INFO",
                "event": "request_completed",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(process_time * 1000, 2),
                "timestamp": datetime.now().isoformat()
            })
    
    return response


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    if model is None:
        model = load_model()
    log.info("🚀 API started - Model ready for inference")


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
    }


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Predict class from uploaded image.
    
    Args:
        file: Image file (jpg, png, etc.)
        
    Returns:
        {
            "class": "cat" or "dog",
            "probabilities": {"cat": 0.95, "dog": 0.05},
            "confidence": 0.95
        }
    """
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (jpg, png, etc.)",
            )

        # Read image bytes
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")

        # Preprocess
        log.info("Processing image: %s", file.filename)
        img_tensor = preprocess_image(image_bytes)
        img_tensor = img_tensor.to(device)

        # Inference
        with torch.no_grad():
            logit = model(img_tensor)
            prob = torch.sigmoid(logit).item()

        # Format response
        predicted_class_idx = 1 if prob >= 0.5 else 0
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = prob if predicted_class_idx == 1 else (1 - prob)

        response = {
            "filename": file.filename,
            "class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "cat": round(1 - prob, 4),
                "dog": round(prob, 4),
            },
        }

        log.info(
            "✅ Prediction: %s (%.2f%%) for %s",
            predicted_class,
            confidence * 100,
            file.filename,
        )
        
        # Log to GCP Cloud Logging
        if use_gcp_logging and gcp_logger:
            gcp_logger.log_struct({
                "severity": "INFO",
                "event": "prediction_completed",
                "predicted_class": predicted_class,
                "confidence": round(confidence, 4),
                "probabilities": {
                    "cat": round(1 - prob, 4),
                    "dog": round(prob, 4),
                },
                "timestamp": datetime.now().isoformat()
            })

        return response

    except HTTPException:
        raise
    except Exception as e:
        log.error("Error during inference: %s", str(e))
        
        # Log error to GCP Cloud Logging
        if use_gcp_logging and gcp_logger:
            gcp_logger.log_struct({
                "severity": "ERROR",
                "event": "prediction_failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/", tags=["Info"])
async def root():
    """API documentation and info."""
    return {
        "message": "Cats vs Dogs Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "docs": "GET /docs (Swagger UI)",
            "redoc": "GET /redoc (ReDoc)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    log.info("Starting API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
