# Use lightweight Python image (smaller than default)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install only essential system dependencies (no CUDA/GPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install dependencies with --no-binary torch to get CPU-only version
# This avoids downloading heavy CUDA libraries
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision && \
    pip install --no-cache-dir \
    -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY params.yaml .

# Copy trained model
COPY models/baseline_cnn.pt ./models/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Run the API
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
