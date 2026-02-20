# M5: Monitoring, Logs & Final Submission

## Overview

M5 implements comprehensive monitoring and logging for the deployed ML model using GCP Cloud Logging.

## What Was Added

### 1. GCP Cloud Logging Integration

The API now automatically logs all events to Google Cloud Logging:

**Logged Events:**
- Request received (method, path)
- Request completed (status code, latency)
- Prediction completed (class, confidence, probabilities)
- Errors (error messages, stack traces)

**Features:**
- ✅ Structured logging (JSON format)
- ✅ Timestamps in ISO format
- ✅ No sensitive data logged (no image bytes)
- ✅ Optional - works with or without GCP credentials
- ✅ Latency tracking for performance monitoring

### 2. Code Changes

**File: `src/api/app.py`**

Added:
```python
# GCP Logging setup (lines 16-22)
try:
    from google.cloud import logging as cloud_logging
    gcp_logging_client = cloud_logging.Client()
    gcp_logger = gcp_logging_client.logger("image-classification-api")
    use_gcp_logging = True
except ImportError:
    gcp_logger = None
    use_gcp_logging = False
```

This gracefully handles cases where GCP credentials are not available.

**Middleware for Request/Response Logging:**
```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log HTTP requests and responses to GCP Cloud Logging."""
    # Logs: method, path, status_code, latency_ms
```

**Prediction Event Logging:**
```python
if use_gcp_logging and gcp_logger:
    gcp_logger.log_struct({
        "severity": "INFO",
        "event": "prediction_completed",
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        ...
    })
```

**File: `requirements.txt`**

Added:
```
google-cloud-logging>=3.5.0
```

## How Logging Works

### Automatic Request Logging
Every HTTP request is logged with:
- Event type (request_received, request_completed)
- HTTP method
- Path
- Response status code
- Latency in milliseconds
- Timestamp

### Prediction Logging
Each prediction includes:
- Event: "prediction_completed"
- Predicted class (cat or dog)
- Confidence score
- Full probability distribution
- Timestamp

### Error Logging
Errors are logged with:
- Severity: "ERROR"
- Error message
- Timestamp

## Accessing Logs in GCP

### Via GCP Console:

1. Go to **Cloud Console** → **Logging** → **Logs Explorer**
2. Filter by logger name: `image-classification-api`
3. View in real-time or query by timestamp

### Sample Query:
```
resource.type="gae_app"
resource.labels.service_name="image-classification-api"
jsonPayload.event="prediction_completed"
```

### View Metrics:
```
jsonPayload.event="request_completed"
| stats avg(jsonPayload.latency_ms) as avg_latency
  by jsonPayload.status_code
```

## Performance Metrics Tracked

1. **Request Count**: Total number of predictions
2. **Latency**: Response time per request (in milliseconds)
3. **Error Rate**: Number of failed predictions
4. **Model Predictions**: Distribution of predictions (cat vs dog)

## Local Testing

Even without GCP credentials, the API works normally:
- Logs go to standard output (console)
- GCP logging is optional
- No failures if credentials are missing

Test locally:
```bash
# Build image
docker build -t image-classification .

# Run container
docker run -p 8080:8080 image-classification

# Make prediction
curl -X POST "http://localhost:8080/predict" \
  -F "file=@/tmp/test_image.jpg"

# Check logs in container
docker logs <container_id>
```

## GCP Authentication

The API automatically uses the service account from:
1. Environment variable: `GOOGLE_APPLICATION_CREDENTIALS`
2. Default application credentials (if running on GCP)
3. If no credentials found, logging is skipped gracefully

**For VM Deployment:**

To enable GCP logging on your VM, set credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
docker-compose up -d
```

Or mount credentials in docker-compose.yml:
```yaml
environment:
  - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
volumes:
  - /path/to/key.json:/app/credentials.json
```

## What's NOT Logged (Security)

- ❌ Image file contents
- ❌ File paths or user information
- ❌ Raw request/response bodies
- ❌ Model weights or architecture details

## Next Steps (M5 Completion)

1. ✅ Implement GCP Cloud Logging
2. ✅ Track request/response metrics
3. ⏭️ Create monitoring dashboard
4. ⏭️ Set up alerts for errors
5. ⏭️ Generate final submission package

## Files Modified

```
✅ src/api/app.py (updated with logging)
✅ requirements.txt (added google-cloud-logging)
✅ M5_MONITORING.md (this file)
```

---

**Ready for M5 Monitoring!** 🚀📊
