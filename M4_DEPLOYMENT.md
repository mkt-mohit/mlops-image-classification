# M4: Continuous Deployment Pipeline

## Overview

M4 implements a complete CD (Continuous Deployment) pipeline that automatically deploys your containerized ML model to a GCP VM whenever changes are pushed to the main branch.

## Architecture

```
GitHub Repository (main branch)
    ↓
GitHub Actions (CI/CD)
    ├── M3: Build & Push Docker Image to GCP Registry
    └── M4: Deploy to VM & Run Smoke Tests
         ├── SSH into VM
         ├── Pull latest Docker image
         ├── Deploy using Docker Compose
         └── Run smoke tests (health check + prediction)
    ↓
GCP VM (34.134.85.31:8080)
    ↓
Running FastAPI Service (Cats vs Dogs Classification)
```

## Components

### 1. Docker Compose Configuration (`docker-compose.yml`)

Defines the containerized service with:
- Container image from GCP Container Registry
- Port mapping (8080:8080)
- Resource limits (CPU: 1, Memory: 2GB)
- Health checks (every 30 seconds)
- Restart policy (unless-stopped)
- JSON logging with rotation

### 2. GitHub Actions Workflow (`.github/workflows/ci.yml`)

The workflow has three jobs:

#### Job 1: Test
- Runs unit tests
- Generates coverage reports
- Uploads artifacts

#### Job 2: Build & Push (M3)
- Builds Docker image
- Pushes to GCP Container Registry (gcr.io/mlops-group113)

#### Job 3: Deploy to VM (M4)
- Deploys via SSH to your VM
- Installs Docker & Docker Compose if needed
- Pulls latest image from registry
- Deploys using docker-compose
- Runs smoke tests

### 3. Smoke Tests

The pipeline runs three smoke tests after deployment:

1. **Health Check Endpoint**
   - Endpoint: `GET /health`
   - Tests if API is running and model is loaded
   - Retries up to 5 times (every 5 seconds)

2. **API Documentation**
   - Endpoint: `GET /docs`
   - Verifies Swagger UI is accessible (non-critical)

3. **Prediction Endpoint**
   - Endpoint: `POST /predict`
   - Sends a test image
   - Validates response contains required fields (class, confidence)
   - Fails pipeline if prediction fails

## Prerequisites

### GitHub Secrets Required

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

| Secret Name | Description | Example |
|------------|-------------|---------|
| `VM_IP` | Public IP of your GCP VM | `34.134.85.31` |
| `DEPLOY_KEY` | SSH private key for authentication | (SSH private key content) |
| `SSH_HOST_KEY` | VM's SSH host key | (Output of ssh-keyscan) |
| `GCP_SA_KEY` | GCP Service Account key (JSON) | (Service account JSON) |
| `GCP_PROJECT_ID` | Your GCP project ID | `mlops-group113` |

### VM Requirements

- GCP VM (e2-small or e2-medium)
- OS: Debian 11 or Ubuntu 20.04 LTS
- Open ports: 22 (SSH), 8080 (HTTP)
- Public IP address
- SSH access enabled

## Setup Instructions

### Step 1: Generate SSH Keys

On your VM:

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/github_actions -N ""
cat ~/.ssh/github_actions.pub >> ~/.ssh/authorized_keys
```

### Step 2: Get SSH Host Key

```bash
ssh-keyscan -H 34.134.85.31
```

### Step 3: Add GitHub Secrets

1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Add all required secrets from the table above

### Step 4: Verify Setup

Push a change to main branch and monitor the GitHub Actions workflow:

```bash
git add .
git commit -m "Trigger CD pipeline"
git push origin main
```

## Deployment Flow

1. **Push to Main**
   ```
   git push origin main
   ```

2. **GitHub Actions Triggers**
   - Tests run
   - Docker image is built and pushed to GCP Registry

3. **Deployment to VM**
   - GitHub Actions SSH into your VM
   - Downloads latest docker-compose.yml
   - Pulls latest Docker image
   - Stops old container (if exists)
   - Starts new container using docker-compose

4. **Health Checks**
   - Pipeline waits for container to be healthy
   - Runs smoke tests
   - If tests pass → Deployment successful ✅
   - If tests fail → Deployment rollback

## Testing the Service

Once deployed, access your service:

### Health Check
```bash
curl http://34.134.85.31:8080/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### API Documentation
```
http://34.134.85.31:8080/docs
```

### Make a Prediction
```bash
curl -X POST "http://34.134.85.31:8080/predict" \
  -F "file=@path/to/image.jpg"
```

Response:
```json
{
  "filename": "image.jpg",
  "class": "cat",
  "confidence": 0.95,
  "probabilities": {
    "cat": 0.95,
    "dog": 0.05
  }
}
```

## Troubleshooting

### Deployment Fails - SSH Connection Error
- Verify `VM_IP` is correct
- Verify `DEPLOY_KEY` is the private key (starts with `-----BEGIN RSA PRIVATE KEY-----`)
- Verify `SSH_HOST_KEY` contains all lines from `ssh-keyscan`
- Check VM is running and port 22 is open

### Deployment Fails - Docker Pull Error
- Verify `GCP_SA_KEY` is valid
- Check if Docker image was built and pushed successfully (check M3 logs)
- Verify image name matches: `gcr.io/mlops-group113/mlops-group113/image-classification:latest`

### Health Check Fails
- SSH into VM and check container logs:
  ```bash
  docker logs image-classification-api
  ```
- Verify model file exists in container
- Check if port 8080 is exposed correctly

### Smoke Test Fails - Prediction Error
- Check if model is loaded: `curl http://34.134.85.31:8080/health`
- Check container logs for errors
- Verify model architecture matches inference code

## Accessing VM

SSH into your VM for debugging:

```bash
ssh -i ~/.ssh/github_actions root@34.134.85.31
```

Useful commands:

```bash
# Check running containers
docker ps

# View container logs
docker logs image-classification-api

# Check docker-compose status
cd ~/mlops-deployment
docker-compose ps

# Stop/start service
docker-compose down
docker-compose up -d
```

## Next Steps

- Monitor deployments in GitHub Actions
- Set up alerts for failed deployments
- Consider scaling to multiple VMs with load balancer (M5+)
- Implement canary deployments
- Add monitoring and logging aggregation

## Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GCP Container Registry](https://cloud.google.com/container-registry)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
