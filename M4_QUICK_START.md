# M4 Quick Reference

## What Was Created

### 1. `docker-compose.yml`
- Deployment configuration for the containerized model
- Specifies image, ports, resources, health checks
- Can be run manually on any machine: `docker-compose up -d`

### 2. Updated `.github/workflows/ci.yml`
- Added `deploy-to-vm` job that runs after `build-and-push`
- Automatically deploys when code is pushed to main branch
- Runs smoke tests after deployment
- Fails pipeline if tests fail

### 3. `M4_DEPLOYMENT.md`
- Complete documentation of the deployment pipeline
- Setup instructions
- Troubleshooting guide

## GitHub Secrets to Add

**REQUIRED** - Add these to GitHub Settings → Secrets and variables → Actions:

```
VM_IP = 34.134.85.31
DEPLOY_KEY = (your SSH private key)
SSH_HOST_KEY = (output of: ssh-keyscan -H 34.134.85.31)
GCP_SA_KEY = (your GCP service account JSON)
GCP_PROJECT_ID = mlops-group113
```

## Deploy Manually (without GitHub)

If you want to deploy manually to your VM:

```bash
# SSH into VM
ssh -i ~/.ssh/github_actions root@34.134.85.31

# Navigate to deployment directory
mkdir -p ~/mlops-deployment
cd ~/mlops-deployment

# Create docker-compose.yml (copy from repo)
# Or download: 
curl -o docker-compose.yml https://raw.githubusercontent.com/YOUR_REPO/main/docker-compose.yml

# Configure Docker auth (if needed)
gcloud auth configure-docker gcr.io

# Pull and run
docker-compose pull
docker-compose up -d

# Check status
docker-compose ps
docker logs image-classification-api

# Test
curl http://localhost:8080/health
```

## Auto-Deploy via GitHub (Recommended)

1. Make changes to your code
2. Commit and push to main:
   ```bash
   git add .
   git commit -m "Update model"
   git push origin main
   ```

3. Watch GitHub Actions:
   - Check .github/workflows/ci.yml status
   - M3: Builds and pushes Docker image
   - M4: Deploys to VM and runs smoke tests

4. Access your service:
   - Health: http://34.134.85.31:8080/health
   - Docs: http://34.134.85.31:8080/docs
   - Predict: POST http://34.134.85.31:8080/predict

## Smoke Tests Automated

The pipeline automatically runs:

1. **Health Check** - Verifies API is running
2. **API Docs** - Verifies Swagger UI is accessible
3. **Prediction** - Sends test image and validates response

If any test fails, the pipeline fails and alerts you.

## Files Modified/Created

```
✅ docker-compose.yml (NEW)
✅ scripts/smoke_test.sh (NEW)
✅ M4_DEPLOYMENT.md (NEW)
✅ .github/workflows/ci.yml (UPDATED)
```

## Command Reference

### Test locally (before pushing)
```bash
docker build -t image-classification .
docker run -p 8080:8080 image-classification
curl http://localhost:8080/health
```

### View VM logs
```bash
ssh root@34.134.85.31
docker logs image-classification-api
```

### Restart service
```bash
ssh root@34.134.85.31
cd ~/mlops-deployment
docker-compose restart
```

### Stop service
```bash
ssh root@34.134.85.31
cd ~/mlops-deployment
docker-compose down
```

## Next Steps

1. ✅ Add GitHub Secrets (if not done)
2. ✅ Push changes to main branch
3. ✅ Monitor GitHub Actions workflow
4. ✅ Test endpoints on VM
5. ⏭️ Monitor and scale (M5+)

---

**Ready to deploy?** Push to main and watch the magic happen! 🚀
