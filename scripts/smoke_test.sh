#!/bin/bash
#
# Smoke test script for image-classification API
# Tests:
#   1. Health endpoint is responding
#   2. API is accessible
#   3. Prediction endpoint works with a test image
#

set -e

API_URL="${1:-http://localhost:8080}"
TEST_IMAGE="${2:-tests/test_image.jpg}"
TIMEOUT=30
MAX_RETRIES=5

echo "🧪 Starting smoke tests for $API_URL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Helper function to retry requests
retry_request() {
    local url=$1
    local description=$2
    local attempt=1
    
    while [ $attempt -le $MAX_RETRIES ]; do
        echo "📍 $description (attempt $attempt/$MAX_RETRIES)..."
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "✅ $description - SUCCESS"
            return 0
        fi
        
        if [ $attempt -lt $MAX_RETRIES ]; then
            echo "⏳ Waiting 5 seconds before retry..."
            sleep 5
        fi
        attempt=$((attempt + 1))
    done
    
    echo "❌ $description - FAILED after $MAX_RETRIES attempts"
    return 1
}

# Test 1: Health Check
echo ""
echo "Test 1️⃣ : Health Check Endpoint"
echo "─────────────────────────────────"
if retry_request "$API_URL/health" "Health check"; then
    health_response=$(curl -s "$API_URL/health")
    echo "   Response: $health_response"
else
    echo "❌ Health check endpoint not responding!"
    exit 1
fi

# Test 2: API Documentation
echo ""
echo "Test 2️⃣ : API Documentation (Swagger UI)"
echo "─────────────────────────────────────────"
if retry_request "$API_URL/docs" "Documentation endpoint"; then
    echo "   Swagger UI is accessible"
else
    echo "⚠️  Documentation endpoint not responding (non-critical)"
fi

# Test 3: Prediction Endpoint
echo ""
echo "Test 3️⃣ : Prediction Endpoint"
echo "──────────────────────────────"

# Check if test image exists
if [ ! -f "$TEST_IMAGE" ]; then
    echo "⚠️  Test image not found at: $TEST_IMAGE"
    echo "   Using a dummy test instead..."
    
    # Create a simple test image (1x1 pixel RGB)
    python3 << 'EOF'
from PIL import Image
Image.new('RGB', (224, 224), color='red').save('/tmp/test_image.jpg')
EOF
    TEST_IMAGE="/tmp/test_image.jpg"
fi

echo "   Sending prediction request with test image..."
if curl -sf -F "file=@$TEST_IMAGE" "$API_URL/predict" > /tmp/prediction_response.json 2>&1; then
    echo "✅ Prediction endpoint - SUCCESS"
    echo "   Response:"
    cat /tmp/prediction_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/prediction_response.json
    
    # Validate response structure
    if grep -q '"class"' /tmp/prediction_response.json && grep -q '"confidence"' /tmp/prediction_response.json; then
        echo "✅ Response contains expected fields"
    else
        echo "⚠️  Response missing expected fields"
        exit 1
    fi
else
    echo "❌ Prediction endpoint - FAILED"
    cat /tmp/prediction_response.json
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All smoke tests passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit 0
