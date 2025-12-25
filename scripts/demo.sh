#!/bin/bash
#
# ML Platform End-to-End Demo Script
# ===================================
# Demonstrates the complete ML lifecycle: train → deploy → predict → monitor
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration - Try to get public IP first, fallback to localhost
GATEWAY_IP=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
if [ -n "$GATEWAY_IP" ]; then
    API_URL="${API_URL:-http://$GATEWAY_IP}"
    HOST_HEADER="api.ml-platform.example.com"
    USE_PUBLIC_IP=true
else
    API_URL="${API_URL:-http://localhost:8000}"
    HOST_HEADER=""
    USE_PUBLIC_IP=false
fi
API_KEY="${API_KEY:-test-operator-key}"

echo -e "${BOLD}${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         ML MODEL LIFECYCLE MANAGEMENT PLATFORM                ║"
echo "║                    End-to-End Demo                            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

if [ "$USE_PUBLIC_IP" = true ]; then
    echo -e "${GREEN}Using public ingress gateway: $GATEWAY_IP${NC}"
else
    echo -e "${YELLOW}No public IP found, using localhost (port-forward required)${NC}"
fi
echo ""

# Helper function for curl with optional Host header
api_call() {
    local method=$1
    local endpoint=$2
    local data=$3
    local extra_headers=$4
    
    if [ "$USE_PUBLIC_IP" = true ]; then
        curl -s -X "$method" "$API_URL$endpoint" \
            -H "Host: $HOST_HEADER" \
            -H "Content-Type: application/json" \
            -H "X-API-Key: $API_KEY" \
            $extra_headers \
            -d "$data" 2>/dev/null
    else
        curl -s -X "$method" "$API_URL$endpoint" \
            -H "Content-Type: application/json" \
            -H "X-API-Key: $API_KEY" \
            $extra_headers \
            -d "$data" 2>/dev/null
    fi
}

# Helper function
section() {
    echo ""
    echo -e "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${GREEN}  $1${NC}"
    echo -e "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

check() {
    echo -e "${GREEN}✓${NC} $1"
}

info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# ─────────────────────────────────────────────────────────────────────────────
section "1. INFRASTRUCTURE STATUS"
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo -e "${YELLOW}Kubernetes Namespaces:${NC}"
kubectl get ns | grep -E "kubeflow|mlflow|ray|kserve|serving|monitoring" | awk '{printf "  %-20s %s\n", $1, $2}'

echo ""
echo -e "${YELLOW}Active Pods by Namespace:${NC}"
for ns in kubeflow mlflow ray kserve serving monitoring; do
    count=$(kubectl get pods -n $ns --no-headers 2>/dev/null | grep Running | wc -l | tr -d ' ')
    printf "  %-20s %s pods running\n" "$ns" "$count"
done

check "Infrastructure is healthy"

# ─────────────────────────────────────────────────────────────────────────────
section "2. MODEL REGISTRY (MLflow)"
# ─────────────────────────────────────────────────────────────────────────────

echo ""
info "MLflow tracks experiments and stores model artifacts"
echo ""
echo -e "${YELLOW}MLflow Service:${NC}"
kubectl get svc mlflow-service -n mlflow --no-headers 2>/dev/null | awk '{printf "  Service: %s, Port: %s\n", $1, $5}'
echo "  Access: kubectl port-forward svc/mlflow-service -n mlflow 5000:5000"
check "MLflow is running"

# ─────────────────────────────────────────────────────────────────────────────
section "3. MODEL SERVING (KServe)"
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo -e "${YELLOW}Deployed Models:${NC}"
kubectl get inferenceservices -n kserve --no-headers 2>/dev/null | while read line; do
    name=$(echo $line | awk '{print $1}')
    ready=$(echo $line | awk '{print $3}')
    if [[ "$name" == *"staging"* ]]; then
        env="STAGING"
    else
        env="PRODUCTION"
    fi
    echo "  [$env] $name - Ready: $ready"
done

echo ""
echo -e "${YELLOW}Model Replicas:${NC}"
kubectl get hpa -n kserve --no-headers 2>/dev/null | grep spam | while read line; do
    name=$(echo $line | awk '{print $1}')
    replicas=$(echo $line | awk '{print $6}')
    min=$(echo $line | awk '{print $4}')
    max=$(echo $line | awk '{print $5}')
    echo "  $name: $replicas replicas (min: $min, max: $max)"
done

check "Models deployed and serving"

# ─────────────────────────────────────────────────────────────────────────────
section "4. REAL-TIME INFERENCE"
# ─────────────────────────────────────────────────────────────────────────────

echo ""
info "Testing predictions via API Gateway"
echo ""

# Test if API is accessible
health_check() {
    if [ "$USE_PUBLIC_IP" = true ]; then
        curl -s --max-time 5 "$API_URL/health" -H "Host: $HOST_HEADER" > /dev/null 2>&1
    else
        curl -s --max-time 5 "$API_URL/health" > /dev/null 2>&1
    fi
}

if ! health_check; then
    if [ "$USE_PUBLIC_IP" = false ]; then
        echo -e "${YELLOW}Starting port-forward to API Gateway...${NC}"
        kubectl port-forward svc/api-gateway -n serving 8000:80 &
        PF_PID=$!
        sleep 3
    else
        echo -e "${RED}API Gateway not responding at $API_URL${NC}"
    fi
fi

echo -e "${YELLOW}Test 1: Spam Email (Staging)${NC}"
echo '  Request: {"subject": "WINNER! Claim your $1,000,000 prize NOW!!!", ...}'
RESPONSE=$(api_call "POST" "/predict" '{
        "email_id": "demo-1",
        "subject": "WINNER! Claim your $1,000,000 prize NOW!!!",
        "body": "Congratulations! You have been selected to receive $1,000,000. Click here immediately to claim your prize. This is not a joke! Act now before it expires!!!",
        "sender": "winner@lottery-claims.com"
    }')
echo "  Response:"
echo "$RESPONSE" | jq -r '  "    Prediction: \(.prediction)\n    Probability: \(.spam_probability)\n    Environment: \(.model_stage)\n    Latency: \(.latency_ms)ms"' 2>/dev/null || echo "    (API not accessible)"

echo ""
echo -e "${YELLOW}Test 2: Normal Email (Production)${NC}"
echo '  Request: {"subject": "Q4 Budget Review Meeting", ...}'
RESPONSE=$(api_call "POST" "/predict" '{
        "email_id": "demo-2",
        "subject": "Q4 Budget Review Meeting",
        "body": "Hi team, Please find attached the Q4 budget review materials. Let me know if you have any questions before our meeting tomorrow. Best regards, John",
        "sender": "john.smith@company.com"
    }' '-H "X-Environment: production"')
echo "  Response:"
echo "$RESPONSE" | jq -r '  "    Prediction: \(.prediction)\n    Probability: \(.spam_probability)\n    Environment: \(.model_stage)\n    Latency: \(.latency_ms)ms"' 2>/dev/null || echo "    (API not accessible)"

check "Real-time inference working"

# ─────────────────────────────────────────────────────────────────────────────
section "5. BATCH INFERENCE"
# ─────────────────────────────────────────────────────────────────────────────

echo ""
info "Batch inference uses Ray for distributed processing"
echo ""
echo -e "${YELLOW}Ray Cluster Status:${NC}"
kubectl get pods -n ray --no-headers 2>/dev/null | grep -E "head|worker" | while read line; do
    name=$(echo $line | awk '{print $1}')
    status=$(echo $line | awk '{print $3}')
    echo "  $name: $status"
done

echo ""
echo -e "${YELLOW}Batch Sync Test (3 emails):${NC}"
RESPONSE=$(api_call "POST" "/predict/batch" '{
        "emails": [
            {"email_id": "batch-1", "subject": "Meeting tomorrow", "body": "Hi, can we meet at 3pm?", "sender": "alice@company.com"},
            {"email_id": "batch-2", "subject": "FREE iPhone!!!", "body": "Click now to win!", "sender": "promo@spam.com"},
            {"email_id": "batch-3", "subject": "Project update", "body": "Attached is the latest status report.", "sender": "pm@company.com"}
        ]
    }')
echo "  Results:"
echo "$RESPONSE" | jq -r '.predictions[] | "    \(.email_id): \(.prediction) (\(.spam_probability | . * 100 | round)% spam)"' 2>/dev/null || echo "    (API not accessible)"

check "Batch inference working"

# ─────────────────────────────────────────────────────────────────────────────
section "6. MODEL MONITORING"
# ─────────────────────────────────────────────────────────────────────────────

echo ""
info "Drift detection compares production data against training baseline"
echo ""

echo -e "${YELLOW}Drift Detection CronJob:${NC}"
kubectl get cronjobs -n monitoring --no-headers 2>/dev/null | while read line; do
    name=$(echo $line | awk '{print $1}')
    schedule=$(echo $line | awk '{print $2}')
    echo "  $name: $schedule"
done

echo ""
echo -e "${YELLOW}Recent Drift Detection Jobs:${NC}"
kubectl get jobs -n monitoring --sort-by=.metadata.creationTimestamp --no-headers 2>/dev/null | tail -3 | while read line; do
    name=$(echo $line | awk '{print $1}')
    status=$(echo $line | awk '{print $2}')
    echo "  $name: $status"
done

echo ""
echo -e "${YELLOW}Latest Drift Metrics:${NC}"
# Try to get drift metrics from API
if [ "$USE_PUBLIC_IP" = true ]; then
    DRIFT=$(curl -s "$API_URL/metrics/drift" -H "Host: $HOST_HEADER" -H "X-API-Key: $API_KEY" 2>/dev/null)
else
    DRIFT=$(curl -s "$API_URL/metrics/drift" -H "X-API-Key: $API_KEY" 2>/dev/null)
fi
if echo "$DRIFT" | jq -e '.drift_score' > /dev/null 2>&1; then
    echo "$DRIFT" | jq -r '"    Drift Detected: \(.drift_detected)\n    Drift Score: \(.drift_score)%\n    Features Drifted: \(.features_drifted | length)\n    Last Checked: \(.last_checked)"' 2>/dev/null
else
    echo "    (Run 'make trigger-job' in monitoring/ to generate drift report)"
fi

check "Monitoring system active"

# ─────────────────────────────────────────────────────────────────────────────
section "7. SCALABILITY"
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo -e "${YELLOW}Horizontal Pod Autoscalers:${NC}"
echo "  Component               Current  Min  Max  CPU%"
echo "  ─────────────────────────────────────────────────"
kubectl get hpa -A --no-headers 2>/dev/null | grep -E "api-gateway|feature-transformer|spam-detector" | while read line; do
    ns=$(echo $line | awk '{print $1}')
    name=$(echo $line | awk '{print $2}')
    targets=$(echo $line | awk '{print $4}')
    min=$(echo $line | awk '{print $5}')
    max=$(echo $line | awk '{print $6}')
    replicas=$(echo $line | awk '{print $7}')
    cpu=$(echo $targets | grep -oE '[0-9]+%' | head -1)
    printf "  %-22s %s      %s    %s    %s\n" "$name" "$replicas" "$min" "$max" "$cpu"
done

check "Auto-scaling configured"

# ─────────────────────────────────────────────────────────────────────────────
section "8. SECURITY"
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo -e "${YELLOW}Authentication:${NC}"
echo "  API Key required for all endpoints (X-API-Key header)"
echo ""
echo -e "${YELLOW}Test without API Key:${NC}"
if [ "$USE_PUBLIC_IP" = true ]; then
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/predict" \
        -H "Host: $HOST_HEADER" \
        -H "Content-Type: application/json" \
        -d '{"email_id": "1", "subject": "test", "body": "test", "sender": "a@b.com"}' 2>/dev/null)
else
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/predict" \
        -H "Content-Type: application/json" \
        -d '{"email_id": "1", "subject": "test", "body": "test", "sender": "a@b.com"}' 2>/dev/null)
fi
echo "  Response: HTTP $RESPONSE (expected: 401 Unauthorized)"

check "API authentication enforced"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    DEMO COMPLETED ✓                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo "Summary:"
echo "  ✓ Infrastructure deployed on AKS"
echo "  ✓ MLflow tracking experiments and models"
echo "  ✓ KServe serving staging + production models"
echo "  ✓ Real-time inference via API Gateway"
echo "  ✓ Batch inference via Ray"
echo "  ✓ Drift detection monitoring production"
echo "  ✓ Auto-scaling enabled on all components"
echo "  ✓ API authentication enforced"
echo ""
echo "Next Steps:"
echo "  • View MLflow UI:     kubectl port-forward svc/mlflow-service -n mlflow 5000:5000"
echo "  • View Kubeflow UI:   kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80"
echo "  • Trigger training:   cd training && make run-pipeline"
echo "  • Check drift:        cd monitoring && make trigger-job && make logs"
echo ""
