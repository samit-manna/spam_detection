# Postman Collection for Spam Detection API

This folder contains a Postman collection and environments for testing the Spam Detection ML API.

## Files

| File | Description |
|------|-------------|
| `Spam_Detection_API.postman_collection.json` | Main collection with all API tests |
| `Staging.postman_environment.json` | Environment variables for staging |
| `Production.postman_environment.json` | Environment variables for production |

## Importing into Postman

1. Open Postman
2. Click **Import** (top-left)
3. Drag and drop all three JSON files, or browse to select them
4. The collection and both environments will be imported

## Test Categories

### 1. Health
- **Health Check**: Verifies API Gateway is healthy

### 2. Predictions
- **Spam Email (Lottery)**: Tests detection of lottery scam
- **Spam Email (Phishing)**: Tests detection of phishing attempt
- **Ham Email (Work Meeting)**: Tests legitimate work email
- **Ham Email (Personal)**: Tests personal/friendly email
- **Staging Model**: Tests staging endpoint (for canary deployments)

### 3. Bias Tests
- Nigerian Prince Scam → Expected: SPAM
- Pharmaceutical Spam → Expected: SPAM
- Legitimate Newsletter → Expected: HAM
- Order Confirmation → Expected: HAM
- Job Scam → Expected: SPAM
- Password Reset (Legitimate) → Expected: HAM

### 4. Error Cases
- Missing API Key → Expected: 401/403
- Invalid Request Body → Expected: 400/422
- Empty Body → Expected: 400/422

## Environment Setup

Both environments use the **same Istio Ingress Gateway IP** - the difference is:
- `predict_path`: `/predict/staging` vs `/predict` (production)
- `api_key`: Test key vs production key

### Get Ingress IP

```bash
# Get Istio Ingress Gateway IP (same for both envs)
kubectl get svc istio-ingressgateway -n istio-system \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

### Staging Environment
| Variable | Value | Description |
|----------|-------|-------------|
| `ingress_ip` | `localhost:8080` | For port-forward, or actual IP |
| `predict_path` | `/predict/staging` | Routes to staging model |
| `api_key` | `test-operator-key` | Test API key |
| `host_header` | `api.ml-platform.example.com` | Istio routing |

### Production Environment
| Variable | Value | Description |
|----------|-------|-------------|
| `ingress_ip` | Same as staging | Same ingress gateway |
| `predict_path` | `/predict` | Routes to production model |
| `api_key` | `REPLACE_WITH_PROD_API_KEY` | Production key (keep secret!) |
| `host_header` | `api.ml-platform.example.com` | Istio routing |

## Running Tests

### Via Postman GUI

1. Select the environment (top-right dropdown)
2. Open the collection
3. Click **Run** to run all tests, or run individual requests

### Via Newman (CLI)

```bash
# Install Newman
npm install -g newman

# Run with staging environment
newman run Spam_Detection_API.postman_collection.json \
  -e Staging.postman_environment.json

# Run with production environment
newman run Spam_Detection_API.postman_collection.json \
  -e Production.postman_environment.json

# Generate HTML report
newman run Spam_Detection_API.postman_collection.json \
  -e Staging.postman_environment.json \
  -r htmlextra --reporter-htmlextra-export report.html
```

### Port-Forward Mode (Local Testing)

If you don't have a load balancer, use port-forwarding:

```bash
# Port-forward Istio ingress
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80 &

# Set ingress_ip to localhost:8080 in Staging environment
# Then run tests
```

## Test Assertions

Each request includes automated tests that verify:

1. **Status Code**: Expected HTTP status (200, 401, 400, etc.)
2. **Response Format**: Valid JSON response
3. **Response Fields**: Required fields present (`prediction`, `spam_probability`)
4. **Prediction Accuracy**: Expected classification (spam/ham)
5. **Response Time**: Within acceptable limits (<10s for predictions)

## Integration with CI/CD

Add to GitHub Actions:

```yaml
- name: Run API Tests
  run: |
    npm install -g newman
    newman run model-serving/postman/Spam_Detection_API.postman_collection.json \
      -e model-serving/postman/Staging.postman_environment.json \
      --env-var "ingress_ip=${{ env.INGRESS_IP }}" \
      --env-var "api_key=${{ secrets.TEST_API_KEY }}"
```

## Troubleshooting

### 401 Unauthorized
- Check that `api_key` is set correctly in environment
- Verify API key is valid in the API Gateway

### Connection Refused
- Verify `ingress_ip` is correct
- Check Istio Ingress Gateway is running
- Try port-forwarding if load balancer isn't available

### Host Header Issues
- Ensure `host_header` matches your VirtualService configuration
- Check Istio VirtualService is properly configured
