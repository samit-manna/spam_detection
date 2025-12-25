# Model Serving Platform Makefile
#
# Usage:
#   make all               # Run full setup: build → export → deploy → test
#   make infra             # Deploy serving infrastructure via Terraform
#   make build-images      # Build and push all Docker images to ACR
#   make deploy-staging    # Deploy to staging
#   make deploy-prod       # Deploy to production
#   make test              # Run integration tests
#   make clean             # Clean up resources
#
# Note: Infrastructure (KServe, Istio, etc.) is managed by Terraform in ../terraform/serving-infra
# Note: ACR, Storage, and Redis details are automatically fetched from Terraform outputs

.PHONY: all infra build-images deploy-staging deploy-prod test clean help setup deploy-all

# Configuration
TERRAFORM_BASE_DIR = ../terraform/base-infra
TERRAFORM_SERVING_DIR = ../terraform/serving-infra
IMAGE_TAG ?= latest
MODEL_NAME ?= spam-detector
MODEL_STAGE ?= Staging

# Namespaces
NS_KSERVE = kserve
NS_SERVING = serving
NS_FEAST = feast
NS_RAY = ray

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m

# =============================================================================
# Full Setup Target - Runs all steps in sequence
# =============================================================================

all: setup
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)Full setup complete!$(NC)"
	@echo "$(GREEN)========================================$(NC)"

setup:
	@echo "$(YELLOW)========================================$(NC)"
	@echo "$(YELLOW)Starting Full Model Serving Setup$(NC)"
	@echo "$(YELLOW)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Step 1/9: Checking prerequisites...$(NC)"
	@$(MAKE) check-env
	@echo ""
	@echo "$(YELLOW)Step 2/9: Building Docker images...$(NC)"
	@$(MAKE) build-images
	@echo ""
	@echo "$(YELLOW)Step 3/9: Deploying API Gateway...$(NC)"
	@$(MAKE) deploy-api-gateway
	@echo ""
	@echo "$(YELLOW)Step 4/9: Deploying Feature Transformer...$(NC)"
	@$(MAKE) deploy-transformer
	@echo ""
	@echo "$(YELLOW)Step 5/9: Deploying Feast materialization...$(NC)"
	@$(MAKE) deploy-feast
	@echo ""
	@echo "$(YELLOW)Step 6/9: Exporting model from MLflow...$(NC)"
	@$(MAKE) export-model
	@echo ""
	@echo "$(YELLOW)Step 7/9: Deploying to staging...$(NC)"
	@$(MAKE) deploy-staging
	@echo ""
	@echo "$(YELLOW)Step 8/9: Running smoke tests...$(NC)"
	@$(MAKE) test
	@echo ""
	@echo "$(YELLOW)Step 9/9: Deploying to production...$(NC)"
	@$(MAKE) deploy-prod
	@echo ""

deploy-all: deploy-api-gateway deploy-transformer deploy-feast deploy-staging deploy-prod
	@echo "$(GREEN)All components deployed$(NC)"

# Help target
help:
	@echo "Model Serving Platform Commands"
	@echo ""
	@echo "Quick Start:"
	@echo "  make all               Run full setup (build → export → deploy → test)"
	@echo "  make setup             Same as 'make all'"
	@echo "  make deploy-all        Deploy all components (gateway, transformer, staging, prod)"
	@echo ""
	@echo "Infrastructure (Terraform):"
	@echo "  make infra             Deploy serving infra (KServe, Istio, Knative)"
	@echo "  make infra-plan        Preview infrastructure changes"
	@echo "  make infra-destroy     Destroy serving infrastructure"
	@echo "  make verify-infra      Verify infrastructure deployment"
	@echo ""
	@echo "Build & Push:"
	@echo "  make build-images      Build and push all Docker images to ACR"
	@echo "  make build-images IMAGE_TAG=v1.0.0  # Build with custom tag"
	@echo ""
	@echo "Model Lifecycle:"
	@echo "  make model-list        List all model versions and stages"
	@echo "  make model-promote     Promote model: VERSION=2 STAGE=Staging"
	@echo "  make model-rollback    Rollback model: STAGE=Staging"
	@echo "  make model-deploy      Export and deploy: STAGE=Staging"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-api-gateway Deploy API gateway"
	@echo "  make deploy-staging    Deploy staging inference service"
	@echo "  make deploy-prod       Deploy production inference service"
	@echo "  make deploy-transformer Deploy feature transformer"
	@echo "  make deploy-feast      Deploy Feast materialization cronjob"
	@echo "  make export-model      Export model from MLflow to ONNX"
	@echo "  make verify-onnx-conversion  Verify ONNX conversion quality (no upload)"
	@echo ""
	@echo "Operations:"
	@echo "  make materialize       Run Feast materialization"
	@echo "  make batch-predict     Run batch prediction job"
	@echo "  make test              Run integration tests"
	@echo "  make test-local        Run tests with docker-compose"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             Remove all deployed resources"
	@echo "  make clean-staging     Remove staging deployment only"
	@echo ""
	@echo "Note: Infrastructure is managed via Terraform in ../terraform/serving-infra"
	@echo "Note: ACR name is automatically fetched from Terraform outputs"

# Fetch terraform outputs
get-terraform-outputs:
	$(eval ACR_NAME := $(shell cd $(TERRAFORM_BASE_DIR) && terraform output -raw acr_login_server 2>/dev/null || echo ""))
	$(eval AZURE_STORAGE_ACCOUNT_NAME := $(shell cd $(TERRAFORM_BASE_DIR) && terraform output -raw storage_account_name 2>/dev/null || echo ""))
	$(eval AZURE_STORAGE_ACCOUNT_KEY := $(shell cd $(TERRAFORM_BASE_DIR) && terraform output -raw storage_account_primary_access_key 2>/dev/null || echo ""))
	$(eval REDIS_HOST := $(shell cd $(TERRAFORM_BASE_DIR) && terraform output -raw redis_hostname 2>/dev/null || echo ""))
	$(eval REDIS_KEY := $(shell cd $(TERRAFORM_BASE_DIR) && terraform output -raw redis_primary_access_key 2>/dev/null || echo ""))

# Check prerequisites
check-env: get-terraform-outputs
	@if [ -z "$(ACR_NAME)" ]; then \
		echo "$(RED)Error: Could not retrieve ACR name from Terraform outputs$(NC)"; \
		echo "Please ensure Terraform has been applied in $(TERRAFORM_BASE_DIR)"; \
		exit 1; \
	fi
	@if [ -z "$(AZURE_STORAGE_ACCOUNT_NAME)" ]; then \
		echo "$(RED)Error: Could not retrieve Storage Account name from Terraform outputs$(NC)"; \
		echo "Please ensure Terraform has been applied in $(TERRAFORM_BASE_DIR)"; \
		exit 1; \
	fi
	@echo "ACR Name: $(ACR_NAME)"
	@echo "Storage Account: $(AZURE_STORAGE_ACCOUNT_NAME)"

# =============================================================================
# Infrastructure targets (Terraform)
# =============================================================================

infra:
	@echo "$(YELLOW)Deploying serving infrastructure via Terraform...$(NC)"
	cd $(TERRAFORM_SERVING_DIR) && terraform init && terraform apply -auto-approve
	@echo "$(GREEN)Serving infrastructure deployed$(NC)"

infra-plan:
	@echo "$(YELLOW)Planning serving infrastructure changes...$(NC)"
	cd $(TERRAFORM_SERVING_DIR) && terraform init && terraform plan

infra-destroy:
	@echo "$(RED)Destroying serving infrastructure...$(NC)"
	cd $(TERRAFORM_SERVING_DIR) && terraform destroy
	@echo "$(GREEN)Serving infrastructure destroyed$(NC)"

verify-infra:
	@echo "$(YELLOW)Verifying serving infrastructure...$(NC)"
	@echo ""
	@echo "=== cert-manager ==="
	@kubectl get pods -n cert-manager --no-headers 2>/dev/null | head -3 || echo "Not deployed"
	@echo ""
	@echo "=== Istio ==="
	@kubectl get pods -n istio-system --no-headers 2>/dev/null | head -5 || echo "Not deployed"
	@echo ""
	@echo "=== Knative Serving ==="
	@kubectl get pods -n knative-serving --no-headers 2>/dev/null | head -5 || echo "Not deployed"
	@echo ""
	@echo "=== KServe ==="
	@kubectl get pods -n kserve --no-headers 2>/dev/null | head -3 || echo "Not deployed"
	@echo ""
	@echo "=== Istio Ingress Gateway ==="
	@kubectl get svc istio-ingressgateway -n istio-system 2>/dev/null || echo "Not deployed"
	@echo ""
	@echo "=== Terraform Outputs ==="
	@cd $(TERRAFORM_SERVING_DIR) && terraform output component_versions 2>/dev/null || echo "Run 'make infra' first"

# =============================================================================
# Build targets
# =============================================================================

build-images:
	@echo "Fetching ACR login server from Terraform outputs..."
	$(eval ACR_NAME := $(shell cd $(TERRAFORM_BASE_DIR) && terraform output -raw acr_login_server 2>/dev/null || echo ""))
	@if [ -z "$(ACR_NAME)" ]; then \
		echo "$(RED)Error: Could not retrieve ACR name from Terraform outputs$(NC)"; \
		echo "Please ensure Terraform has been applied in $(TERRAFORM_BASE_DIR)"; \
		exit 1; \
	fi
	@echo "ACR Name: $(ACR_NAME)"
	@echo "Image Tag: $(IMAGE_TAG)"
	@echo ""
	@bash scripts/build_images.sh $(ACR_NAME) $(IMAGE_TAG)

# Deployment targets
deploy-staging:
	@echo "$(YELLOW)Deploying staging inference service...$(NC)"
	@# No envsubst needed - YAML uses K8s secretKeyRef
	kubectl apply -f inference-service/staging-isvc.yaml
	kubectl wait --for=condition=Ready inferenceservice/spam-detector-staging -n $(NS_KSERVE) --timeout=300s
	@echo "$(GREEN)Staging deployment complete$(NC)"
	@kubectl get inferenceservice spam-detector-staging -n $(NS_KSERVE)

deploy-prod:
	@echo "$(YELLOW)Deploying production inference service...$(NC)"
	@# No envsubst needed - YAML uses K8s secretKeyRef
	kubectl apply -f inference-service/production-isvc.yaml
	kubectl wait --for=condition=Ready inferenceservice/spam-detector -n $(NS_KSERVE) --timeout=300s
	@echo "$(GREEN)Production deployment complete$(NC)"
	@kubectl get inferenceservice spam-detector -n $(NS_KSERVE)

deploy-transformer: get-terraform-outputs
	@echo "$(YELLOW)Deploying feature transformer...$(NC)"
	ACR_NAME=$(ACR_NAME) IMAGE_TAG=$(IMAGE_TAG) AZURE_STORAGE_ACCOUNT_NAME=$(AZURE_STORAGE_ACCOUNT_NAME) AZURE_STORAGE_ACCOUNT_KEY=$(AZURE_STORAGE_ACCOUNT_KEY) \
		REDIS_HOST=$(REDIS_HOST) REDIS_KEY=$(REDIS_KEY) \
		envsubst < feature-transformer/deployment.yaml | kubectl apply -f -
	kubectl wait --for=condition=Available --timeout=300s deployment/feature-transformer -n $(NS_SERVING)
	@echo "$(GREEN)Feature transformer deployed$(NC)"

deploy-api-gateway: get-terraform-outputs
	@echo "$(YELLOW)Deploying API gateway...$(NC)"
	ACR_NAME=$(ACR_NAME) IMAGE_TAG=$(IMAGE_TAG) envsubst < api-gateway/deployment.yaml | kubectl apply -f -
	kubectl apply -f api-gateway/istio.yaml
	kubectl wait --for=condition=Available --timeout=300s deployment/api-gateway -n $(NS_SERVING)
	@echo "$(GREEN)API gateway deployed$(NC)"
	@kubectl get pods -n $(NS_SERVING) -l app=api-gateway

deploy-feast: get-terraform-outputs
	@echo "$(YELLOW)Deploying Feast materialization cronjob...$(NC)"
	AZURE_STORAGE_ACCOUNT_NAME=$(AZURE_STORAGE_ACCOUNT_NAME) AZURE_STORAGE_ACCOUNT_KEY=$(AZURE_STORAGE_ACCOUNT_KEY) \
		REDIS_HOST=$(REDIS_HOST) REDIS_KEY=$(REDIS_KEY) \
		envsubst < feast/materialize_job.yaml | kubectl apply -f -
	@echo "$(GREEN)Feast materialization cronjob deployed$(NC)"
	@kubectl get cronjob -n $(NS_FEAST)

export-model: get-terraform-outputs
	@echo "$(YELLOW)Exporting model from MLflow...$(NC)"
	$(eval AZURE_STORAGE_CONNECTION_STRING := DefaultEndpointsProtocol=https;AccountName=$(AZURE_STORAGE_ACCOUNT_NAME);AccountKey=$(AZURE_STORAGE_ACCOUNT_KEY);EndpointSuffix=core.windows.net)
	kubectl run model-export-$(shell date +%s) \
		--image=$(ACR_NAME)/model-export:$(IMAGE_TAG) \
		--env="MLFLOW_TRACKING_URI=http://mlflow-service.mlflow.svc.cluster.local:5000" \
		--env="AZURE_STORAGE_ACCOUNT_NAME=$(AZURE_STORAGE_ACCOUNT_NAME)" \
		--env="AZURE_STORAGE_ACCOUNT_KEY=$(AZURE_STORAGE_ACCOUNT_KEY)" \
		--env="AZURE_STORAGE_CONNECTION_STRING=$(AZURE_STORAGE_CONNECTION_STRING)" \
		--restart=Never \
		-it \
		-- --model-name $(MODEL_NAME) --model-stage $(MODEL_STAGE)
	@echo "$(GREEN)Model exported$(NC)"

# Verify ONNX conversion quality (run before deployment)
verify-onnx-conversion: get-terraform-outputs
	@echo "$(YELLOW)Verifying ONNX conversion quality...$(NC)"
	@echo ""
	@echo "This will compare XGBoost vs ONNX predictions to detect conversion issues."
	@echo "If prediction difference > 0.1%, the export will fail."
	@echo ""
	$(eval AZURE_STORAGE_CONNECTION_STRING := DefaultEndpointsProtocol=https;AccountName=$(AZURE_STORAGE_ACCOUNT_NAME);AccountKey=$(AZURE_STORAGE_ACCOUNT_KEY);EndpointSuffix=core.windows.net)
	kubectl run onnx-verify-$(shell date +%s) \
		--image=$(ACR_NAME)/model-export:$(IMAGE_TAG) \
		--env="MLFLOW_TRACKING_URI=http://mlflow-service.mlflow.svc.cluster.local:5000" \
		--env="AZURE_STORAGE_ACCOUNT_NAME=$(AZURE_STORAGE_ACCOUNT_NAME)" \
		--env="AZURE_STORAGE_ACCOUNT_KEY=$(AZURE_STORAGE_ACCOUNT_KEY)" \
		--env="AZURE_STORAGE_CONNECTION_STRING=$(AZURE_STORAGE_CONNECTION_STRING)" \
		--restart=Never \
		-it \
		-- --model-name $(MODEL_NAME) --model-stage $(MODEL_STAGE) --no-upload
	@echo ""
	@echo "$(GREEN)✓ ONNX conversion validation passed$(NC)"
	@echo "The model can be safely exported without degradation."

# Operations targets
materialize: get-terraform-outputs
	@echo "$(YELLOW)Running Feast materialization...$(NC)"
	kubectl create job --from=cronjob/feast-materialize feast-materialize-manual-$(shell date +%s) -n $(NS_FEAST)
	@echo "$(GREEN)Materialization job created$(NC)"

batch-predict: get-terraform-outputs
	@echo "$(YELLOW)Submitting batch prediction job...$(NC)"
	ACR_NAME=$(ACR_NAME) IMAGE_TAG=$(IMAGE_TAG) AZURE_STORAGE_ACCOUNT_NAME=$(AZURE_STORAGE_ACCOUNT_NAME) AZURE_STORAGE_ACCOUNT_KEY=$(AZURE_STORAGE_ACCOUNT_KEY) \
		envsubst < batch-inference/rayjob.yaml | kubectl apply -f -
	@echo "$(GREEN)Batch job submitted$(NC)"
	@echo "Check status with: kubectl get rayjob batch-prediction -n $(NS_RAY)"

# Test targets
test: get-terraform-outputs
	@echo "$(YELLOW)Running integration tests...$(NC)"
	@# Get Istio ingress gateway IP for public access
	$(eval INGRESS_IP := $(shell kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}'))
	@echo "Ingress Gateway IP: $(INGRESS_IP)"
	@echo ""
	@echo "Testing via API Gateway at http://$(INGRESS_IP)"
	@# Test health endpoint with Host header
	@curl -s -H "Host: api.ml-platform.example.com" http://$(INGRESS_IP)/health | jq . || echo "Health check returned no JSON"
	@echo ""
	@# Run integration tests - uses API gateway public URL with Host header and API key
	python3 tests/integration_test.py \
		--api-gateway-url http://$(INGRESS_IP) \
		--host-header "api.ml-platform.example.com" \
		--api-key "test-operator-key" \
		--model-name $(MODEL_NAME)
	@echo "$(GREEN)Tests complete$(NC)"

# Test with port forwarding (for local debugging)
test-local-cluster: get-terraform-outputs
	@echo "$(YELLOW)Running integration tests with port forwarding...$(NC)"
	@pkill -f "port-forward.*8080" || true
	@pkill -f "port-forward.*8081" || true
	@sleep 1
	@kubectl port-forward svc/feature-transformer 8080:80 -n $(NS_SERVING) &
	@kubectl port-forward svc/spam-detector-staging-predictor 8081:80 -n $(NS_KSERVE) &
	@sleep 3
	python3 tests/integration_test.py \
		--transformer-url http://localhost:8080 \
		--triton-url http://localhost:8081 \
		--model-name $(MODEL_NAME)
	@pkill -f "port-forward svc/feature-transformer" || true
	@pkill -f "port-forward svc/spam-detector-staging-predictor" || true
	@echo "$(GREEN)Tests complete$(NC)"

test-local:
	@echo "$(YELLOW)Running local tests with docker-compose...$(NC)"
	docker-compose up -d
	@sleep 10
	docker-compose run test
	docker-compose down
	@echo "$(GREEN)Local tests complete$(NC)"

# Cleanup targets
clean: clean-staging clean-prod clean-transformer
	@echo "$(GREEN)All resources cleaned$(NC)"

clean-staging:
	@echo "$(YELLOW)Removing staging deployment...$(NC)"
	kubectl delete inferenceservice spam-detector-staging -n $(NS_KSERVE) --ignore-not-found

clean-prod:
	@echo "$(YELLOW)Removing production deployment...$(NC)"
	kubectl delete inferenceservice spam-detector -n $(NS_KSERVE) --ignore-not-found

clean-transformer:
	@echo "$(YELLOW)Removing feature transformer...$(NC)"
	kubectl delete deployment feature-transformer -n $(NS_SERVING) --ignore-not-found
	kubectl delete service feature-transformer -n $(NS_SERVING) --ignore-not-found

# Status targets
status:
	@echo "$(YELLOW)=== Inference Services ===$(NC)"
	@kubectl get inferenceservice -n $(NS_KSERVE) 2>/dev/null || echo "No inference services found"
	@echo ""
	@echo "$(YELLOW)=== Feature Transformer ===$(NC)"
	@kubectl get deployment,pod -l app=feature-transformer -n $(NS_SERVING) 2>/dev/null || echo "Not deployed"
	@echo ""
	@echo "$(YELLOW)=== Feast ===$(NC)"
	@kubectl get deployment,pod -n $(NS_FEAST) 2>/dev/null || echo "Not deployed"
	@echo ""
	@echo "$(YELLOW)=== Ray Jobs ===$(NC)"
	@kubectl get rayjob -n $(NS_RAY) 2>/dev/null || echo "No Ray jobs found"

logs-transformer:
	kubectl logs -f deployment/feature-transformer -n $(NS_SERVING)

logs-staging:
	kubectl logs -f -l serving.kserve.io/inferenceservice=spam-detector-staging -n $(NS_KSERVE)

logs-feast:
	kubectl logs -f deployment/feast-redis -n $(NS_FEAST)

# =============================================================================
# Model Lifecycle Management
# =============================================================================

# MLflow URI for model lifecycle commands
MLFLOW_URI ?= http://localhost:5000

# List all model versions
model-list:
	@echo "$(YELLOW)Listing model versions...$(NC)"
	@echo "Note: Ensure MLflow port-forward is running: kubectl port-forward svc/mlflow-service -n mlflow 5000:5000"
	@python3 scripts/model_lifecycle.py list --model-name $(MODEL_NAME) --mlflow-uri $(MLFLOW_URI)

# Promote a model version to a stage
# Usage: make model-promote VERSION=2 STAGE=Staging
model-promote:
	@if [ -z "$(VERSION)" ]; then echo "$(RED)Error: VERSION is required. Usage: make model-promote VERSION=2 STAGE=Staging$(NC)"; exit 1; fi
	@if [ -z "$(STAGE)" ]; then echo "$(RED)Error: STAGE is required. Usage: make model-promote VERSION=2 STAGE=Staging$(NC)"; exit 1; fi
	@echo "$(YELLOW)Promoting model version $(VERSION) to $(STAGE)...$(NC)"
	@python3 scripts/model_lifecycle.py promote --model-name $(MODEL_NAME) --version $(VERSION) --stage $(STAGE) --mlflow-uri $(MLFLOW_URI)

# Rollback to previous model version (MLflow only - does NOT deploy!)
# Usage: make model-rollback STAGE=Staging
# Note: Use model-rollback-deploy for full rollback with deployment
model-rollback:
	@if [ -z "$(STAGE)" ]; then echo "$(RED)Error: STAGE is required. Usage: make model-rollback STAGE=Staging$(NC)"; exit 1; fi
	@echo "$(YELLOW)Rolling back $(STAGE) to previous version in MLflow...$(NC)"
	@python3 scripts/model_lifecycle.py rollback --model-name $(MODEL_NAME) --stage $(STAGE) --mlflow-uri $(MLFLOW_URI)
	@echo ""
	@echo "$(YELLOW)⚠ WARNING: This only updated MLflow. The deployed model has NOT changed!$(NC)"
	@echo "$(YELLOW)  To deploy the rolled-back model, run: make model-deploy STAGE=$(STAGE)$(NC)"
	@echo "$(YELLOW)  Or use: make model-rollback-deploy STAGE=$(STAGE) for both in one step$(NC)"

# Full model deployment (export + deploy)
# Usage: make model-deploy STAGE=Staging
model-deploy: get-terraform-outputs
	@if [ -z "$(STAGE)" ]; then echo "$(RED)Error: STAGE is required. Usage: make model-deploy STAGE=Staging$(NC)"; exit 1; fi
	@echo "$(YELLOW)Deploying model to $(STAGE)...$(NC)"
	@echo ""
	@echo "Step 1: Exporting model from MLflow..."
	@python3 scripts/model_lifecycle.py export \
		--model-name $(MODEL_NAME) \
		--stage $(STAGE) \
		--acr-name $(ACR_NAME) \
		--image-tag $(IMAGE_TAG)
	@echo ""
	@echo "Step 2: Deploying to KServe..."
	@python3 scripts/model_lifecycle.py deploy \
		--model-name $(MODEL_NAME) \
		--stage $(STAGE)
	@echo ""
	@echo "$(GREEN)✓ Model deployment complete!$(NC)"

# Quick rollback and deploy (single command)
# Usage: make model-rollback-deploy STAGE=Staging
model-rollback-deploy: model-rollback
	@echo ""
	@echo "$(YELLOW)Deploying rolled-back model...$(NC)"
	$(MAKE) model-deploy STAGE=$(STAGE)
	@echo "$(GREEN)✓ Rollback and deployment complete!$(NC)"
