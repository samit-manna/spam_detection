# Monitoring Module Makefile
# ===========================

# Configuration
PYTHON := python3
PIP := pip3
PYTEST := pytest
IMAGE_TAG ?= latest
NAMESPACE := monitoring

# Terraform directories
TERRAFORM_BASE_DIR := ../terraform/base-infra

# Model configuration
MODEL_NAME ?= spam-detector
MODEL_VERSION ?= latest

# Storage configuration (can be overridden)
CONTAINER ?= feast
TRAINING_DATA_PATH ?= training/train_features.parquet

# Paths
BASELINE_PATH := baselines/$(MODEL_NAME)/$(MODEL_VERSION)
INFERENCE_LOGS_PATH := inference-logs
DRIFT_REPORTS_PATH := drift-reports

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: help install install-dev test test-cov lint format clean \
        build-images deploy undeploy logs status \
        generate-baseline run-drift-detection \
        get-terraform-outputs check-env

# Default target
.DEFAULT_GOAL := help

##@ General

help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Terraform Integration

get-terraform-outputs: ## Fetch values from Terraform outputs
	$(eval ACR_NAME := $(shell cd $(TERRAFORM_BASE_DIR) && terraform output -raw acr_login_server 2>/dev/null || echo ""))
	$(eval AZURE_STORAGE_ACCOUNT_NAME := $(shell cd $(TERRAFORM_BASE_DIR) && terraform output -raw storage_account_name 2>/dev/null || echo ""))
	$(eval AZURE_STORAGE_ACCOUNT_KEY := $(shell cd $(TERRAFORM_BASE_DIR) && terraform output -raw storage_account_primary_access_key 2>/dev/null || echo ""))

check-env: get-terraform-outputs ## Verify environment variables are set
	@echo "$(GREEN)Checking environment (from Terraform outputs)...$(NC)"
	@if [ -z "$(ACR_NAME)" ]; then \
		echo "$(RED)ERROR: Could not retrieve ACR name from Terraform outputs$(NC)"; \
		echo "Please ensure Terraform has been applied in $(TERRAFORM_BASE_DIR)"; \
		exit 1; \
	fi
	@if [ -z "$(AZURE_STORAGE_ACCOUNT_NAME)" ]; then \
		echo "$(RED)ERROR: Could not retrieve Storage Account name from Terraform outputs$(NC)"; \
		echo "Please ensure Terraform has been applied in $(TERRAFORM_BASE_DIR)"; \
		exit 1; \
	fi
	@echo "ACR Name: $(ACR_NAME)"
	@echo "Storage Account: $(AZURE_STORAGE_ACCOUNT_NAME)"
	@echo "MODEL_NAME: $(MODEL_NAME)"
	@echo "MODEL_VERSION: $(MODEL_VERSION)"
	@echo "$(GREEN)Environment OK$(NC)"

##@ Development

install: ## Install production dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt

install-dev: install ## Install development dependencies
	@echo "$(GREEN)Installing dev dependencies...$(NC)"
	$(PIP) install pytest pytest-cov pytest-asyncio black flake8 mypy

test: ## Run tests
	@echo "$(GREEN)Running tests...$(NC)"
	$(PYTEST) tests/ -v

test-cov: ## Run tests with coverage
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTEST) tests/ -v --cov=. --cov-report=html --cov-report=term-missing

test-synthetic: ## Run synthetic drift tests
	@echo "$(GREEN)Running synthetic drift tests...$(NC)"
	$(PYTEST) tests/test_synthetic_drift.py -v

lint: ## Run linting
	@echo "$(GREEN)Running linter...$(NC)"
	flake8 --max-line-length=120 --exclude=__pycache__,.git .
	mypy --ignore-missing-imports .

format: ## Format code with black
	@echo "$(GREEN)Formatting code...$(NC)"
	black --line-length=120 .

clean: ## Clean up generated files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info build dist

##@ Docker

build-images: get-terraform-outputs ## Build and push all Docker images to ACR
	@echo "$(GREEN)Building and pushing all images...$(NC)"
	@bash scripts/build_images.sh $(ACR_NAME) $(IMAGE_TAG)

##@ Kubernetes

deploy: get-terraform-outputs ## Deploy monitoring CronJob to Kubernetes
	@echo "$(GREEN)Deploying monitoring components...$(NC)"
	kubectl apply -f drift_detector/namespace.yaml
	@# Substitute environment variables in cronjob
	ACR_NAME=$(ACR_NAME) IMAGE_TAG=$(IMAGE_TAG) \
		AZURE_STORAGE_ACCOUNT_NAME=$(AZURE_STORAGE_ACCOUNT_NAME) \
		AZURE_STORAGE_ACCOUNT_KEY=$(AZURE_STORAGE_ACCOUNT_KEY) \
		MODEL_NAME=$(MODEL_NAME) \
		envsubst < drift_detector/cronjob.yaml | kubectl apply -f -
	@echo "$(GREEN)Monitoring deployed$(NC)"

undeploy: ## Remove monitoring from Kubernetes
	@echo "$(YELLOW)Removing monitoring components...$(NC)"
	kubectl delete -k drift_detector/ --ignore-not-found
	kubectl delete -f drift_detector/namespace.yaml --ignore-not-found

status: ## Check monitoring deployment status
	@echo "$(GREEN)Checking monitoring status...$(NC)"
	@echo "\n--- Namespace ---"
	kubectl get namespace $(NAMESPACE) 2>/dev/null || echo "Namespace not found"
	@echo "\n--- CronJobs ---"
	kubectl get cronjobs -n $(NAMESPACE) 2>/dev/null || echo "No CronJobs"
	@echo "\n--- Recent Jobs ---"
	kubectl get jobs -n $(NAMESPACE) --sort-by=.metadata.creationTimestamp 2>/dev/null | tail -5 || echo "No Jobs"
	@echo "\n--- Pods ---"
	kubectl get pods -n $(NAMESPACE) 2>/dev/null || echo "No Pods"

logs: ## View logs from latest drift detection job
	@echo "$(GREEN)Fetching logs from latest job...$(NC)"
	@JOB=$$(kubectl get jobs -n $(NAMESPACE) --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null); \
	if [ -n "$$JOB" ]; then \
		kubectl logs job/$$JOB -n $(NAMESPACE); \
	else \
		echo "$(RED)No jobs found$(NC)"; \
	fi

trigger-job: ## Manually trigger drift detection job
	@echo "$(GREEN)Triggering drift detection job...$(NC)"
	kubectl create job --from=cronjob/drift-detector drift-detector-manual-$$(date +%s) -n $(NAMESPACE)

##@ Local Operations

generate-baseline: get-terraform-outputs ## Generate baseline from training data (local)
	@echo "$(GREEN)Generating baseline...$(NC)"
	@echo "Using Storage Account: $(AZURE_STORAGE_ACCOUNT_NAME)"
	@echo "Container: $(CONTAINER)"
	@echo "Training data path: $(TRAINING_DATA_PATH)"
	AZURE_STORAGE_ACCOUNT_NAME=$(AZURE_STORAGE_ACCOUNT_NAME) \
	AZURE_STORAGE_ACCOUNT_KEY='$(AZURE_STORAGE_ACCOUNT_KEY)' \
	$(PYTHON) -m baseline.generator \
		--training-data "$(TRAINING_DATA_PATH)" \
		--model-name "$(MODEL_NAME)" \
		--model-version "$(MODEL_VERSION)" \
		--output-path "$(BASELINE_PATH)" \
		--container "$(CONTAINER)"

run-drift-detection: get-terraform-outputs ## Run drift detection locally
	@echo "$(GREEN)Running drift detection...$(NC)"
	@echo "Using Storage Account: $(AZURE_STORAGE_ACCOUNT_NAME)"
	AZURE_STORAGE_ACCOUNT_NAME=$(AZURE_STORAGE_ACCOUNT_NAME) \
	AZURE_STORAGE_ACCOUNT_KEY='$(AZURE_STORAGE_ACCOUNT_KEY)' \
	$(PYTHON) -m drift_detector.detector \
		--baseline-path "$(BASELINE_PATH)/baseline.json" \
		--inference-logs-path "$(INFERENCE_LOGS_PATH)" \
		--output-path "$(DRIFT_REPORTS_PATH)" \
		--window-hours 24 \
		--model-name "$(MODEL_NAME)" \
		--generate-html

run-drift-detection-dry: get-terraform-outputs ## Run drift detection in dry-run mode (no alerts)
	@echo "$(GREEN)Running drift detection (dry run)...$(NC)"
	AZURE_STORAGE_ACCOUNT_NAME=$(AZURE_STORAGE_ACCOUNT_NAME) \
	AZURE_STORAGE_ACCOUNT_KEY='$(AZURE_STORAGE_ACCOUNT_KEY)' \
	ALERT_ENABLED=false $(PYTHON) -m drift_detector.detector \
		--baseline-path "$(BASELINE_PATH)/baseline.json" \
		--inference-logs-path "$(INFERENCE_LOGS_PATH)" \
		--output-path "$(DRIFT_REPORTS_PATH)" \
		--window-hours 24 \
		--model-name "$(MODEL_NAME)" \
		--generate-html

##@ Utilities

list-baselines: get-terraform-outputs ## List available baselines in blob storage
	@echo "$(GREEN)Listing baselines...$(NC)"
	az storage blob list \
		--account-name $(AZURE_STORAGE_ACCOUNT_NAME) \
		--container-name $(CONTAINER) \
		--prefix "baselines/" \
		--query "[].name" \
		--output table

list-reports: get-terraform-outputs ## List recent drift reports
	@echo "$(GREEN)Listing drift reports...$(NC)"
	az storage blob list \
		--account-name $(AZURE_STORAGE_ACCOUNT_NAME) \
		--container-name $(CONTAINER) \
		--prefix "drift-reports/$(MODEL_NAME)/" \
		--query "[].{Name:name, LastModified:properties.lastModified}" \
		--output table

download-latest-report: get-terraform-outputs ## Download the latest drift report
	@echo "$(GREEN)Downloading latest report...$(NC)"
	@mkdir -p ./reports
	az storage blob download \
		--account-name $(AZURE_STORAGE_ACCOUNT_NAME) \
		--container-name $(CONTAINER) \
		--name "drift-reports/$(MODEL_NAME)/latest.json" \
		--file ./reports/latest.json
	@echo "Report saved to ./reports/latest.json"

view-metrics: get-terraform-outputs ## Quick view of drift metrics (requires jq)
	@echo "$(GREEN)Current Drift Metrics:$(NC)"
	@az storage blob download \
		--account-name $(AZURE_STORAGE_ACCOUNT_NAME) \
		--container-name $(CONTAINER) \
		--name "drift-reports/$(MODEL_NAME)/latest.json" \
		--file /tmp/drift_latest.json --output none 2>/dev/null && \
	jq '{drift_detected, drift_score, drifted_features: .drifted_features | length, timestamp}' /tmp/drift_latest.json || \
	echo "$(YELLOW)No drift report found or jq not installed$(NC)"
