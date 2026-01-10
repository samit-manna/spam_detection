# Data Pipeline Makefile
# Orchestrates Ray jobs for data preprocessing via port-forward to existing Ray cluster

NAMESPACE := ray
RAY_ADDRESS := http://localhost:8265
TIMEOUT := 1800  # 30 minutes per stage
SCRIPTS_DIR := data-pipeline
VENV := ../.venv/bin/activate

# Shell setup - activate venv for ray commands
SHELL := /bin/bash
.SHELLFLAGS := -c

.PHONY: help run-pipeline download-dataset parse-emails extract-features \
        status logs clean port-forward stop-port-forward check-ray-connection \
        feast-apply feast-status verify-data

help:
	@echo "Data Pipeline - Ray Job Orchestration"
	@echo ""
	@echo "Prerequisites:"
	@echo "  make port-forward       Start port-forward to Ray cluster (run in separate terminal)"
	@echo "  make check-ray-connection  Verify Ray cluster is accessible"
	@echo ""
	@echo "Pipeline Stages:"
	@echo "  make run-pipeline       Run complete pipeline (download → parse → extract)"
	@echo "  make download-dataset   Stage 1: Download SpamAssassin corpus"
	@echo "  make parse-emails       Stage 2: Parse raw emails to Parquet"
	@echo "  make extract-features   Stage 3: Extract ML features"
	@echo ""
	@echo "Feast Feature Store:"
	@echo "  make feast-apply        Register features in AKS (serverless)"
	@echo "  make feast-status       Check Feast job status"
	@echo ""
	@echo "Monitoring:"
	@echo "  make status             Check status of Ray cluster and jobs"
	@echo "  make logs               View Ray dashboard (open browser)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean              Clean up any old RayJob resources"
	@echo "  make verify-data        Verify output data in Azure Blob"

# =============================================================================
# PORT FORWARD MANAGEMENT
# =============================================================================

port-forward:
	@echo "Starting port-forward to Ray cluster..."
	@echo "Press Ctrl+C to stop"
	kubectl port-forward svc/ray-cluster-head-svc 8265:8265 10001:10001 -n $(NAMESPACE)

stop-port-forward:
	@echo "Stopping port-forward..."
	@pkill -f "kubectl port-forward.*ray-cluster-head-svc" || echo "No port-forward running"

check-ray-connection:
	@echo "Checking Ray cluster connection..."
	@curl -s $(RAY_ADDRESS) > /dev/null 2>&1 && \
		echo "✓ Ray cluster is accessible at $(RAY_ADDRESS)" || \
		(echo "✗ Cannot connect to Ray cluster. Run 'make port-forward' in another terminal first." && exit 1)

# Helper to get Azure credentials
define get_azure_creds
	$(eval AZURE_STORAGE_ACCOUNT_NAME := $(shell kubectl get secret ray-azure-secret -n $(NAMESPACE) -o jsonpath='{.data.azure-storage-account-name}' | base64 -d))
	$(eval AZURE_STORAGE_ACCOUNT_KEY := $(shell kubectl get secret ray-azure-secret -n $(NAMESPACE) -o jsonpath='{.data.azure-storage-account-key}' | base64 -d))
endef

# =============================================================================
# FULL PIPELINE
# =============================================================================

run-pipeline: check-ray-connection
	@echo "============================================================"
	@echo "STARTING DATA PIPELINE"
	@echo "============================================================"
	@$(MAKE) download-dataset
	@$(MAKE) parse-emails
	@$(MAKE) extract-features
	@$(MAKE) feast-apply
	@echo "============================================================"
	@echo "DATA PIPELINE COMPLETED SUCCESSFULLY"
	@echo "============================================================"

# =============================================================================
# INDIVIDUAL STAGES (using ray job submit)
# =============================================================================

download-dataset: check-ray-connection
	@echo "▶ Stage 1: Downloading SpamAssassin dataset..."
	$(call get_azure_creds)
	@source $(VENV) && ray job submit \
		--address $(RAY_ADDRESS) \
		--submission-id download-dataset-$$(date +%s) \
		--runtime-env-json '{ \
			"pip": ["adlfs>=2023.1.0", "azure-identity>=1.12.0"], \
			"env_vars": { \
				"AZURE_STORAGE_ACCOUNT_NAME": "$(AZURE_STORAGE_ACCOUNT_NAME)", \
				"AZURE_STORAGE_ACCOUNT_KEY": "$(AZURE_STORAGE_ACCOUNT_KEY)" \
			} \
		}' \
		--working-dir $(SCRIPTS_DIR) \
		-- python download_dataset.py
	@echo "  ✓ Stage 1 completed"

parse-emails: check-ray-connection
	@echo "▶ Stage 2: Parsing emails to Parquet..."
	$(call get_azure_creds)
	@source $(VENV) && ray job submit \
		--address $(RAY_ADDRESS) \
		--submission-id parse-emails-$$(date +%s) \
		--runtime-env-json '{ \
			"pip": ["adlfs>=2023.1.0", "pandas>=2.0.0", "pyarrow>=12.0.0"], \
			"env_vars": { \
				"AZURE_STORAGE_ACCOUNT_NAME": "$(AZURE_STORAGE_ACCOUNT_NAME)", \
				"AZURE_STORAGE_ACCOUNT_KEY": "$(AZURE_STORAGE_ACCOUNT_KEY)" \
			} \
		}' \
		--working-dir $(SCRIPTS_DIR) \
		-- python parse_emails.py
	@echo "  ✓ Stage 2 completed"

extract-features: check-ray-connection
	@echo "▶ Stage 3: Extracting ML features..."
	$(call get_azure_creds)
	@source $(VENV) && ray job submit \
		--address $(RAY_ADDRESS) \
		--submission-id extract-features-$$(date +%s) \
		--runtime-env-json '{ \
			"pip": ["adlfs>=2023.1.0", "pandas>=2.0.0", "pyarrow>=12.0.0", "scikit-learn==1.4.0"], \
			"env_vars": { \
				"AZURE_STORAGE_ACCOUNT_NAME": "$(AZURE_STORAGE_ACCOUNT_NAME)", \
				"AZURE_STORAGE_ACCOUNT_KEY": "$(AZURE_STORAGE_ACCOUNT_KEY)" \
			} \
		}' \
		--working-dir $(SCRIPTS_DIR) \
		-- python extract_features.py
	@echo "  ✓ Stage 3 completed"

# =============================================================================
# MONITORING
# =============================================================================

status:
	@echo "Ray Cluster Status:"
	@echo "============================================================"
	@kubectl get rayclusters -n $(NAMESPACE) -o wide 2>/dev/null || echo "No RayClusters found"
	@echo ""
	@echo "Ray Pods:"
	@kubectl get pods -n $(NAMESPACE) -l ray.io/cluster=ray-cluster 2>/dev/null || echo "No Ray pods found"
	@echo ""
	@echo "To view Ray Dashboard, run: make port-forward"
	@echo "Then open: http://localhost:8265"

logs:
	@echo "Opening Ray Dashboard..."
	@echo "Make sure port-forward is running (make port-forward in another terminal)"
	@open http://localhost:8265 2>/dev/null || echo "Open http://localhost:8265 in your browser"

clean:
	@echo "Cleaning up old RayJob resources (if any)..."
	@kubectl delete rayjob download-dataset parse-emails extract-features -n $(NAMESPACE) --ignore-not-found 2>/dev/null || true
	@echo "Done."

# =============================================================================
# VERIFICATION
# =============================================================================

verify-data:
	@echo "Verifying pipeline outputs..."
	$(call get_azure_creds)
	@echo ""
	@echo "Raw Dataset:"
	@az storage blob list --account-name $(AZURE_STORAGE_ACCOUNT_NAME) \
		--container-name datasets --prefix raw/spamassassin/ \
		--query "length(@)" -o tsv 2>/dev/null | xargs -I {} echo "  Files: {}"
	@echo ""
	@echo "Processed Emails:"
	@az storage blob list --account-name $(AZURE_STORAGE_ACCOUNT_NAME) \
		--container-name datasets --prefix processed/emails/ \
		--query "[].name" -o tsv 2>/dev/null | head -5
	@echo ""
	@echo "Feature Files:"
	@az storage blob list --account-name $(AZURE_STORAGE_ACCOUNT_NAME) \
		--container-name feast --prefix features/ \
		--query "[].name" -o tsv 2>/dev/null | head -10

# =============================================================================
# FEAST FEATURE STORE
# =============================================================================

feast-apply:
	@echo "▶ Registering features in Feast (serverless mode)..."
	@cd feast/feature_repo && ./apply_to_aks.sh

feast-status:
	@echo "Feast Apply Jobs:"
	@kubectl get jobs -n feast -l app=feast-apply -o wide 2>/dev/null || echo "No Feast jobs found"
	@echo ""
	@echo "Latest Feast job logs:"
	@kubectl logs -n feast -l app=feast-apply --tail=20 2>/dev/null || echo "No logs available"
