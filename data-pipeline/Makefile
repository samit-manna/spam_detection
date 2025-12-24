# Data Pipeline Makefile
# Orchestrates Ray jobs for data preprocessing

NAMESPACE := ray
TIMEOUT := 1800  # 30 minutes per stage

.PHONY: help run-pipeline download-dataset parse-emails extract-features \
        status logs clean wait-for-job list-jobs feast-apply feast-status

help:
	@echo "Data Pipeline - Ray Job Orchestration"
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
	@echo "  make status             Check status of all jobs"
	@echo "  make logs JOB=<name>    View logs for a specific job"
	@echo "  make list-jobs          List all RayJobs"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean              Delete all completed/failed jobs"
	@echo "  make clean-all          Delete ALL jobs (including running)"
	@echo "  make verify-data        Verify output data in Azure Blob"

# =============================================================================
# FULL PIPELINE
# =============================================================================

run-pipeline: clean
	@echo "============================================================"
	@echo "STARTING DATA PIPELINE"
	@echo "============================================================"
	@$(MAKE) download-dataset
	@$(MAKE) wait-for-job JOB=download-dataset
	@$(MAKE) parse-emails
	@$(MAKE) wait-for-job JOB=parse-emails
	@$(MAKE) extract-features
	@$(MAKE) wait-for-job JOB=extract-features
	@$(MAKE) feast-apply
	@echo "============================================================"
	@echo "DATA PIPELINE COMPLETED SUCCESSFULLY"
	@echo "============================================================"

# =============================================================================
# INDIVIDUAL STAGES
# =============================================================================

download-dataset:
	@echo "▶ Stage 1: Downloading SpamAssassin dataset..."
	@kubectl delete rayjob download-dataset -n $(NAMESPACE) --ignore-not-found
	@kubectl apply -f data-pipeline/rayjob-download-dataset.yaml
	@echo "  Job submitted. Use 'make logs JOB=download-dataset' to monitor."

parse-emails:
	@echo "▶ Stage 2: Parsing emails to Parquet..."
	@kubectl delete rayjob parse-emails -n $(NAMESPACE) --ignore-not-found
	@kubectl apply -f data-pipeline/rayjob-parse-emails.yaml
	@echo "  Job submitted. Use 'make logs JOB=parse-emails' to monitor."

extract-features:
	@echo "▶ Stage 3: Extracting ML features..."
	@kubectl delete rayjob extract-features -n $(NAMESPACE) --ignore-not-found
	@kubectl apply -f data-pipeline/rayjob-extract-features.yaml
	@echo "  Job submitted. Use 'make logs JOB=extract-features' to monitor."

# =============================================================================
# JOB MANAGEMENT
# =============================================================================

wait-for-job:
ifndef JOB
	$(error JOB is required. Usage: make wait-for-job JOB=download-dataset)
endif
	@echo "  Waiting for $(JOB) to complete..."
	@timeout=$(TIMEOUT); \
	while [ $$timeout -gt 0 ]; do \
		status=$$(kubectl get rayjob $(JOB) -n $(NAMESPACE) -o jsonpath='{.status.jobStatus}' 2>/dev/null); \
		if [ "$$status" = "SUCCEEDED" ]; then \
			echo "  ✓ $(JOB) completed successfully"; \
			exit 0; \
		elif [ "$$status" = "FAILED" ]; then \
			echo "  ✗ $(JOB) failed. Check logs with: make logs JOB=$(JOB)"; \
			exit 1; \
		fi; \
		sleep 10; \
		timeout=$$((timeout - 10)); \
		echo "    Status: $$status ($$timeout s remaining)"; \
	done; \
	echo "  ✗ $(JOB) timed out after $(TIMEOUT) seconds"; \
	exit 1

status:
	@echo "RayJob Status:"
	@echo "============================================================"
	@kubectl get rayjobs -n $(NAMESPACE) -o wide 2>/dev/null || echo "No RayJobs found"
	@echo ""
	@echo "Ray Cluster Status:"
	@kubectl get rayclusters -n $(NAMESPACE) -o wide 2>/dev/null || echo "No RayClusters found"

logs:
ifndef JOB
	$(error JOB is required. Usage: make logs JOB=download-dataset)
endif
	@echo "Logs for $(JOB):"
	@kubectl logs -n $(NAMESPACE) -l ray.io/job-name=$(JOB) --tail=100 -f 2>/dev/null || \
		kubectl logs -n $(NAMESPACE) job/$(JOB) --tail=100 2>/dev/null || \
		echo "No logs found for $(JOB). Job may not have started yet."

list-jobs:
	@kubectl get rayjobs -n $(NAMESPACE) -o custom-columns=\
NAME:.metadata.name,\
STATUS:.status.jobStatus,\
START:.status.startTime,\
END:.status.endTime

clean:
	@echo "Cleaning up completed/failed jobs..."
	@for job in download-dataset parse-emails extract-features; do \
		status=$$(kubectl get rayjob $$job -n $(NAMESPACE) -o jsonpath='{.status.jobStatus}' 2>/dev/null); \
		if [ "$$status" = "SUCCEEDED" ] || [ "$$status" = "FAILED" ]; then \
			kubectl delete rayjob $$job -n $(NAMESPACE) --ignore-not-found; \
			echo "  Deleted: $$job"; \
		fi; \
	done

clean-all:
	@echo "Deleting ALL RayJobs..."
	@kubectl delete rayjob download-dataset parse-emails extract-features -n $(NAMESPACE) --ignore-not-found
	@echo "Done."

# =============================================================================
# VERIFICATION
# =============================================================================

verify-data:
	@echo "Verifying pipeline outputs..."
	@echo ""
	@echo "Raw Dataset:"
	@az storage blob list --account-name $$AZURE_STORAGE_ACCOUNT_NAME \
		--container-name datasets --prefix raw/spamassassin/ \
		--query "length(@)" -o tsv 2>/dev/null | xargs -I {} echo "  Files: {}"
	@echo ""
	@echo "Processed Emails:"
	@az storage blob list --account-name $$AZURE_STORAGE_ACCOUNT_NAME \
		--container-name datasets --prefix processed/emails/ \
		--query "[].name" -o tsv 2>/dev/null | head -5
	@echo ""
	@echo "Feature Files:"
	@az storage blob list --account-name $$AZURE_STORAGE_ACCOUNT_NAME \
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
