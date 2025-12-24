.PHONY: help build-images compile-pipeline

# Default values
IMAGE_TAG ?= latest
TERRAFORM_DIR = ../terraform/base-infra

help:
	@echo "ML Training Pipeline - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  build-images       Build and push Docker images to ACR"
	@echo "  compile-pipeline   Compile Kubeflow pipeline YAML"
	@echo ""
	@echo "Usage:"
	@echo "  make build-images                    # Build with default tag 'latest'"
	@echo "  make build-images IMAGE_TAG=v1.0.0   # Build with custom tag"
	@echo "  make compile-pipeline                # Compile pipeline with latest tag"
	@echo "  make compile-pipeline IMAGE_TAG=v1.0.0  # Compile with custom tag"
	@echo ""
	@echo "Note: ACR name and Storage Account name are automatically fetched from Terraform outputs"

build-images:
	@echo "Fetching ACR login server from Terraform outputs..."
	$(eval ACR_NAME := $(shell cd $(TERRAFORM_DIR) && terraform output -raw acr_login_server 2>/dev/null || echo ""))
	@if [ -z "$(ACR_NAME)" ]; then \
		echo "Error: Could not retrieve ACR name from Terraform outputs"; \
		echo "Please ensure Terraform has been applied in $(TERRAFORM_DIR)"; \
		exit 1; \
	fi
	@echo "ACR Name: $(ACR_NAME)"
	@echo "Image Tag: $(IMAGE_TAG)"
	@echo ""
	@bash scripts/build_images.sh $(ACR_NAME) $(IMAGE_TAG)

compile-pipeline:
	@echo "Fetching configuration from Terraform outputs..."
	$(eval ACR_NAME := $(shell cd $(TERRAFORM_DIR) && terraform output -raw acr_login_server 2>/dev/null || echo ""))
	$(eval STORAGE_ACCOUNT := $(shell cd $(TERRAFORM_DIR) && terraform output -raw storage_account_name 2>/dev/null || echo ""))
	@if [ -z "$(ACR_NAME)" ]; then \
		echo "Error: Could not retrieve ACR name from Terraform outputs"; \
		echo "Please ensure Terraform has been applied in $(TERRAFORM_DIR)"; \
		exit 1; \
	fi
	@if [ -z "$(STORAGE_ACCOUNT)" ]; then \
		echo "Error: Could not retrieve Storage Account name from Terraform outputs"; \
		echo "Please ensure Terraform has been applied in $(TERRAFORM_DIR)"; \
		exit 1; \
	fi
	@echo "ACR Name: $(ACR_NAME)"
	@echo "Storage Account: $(STORAGE_ACCOUNT)"
	@echo "Image Tag: $(IMAGE_TAG)"
	@echo ""
	@cd pipeline && python spam_detection_pipeline.py $(ACR_NAME) $(IMAGE_TAG) $(STORAGE_ACCOUNT)
