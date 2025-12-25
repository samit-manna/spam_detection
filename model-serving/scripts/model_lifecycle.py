#!/usr/bin/env python3
"""
Model Lifecycle Management Script

Provides commands for:
- Listing model versions and their stages
- Getting model info (for CI/CD pipelines)
- Promoting models between stages (None -> Staging -> Production)
- Rolling back to previous model versions
- Exporting models to ONNX format
- Deploying models to KServe

This script is designed to be used both locally and in CI/CD pipelines.
It uses kubectl and MLflow REST API directly without depending on Make targets.

Usage:
    python model_lifecycle.py list --model-name spam-detector
    python model_lifecycle.py get-model-info --model-name spam-detector --stage Staging
    python model_lifecycle.py promote --model-name spam-detector --version 2 --stage Staging
    python model_lifecycle.py rollback --model-name spam-detector --stage Staging
    python model_lifecycle.py export --model-name spam-detector --stage Staging
    python model_lifecycle.py deploy --model-name spam-detector --stage Staging
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", 
    "http://localhost:5000"
)

# Kubernetes namespaces
KSERVE_NAMESPACE = os.environ.get("KSERVE_NAMESPACE", "kserve")
MLFLOW_NAMESPACE = os.environ.get("MLFLOW_NAMESPACE", "mlflow")

# Default image settings
DEFAULT_IMAGE_TAG = os.environ.get("IMAGE_TAG", "v0.35")
ACR_NAME = os.environ.get("ACR_NAME", "")

VALID_STAGES = ["None", "Staging", "Production", "Archived"]
PROMOTION_ORDER = ["None", "Staging", "Production"]


class MLflowClient:
    """Simple MLflow REST API client."""
    
    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri.rstrip("/")
        self.api_base = f"{self.tracking_uri}/api/2.0/mlflow"
    
    def get_registered_model(self, name: str) -> Dict[str, Any]:
        """Get registered model details."""
        response = requests.get(
            f"{self.api_base}/registered-models/get",
            params={"name": name}
        )
        response.raise_for_status()
        return response.json().get("registered_model", {})
    
    def get_model_version(self, name: str, version: str) -> Dict[str, Any]:
        """Get specific model version details."""
        response = requests.get(
            f"{self.api_base}/model-versions/get",
            params={"name": name, "version": version}
        )
        response.raise_for_status()
        return response.json().get("model_version", {})
    
    def search_model_versions(self, name: str) -> List[Dict[str, Any]]:
        """Search all versions of a model."""
        response = requests.get(
            f"{self.api_base}/model-versions/search",
            params={"filter": f"name='{name}'"}
        )
        response.raise_for_status()
        return response.json().get("model_versions", [])
    
    def get_latest_versions(self, name: str, stages: List[str] = None) -> List[Dict[str, Any]]:
        """Get latest versions for specific stages."""
        payload = {"name": name}
        if stages:
            payload["stages"] = stages
        
        response = requests.post(
            f"{self.api_base}/registered-models/get-latest-versions",
            json=payload
        )
        response.raise_for_status()
        return response.json().get("model_versions", [])
    
    def transition_model_version_stage(
        self, 
        name: str, 
        version: str, 
        stage: str,
        archive_existing: bool = True
    ) -> Dict[str, Any]:
        """Transition a model version to a new stage."""
        response = requests.post(
            f"{self.api_base}/model-versions/transition-stage",
            json={
                "name": name,
                "version": version,
                "stage": stage,
                "archive_existing_versions": archive_existing
            }
        )
        response.raise_for_status()
        return response.json().get("model_version", {})
    
    def set_model_version_tag(
        self, 
        name: str, 
        version: str, 
        key: str, 
        value: str
    ):
        """Set a tag on a model version."""
        response = requests.post(
            f"{self.api_base}/model-versions/set-tag",
            json={
                "name": name,
                "version": version,
                "key": key,
                "value": value
            }
        )
        response.raise_for_status()


def run_kubectl(args: List[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """Run a kubectl command."""
    cmd = ["kubectl"] + args
    logger.debug(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=capture_output, text=True, check=check)


def list_models(client: MLflowClient, model_name: str, show_all: bool = False):
    """List all versions of a model with their stages."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        versions = client.search_model_versions(model_name)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch model versions: {e}")
        sys.exit(1)
    
    if not versions:
        print("No model versions found.")
        return
    
    # Sort by version number descending
    versions.sort(key=lambda x: int(x.get("version", 0)), reverse=True)
    
    # Filter if not showing all
    if not show_all:
        versions = [v for v in versions if v.get("current_stage") != "Archived"]
    
    # Print header
    print(f"{'Version':<10} {'Stage':<15} {'Created':<20} {'Tags'}")
    print("-" * 70)
    
    for v in versions:
        version = v.get("version", "?")
        stage = v.get("current_stage", "None")
        
        # Format timestamp
        ts = v.get("creation_timestamp", 0)
        created = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M") if ts else "Unknown"
        
        # Get relevant tags
        tags = v.get("tags", [])
        tag_str = ", ".join([f"{t['key']}={t['value']}" for t in tags[:3]])
        if len(tags) > 3:
            tag_str += "..."
        
        # Highlight current staging/production
        stage_display = stage
        if stage == "Staging":
            stage_display = f"\033[93m{stage}\033[0m"  # Yellow
        elif stage == "Production":
            stage_display = f"\033[92m{stage}\033[0m"  # Green
        elif stage == "Archived":
            stage_display = f"\033[90m{stage}\033[0m"  # Gray
        
        print(f"v{version:<9} {stage_display:<24} {created:<20} {tag_str}")
    
    print()
    
    # Show current active versions
    staging = [v for v in versions if v.get("current_stage") == "Staging"]
    production = [v for v in versions if v.get("current_stage") == "Production"]
    
    staging_str = f"v{staging[0]['version']}" if staging else "None"
    production_str = f"v{production[0]['version']}" if production else "None"
    
    print("Current Active Versions:")
    print(f"  Staging:    {staging_str}")
    print(f"  Production: {production_str}")
    print()


def get_model_info(
    client: MLflowClient, 
    model_name: str, 
    stage: str,
    output_format: str = "text"
) -> Optional[Dict[str, Any]]:
    """
    Get model info for a specific stage.
    
    Returns version, run_id, and other metadata.
    Useful for CI/CD pipelines that need to verify model existence.
    """
    try:
        versions = client.get_latest_versions(model_name, stages=[stage])
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch model versions: {e}")
        if output_format == "json":
            print(json.dumps({"error": str(e), "found": False}))
        sys.exit(1)
    
    if not versions:
        if output_format == "json":
            print(json.dumps({"found": False, "model_name": model_name, "stage": stage}))
        else:
            print(f"No model found in {stage} stage")
        return None
    
    model_version = versions[0]
    
    result = {
        "found": True,
        "model_name": model_name,
        "version": model_version.get("version"),
        "run_id": model_version.get("run_id"),
        "stage": model_version.get("current_stage"),
        "source": model_version.get("source"),
        "creation_timestamp": model_version.get("creation_timestamp"),
        "tags": {t["key"]: t["value"] for t in model_version.get("tags", [])}
    }
    
    if output_format == "json":
        print(json.dumps(result))
    else:
        print(f"\n{'='*60}")
        print(f"Model Info: {model_name}")
        print(f"{'='*60}")
        print(f"  Version:  v{result['version']}")
        print(f"  Stage:    {result['stage']}")
        print(f"  Run ID:   {result['run_id']}")
        print(f"  Source:   {result['source']}")
        if result['tags']:
            print(f"  Tags:")
            for k, v in list(result['tags'].items())[:5]:
                print(f"    {k}: {v}")
        print()
    
    return result


def promote_model(
    client: MLflowClient, 
    model_name: str, 
    version: str, 
    target_stage: str,
    skip_confirmation: bool = False
):
    """Promote a model version to a target stage."""
    if target_stage not in VALID_STAGES:
        logger.error(f"Invalid stage: {target_stage}. Must be one of {VALID_STAGES}")
        sys.exit(1)
    
    # Get current version info
    try:
        model_version = client.get_model_version(model_name, version)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get model version: {e}")
        sys.exit(1)
    
    current_stage = model_version.get("current_stage", "None")
    
    print(f"\n{'='*60}")
    print(f"Model Promotion")
    print(f"{'='*60}")
    print(f"  Model:         {model_name}")
    print(f"  Version:       v{version}")
    print(f"  Current Stage: {current_stage}")
    print(f"  Target Stage:  {target_stage}")
    print()
    
    # Check if this is a valid promotion
    if current_stage == target_stage:
        logger.warning(f"Model is already in {target_stage} stage")
        return
    
    # Confirm promotion
    if not skip_confirmation:
        confirm = input(f"Promote v{version} to {target_stage}? [y/N]: ")
        if confirm.lower() != 'y':
            print("Promotion cancelled.")
            return
    
    # Perform promotion
    try:
        result = client.transition_model_version_stage(
            model_name, 
            version, 
            target_stage,
            archive_existing=True
        )
        
        # Add promotion metadata
        timestamp = datetime.utcnow().isoformat()
        client.set_model_version_tag(model_name, version, "promoted_at", timestamp)
        client.set_model_version_tag(model_name, version, "promoted_from", current_stage)
        
        print(f"\n✓ Successfully promoted v{version} to {target_stage}")
        
        # Show next steps
        print(f"\nNext steps:")
        print(f"  1. Export model:  python model_lifecycle.py export --model-name {model_name} --stage {target_stage}")
        print(f"  2. Deploy:        python model_lifecycle.py deploy --model-name {model_name} --stage {target_stage}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to promote model: {e}")
        sys.exit(1)


def rollback_model(
    client: MLflowClient, 
    model_name: str, 
    stage: str,
    skip_confirmation: bool = False
):
    """Rollback to the previous model version in a stage."""
    if stage not in ["Staging", "Production"]:
        logger.error("Rollback only supported for Staging and Production stages")
        sys.exit(1)
    
    # Get all versions
    try:
        versions = client.search_model_versions(model_name)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch model versions: {e}")
        sys.exit(1)
    
    # Find current version in target stage
    current = [v for v in versions if v.get("current_stage") == stage]
    if not current:
        logger.error(f"No model currently in {stage} stage")
        sys.exit(1)
    
    current_version = current[0]
    
    # Find previous version (archived from this stage, or next lower version)
    archived = [
        v for v in versions 
        if v.get("current_stage") == "Archived"
        and int(v.get("version", 0)) < int(current_version.get("version", 0))
    ]
    
    # Sort by version descending to get most recent archived
    archived.sort(key=lambda x: int(x.get("version", 0)), reverse=True)
    
    if not archived:
        # Try to find any lower version that's not the current one
        lower_versions = [
            v for v in versions
            if int(v.get("version", 0)) < int(current_version.get("version", 0))
            and v.get("current_stage") != stage
        ]
        lower_versions.sort(key=lambda x: int(x.get("version", 0)), reverse=True)
        
        if not lower_versions:
            logger.error(f"No previous version available for rollback")
            sys.exit(1)
        
        rollback_to = lower_versions[0]
    else:
        rollback_to = archived[0]
    
    print(f"\n{'='*60}")
    print(f"Model Rollback")
    print(f"{'='*60}")
    print(f"  Model:           {model_name}")
    print(f"  Stage:           {stage}")
    print(f"  Current Version: v{current_version.get('version')}")
    print(f"  Rollback To:     v{rollback_to.get('version')}")
    print()
    
    # Confirm rollback
    if not skip_confirmation:
        confirm = input(f"Rollback {stage} from v{current_version.get('version')} to v{rollback_to.get('version')}? [y/N]: ")
        if confirm.lower() != 'y':
            print("Rollback cancelled.")
            return
    
    # Perform rollback
    try:
        result = client.transition_model_version_stage(
            model_name,
            rollback_to.get("version"),
            stage,
            archive_existing=True
        )
        
        # Add rollback metadata
        timestamp = datetime.utcnow().isoformat()
        client.set_model_version_tag(
            model_name, 
            rollback_to.get("version"), 
            "rollback_at", 
            timestamp
        )
        client.set_model_version_tag(
            model_name,
            rollback_to.get("version"),
            "rollback_from",
            current_version.get("version")
        )
        
        print(f"\n✓ Successfully rolled back {stage} to v{rollback_to.get('version')}")
        
        # Show next steps
        print(f"\nNext steps:")
        print(f"  1. Export model:  python model_lifecycle.py export --model-name {model_name} --stage {stage}")
        print(f"  2. Deploy:        python model_lifecycle.py deploy --model-name {model_name} --stage {stage}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to rollback model: {e}")
        sys.exit(1)


def export_model(
    model_name: str,
    stage: str,
    acr_name: str,
    image_tag: str = "v0.35",
    namespace: str = "kserve",
    timeout: int = 300
):
    """
    Export model from MLflow to ONNX format and upload to Azure Blob Storage.
    
    Creates a K8s Job that runs the model-export container.
    Uses azure-storage-secret K8s secret for credentials.
    """
    if stage not in ["Staging", "Production"]:
        logger.error("Export only supported for Staging and Production stages")
        sys.exit(1)
    
    if not acr_name:
        logger.error("ACR_NAME is required. Set via --acr-name or ACR_NAME env var")
        sys.exit(1)
    
    timestamp = int(time.time())
    job_name = f"model-export-{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"Model Export")
    print(f"{'='*60}")
    print(f"  Model:     {model_name}")
    print(f"  Stage:     {stage}")
    print(f"  Job Name:  {job_name}")
    print(f"  Image:     {acr_name}/model-export:{image_tag}")
    print()
    
    # Create Job manifest
    job_manifest = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {namespace}
  labels:
    app: model-export
    model-stage: {stage}
spec:
  ttlSecondsAfterFinished: 3600
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: model-export
    spec:
      restartPolicy: Never
      containers:
        - name: model-export
          image: {acr_name}/model-export:{image_tag}
          args:
            - "--model-name"
            - "{model_name}"
            - "--model-stage"
            - "{stage}"
          env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow-service.mlflow.svc.cluster.local:5000"
            - name: AZURE_STORAGE_ACCOUNT_NAME
              valueFrom:
                secretKeyRef:
                  name: azure-storage-secret
                  key: AZURE_STORAGE_ACCOUNT_NAME
            - name: AZURE_STORAGE_ACCOUNT_KEY
              valueFrom:
                secretKeyRef:
                  name: azure-storage-secret
                  key: AZURE_STORAGE_ACCESS_KEY
            - name: AZURE_STORAGE_CONNECTION_STRING
              valueFrom:
                secretKeyRef:
                  name: azure-storage-secret
                  key: AZURE_STORAGE_CONNECTION_STRING
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
"""
    
    # Apply the job
    print("Creating export job...")
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=job_manifest,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Failed to create job: {result.stderr}")
        sys.exit(1)
    
    print(f"✓ Job {job_name} created")
    
    # Wait for job to complete
    print(f"Waiting for job to complete (timeout: {timeout}s)...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check job status
        result = run_kubectl([
            "get", "job", job_name, "-n", namespace,
            "-o", "jsonpath={.status.succeeded},{.status.failed}"
        ], check=False)
        
        if result.returncode == 0:
            status = result.stdout.strip().split(",")
            succeeded = status[0] if status[0] else "0"
            failed = status[1] if len(status) > 1 and status[1] else "0"
            
            if succeeded == "1":
                print(f"\n✓ Model export completed successfully!")
                
                # Get job logs
                print("\nJob logs:")
                print("-" * 40)
                run_kubectl(["logs", f"job/{job_name}", "-n", namespace], capture_output=False, check=False)
                print("-" * 40)
                return
            
            if int(failed) > 0:
                print(f"\n✗ Model export failed!")
                print("\nJob logs:")
                run_kubectl(["logs", f"job/{job_name}", "-n", namespace], capture_output=False, check=False)
                sys.exit(1)
        
        # Show progress
        elapsed = int(time.time() - start_time)
        print(f"  Waiting... ({elapsed}s)", end="\r")
        time.sleep(5)
    
    logger.error(f"Job timed out after {timeout}s")
    print("\nJob logs:")
    run_kubectl(["logs", f"job/{job_name}", "-n", namespace], capture_output=False, check=False)
    sys.exit(1)


def deploy_model(
    model_name: str,
    stage: str,
    namespace: str = "kserve",
    timeout: int = 300,
    manifest_dir: str = None
):
    """
    Deploy model to KServe.
    
    Applies the InferenceService YAML and waits for it to be ready.
    """
    if stage not in ["Staging", "Production"]:
        logger.error("Deploy only supported for Staging and Production stages")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Model Deployment")
    print(f"{'='*60}")
    print(f"  Model:     {model_name}")
    print(f"  Stage:     {stage}")
    print(f"  Namespace: {namespace}")
    print()
    
    # Determine manifest path
    if manifest_dir is None:
        # Default to inference-service directory relative to script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        manifest_dir = os.path.join(os.path.dirname(script_dir), "inference-service")
    
    if stage == "Staging":
        manifest_path = os.path.join(manifest_dir, "staging-isvc.yaml")
        isvc_name = f"{model_name}-staging"
    else:
        manifest_path = os.path.join(manifest_dir, "production-isvc.yaml")
        isvc_name = model_name
    
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)
    
    print(f"Applying manifest: {manifest_path}")
    
    # Apply the manifest
    result = run_kubectl(["apply", "-f", manifest_path], check=False)
    if result.returncode != 0:
        logger.error(f"Failed to apply manifest: {result.stderr}")
        sys.exit(1)
    
    print(f"✓ InferenceService {isvc_name} applied")
    
    # Wait for InferenceService to be ready
    print(f"Waiting for InferenceService to be ready (timeout: {timeout}s)...")
    
    result = run_kubectl([
        "wait", "--for=condition=Ready",
        f"inferenceservice/{isvc_name}",
        "-n", namespace,
        f"--timeout={timeout}s"
    ], check=False)
    
    if result.returncode != 0:
        logger.error(f"InferenceService failed to become ready: {result.stderr}")
        
        # Get status for debugging
        print("\nInferenceService status:")
        run_kubectl(["get", "inferenceservice", isvc_name, "-n", namespace, "-o", "yaml"], capture_output=False, check=False)
        
        print("\nPod status:")
        run_kubectl(["get", "pods", "-n", namespace, "-l", f"serving.kserve.io/inferenceservice={isvc_name}"], capture_output=False, check=False)
        
        sys.exit(1)
    
    print(f"\n✓ InferenceService {isvc_name} is ready!")
    
    # Show status
    print("\nInferenceService status:")
    run_kubectl(["get", "inferenceservice", isvc_name, "-n", namespace], capture_output=False, check=False)


def smoke_test(
    model_name: str,
    stage: str,
    namespace: str = "kserve"
):
    """
    Run basic smoke tests against deployed model.
    """
    if stage not in ["Staging", "Production"]:
        logger.error("Smoke test only supported for Staging and Production stages")
        sys.exit(1)
    
    isvc_name = f"{model_name}-staging" if stage == "Staging" else model_name
    predictor_svc = f"{isvc_name}-predictor.{namespace}.svc.cluster.local"
    
    print(f"\n{'='*60}")
    print(f"Smoke Tests")
    print(f"{'='*60}")
    print(f"  InferenceService: {isvc_name}")
    print(f"  Predictor URL:    http://{predictor_svc}")
    print()
    
    # Test 1: Health check
    print("Test 1: Health Check")
    timestamp = int(time.time())
    pod_name = f"smoke-test-{timestamp}"
    
    result = subprocess.run([
        "kubectl", "run", pod_name,
        "-n", namespace,
        "--image=curlimages/curl:latest",
        "--restart=Never",
        "--rm", "-i",
        "--command", "--",
        "curl", "-sf", f"http://{predictor_svc}/v2/health/ready"
    ], capture_output=True, text=True, timeout=60)
    
    if result.returncode == 0:
        print("  ✓ Health check passed")
    else:
        print(f"  ✗ Health check failed: {result.stderr}")
        sys.exit(1)
    
    # Test 2: Model metadata
    print("\nTest 2: Model Metadata")
    timestamp = int(time.time())
    pod_name = f"smoke-test-{timestamp}"
    
    result = subprocess.run([
        "kubectl", "run", pod_name,
        "-n", namespace,
        "--image=curlimages/curl:latest",
        "--restart=Never",
        "--rm", "-i",
        "--command", "--",
        "curl", "-sf", f"http://{predictor_svc}/v2/models/spam-detector"
    ], capture_output=True, text=True, timeout=60)
    
    if result.returncode == 0:
        print("  ✓ Model metadata retrieved")
        try:
            metadata = json.loads(result.stdout)
            print(f"    Name: {metadata.get('name', 'N/A')}")
            print(f"    Versions: {metadata.get('versions', [])}")
        except json.JSONDecodeError:
            print(f"    Response: {result.stdout[:200]}")
    else:
        print(f"  ✗ Failed to get model metadata: {result.stderr}")
        # Not a fatal error - some models might not expose this
    
    print(f"\n{'='*60}")
    print("✓ Smoke tests passed!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Model Lifecycle Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all model versions
  python model_lifecycle.py list --model-name spam-detector

  # Get model info (for CI/CD)
  python model_lifecycle.py get-model-info --model-name spam-detector --stage Staging --format json

  # Promote version 2 to Staging
  python model_lifecycle.py promote --model-name spam-detector --version 2 --stage Staging

  # Rollback Staging to previous version
  python model_lifecycle.py rollback --model-name spam-detector --stage Staging

  # Export model to ONNX
  python model_lifecycle.py export --model-name spam-detector --stage Staging --acr-name myacr.azurecr.io

  # Deploy model to KServe
  python model_lifecycle.py deploy --model-name spam-detector --stage Staging

  # Run smoke tests
  python model_lifecycle.py smoke-test --model-name spam-detector --stage Staging
        """
    )
    
    # Parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--mlflow-uri", 
        default=MLFLOW_TRACKING_URI,
        help="MLflow tracking URI"
    )
    parent_parser.add_argument(
        "--namespace",
        default=KSERVE_NAMESPACE,
        help="Kubernetes namespace for KServe"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List model versions", parents=[parent_parser])
    list_parser.add_argument("--model-name", required=True, help="Model name")
    list_parser.add_argument("--all", action="store_true", help="Show archived versions")
    
    # Get model info command (for CI/CD)
    info_parser = subparsers.add_parser("get-model-info", help="Get model info for a stage", parents=[parent_parser])
    info_parser.add_argument("--model-name", required=True, help="Model name")
    info_parser.add_argument("--stage", required=True, choices=["Staging", "Production"], help="Stage to query")
    info_parser.add_argument("--format", dest="output_format", default="text", choices=["text", "json"], help="Output format")
    
    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Promote model to stage", parents=[parent_parser])
    promote_parser.add_argument("--model-name", required=True, help="Model name")
    promote_parser.add_argument("--version", required=True, help="Version to promote")
    promote_parser.add_argument("--stage", required=True, choices=VALID_STAGES, help="Target stage")
    promote_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to previous version", parents=[parent_parser])
    rollback_parser.add_argument("--model-name", required=True, help="Model name")
    rollback_parser.add_argument("--stage", required=True, choices=["Staging", "Production"], help="Stage to rollback")
    rollback_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX", parents=[parent_parser])
    export_parser.add_argument("--model-name", required=True, help="Model name")
    export_parser.add_argument("--stage", required=True, choices=["Staging", "Production"], help="Stage to export")
    export_parser.add_argument("--acr-name", default=ACR_NAME, help="Azure Container Registry name")
    export_parser.add_argument("--image-tag", default=DEFAULT_IMAGE_TAG, help="Docker image tag")
    export_parser.add_argument("--timeout", type=int, default=300, help="Job timeout in seconds")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy model to KServe", parents=[parent_parser])
    deploy_parser.add_argument("--model-name", required=True, help="Model name")
    deploy_parser.add_argument("--stage", required=True, choices=["Staging", "Production"], help="Stage to deploy")
    deploy_parser.add_argument("--manifest-dir", help="Directory containing InferenceService manifests")
    deploy_parser.add_argument("--timeout", type=int, default=300, help="Deployment timeout in seconds")
    
    # Smoke test command
    test_parser = subparsers.add_parser("smoke-test", help="Run smoke tests", parents=[parent_parser])
    test_parser.add_argument("--model-name", required=True, help="Model name")
    test_parser.add_argument("--stage", required=True, choices=["Staging", "Production"], help="Stage to test")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize client for commands that need MLflow
    client = None
    if args.command in ["list", "get-model-info", "promote", "rollback"]:
        client = MLflowClient(args.mlflow_uri)
    
    # Execute command
    if args.command == "list":
        list_models(client, args.model_name, args.all)
    elif args.command == "get-model-info":
        get_model_info(client, args.model_name, args.stage, args.output_format)
    elif args.command == "promote":
        promote_model(client, args.model_name, args.version, args.stage, args.yes)
    elif args.command == "rollback":
        rollback_model(client, args.model_name, args.stage, args.yes)
    elif args.command == "export":
        export_model(
            args.model_name, 
            args.stage, 
            args.acr_name,
            args.image_tag,
            args.namespace,
            args.timeout
        )
    elif args.command == "deploy":
        deploy_model(
            args.model_name,
            args.stage,
            args.namespace,
            args.timeout,
            args.manifest_dir
        )
    elif args.command == "smoke-test":
        smoke_test(args.model_name, args.stage, args.namespace)


if __name__ == "__main__":
    main()
